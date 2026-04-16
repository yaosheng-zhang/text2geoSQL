from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional
import json
import logging
import time

from app.config import llm, bge_embedding_model
from app.db import get_connection
from app.schema_rag import get_schema_vectorstore
from app.utils import is_safe_sql
from app.prompts import (
    ENTITY_EXTRACTION_TEMPLATE,
    RERANK_TEMPLATE,
    SQL_GENERATION_SYSTEM_TEMPLATE,
    SQL_FIX_TEMPLATE,
)

logger = logging.getLogger(__name__)


# ====================== State ======================
class AgentState(TypedDict):
    query: str
    grounded_entities: List[dict]
    relevant_schema: str
    sql: str
    final_sql: str
    error: Optional[str]
    query_results: Optional[List[tuple]]
    column_names: Optional[List[str]]


# ====================== Pydantic models for entity extraction ======================
class ExtractedEntity(BaseModel):
    original: str = Field(description="用户查询中出现的原始文本")
    entity_type: str = Field(description="实体类型")
    aliases: List[str] = Field(default=[], description="可能的别名、简称、全称")


class EntityList(BaseModel):
    entities: List[ExtractedEntity] = Field(description="提取出的实体列表")


json_parser = JsonOutputParser(pydantic_object=EntityList)


# ====================== Helper: strip markdown code fences ======================
def strip_sql_markdown(raw: str) -> str:
    """Remove ```sql ... ``` wrappers from LLM output."""
    sql = raw.strip()
    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql:
        sql = sql.split("```")[1].split("```")[0].strip()
    return sql


# ====================== Agent Nodes ======================

def entity_extractor(state: dict):
    """Node 1: Extract entities from user query."""
    logger.info("[1] Entity Extractor start")
    start = time.time()

    prompt_template = PromptTemplate(
        template=ENTITY_EXTRACTION_TEMPLATE,
        input_variables=["query"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()}
    )

    prompt = prompt_template.format(query=state['query'])
    resp = llm.invoke([HumanMessage(content=prompt)])
    raw_content = resp.content.strip()

    logger.info("[1] LLM raw output (first 500): %s", raw_content[:500])

    try:
        parsed = json_parser.parse(raw_content)
        entities = parsed.get("entities", [])
        logger.info("[1] Extracted %d entities", len(entities))
        for ent in entities[:5]:
            logger.info("     - %s (%s)", ent.get('original'), ent.get('entity_type'))
    except Exception as e:
        logger.error("[1] Parser failed: %s", e)
        entities = []

    logger.info("[1] Entity Extractor done (%.2fs)", time.time() - start)
    return {"grounded_entities": entities}


def dynamic_grounding(state: AgentState):
    """Node 2: Ground entities to canonical DB values using hybrid search."""
    logger.info("[2] Dynamic Grounding start")
    start = time.time()
    grounded = []
    entities = state.get("grounded_entities", [])

    if not entities:
        logger.info("[2] No entities to ground, skipping")
        return {"grounded_entities": []}

    logger.info("[2] Entities to process: %d", len(entities))

    # --- Batch encode all entity originals at once ---
    originals = [ent.get("original", "") for ent in entities]
    all_embeddings = bge_embedding_model.encode(originals, normalize_embeddings=True, batch_size=32)
    logger.info("[2] Batch encoded %d entity embeddings", len(originals))

    # --- Single DB connection for all entities ---
    with get_connection() as conn:
        for i, ent in enumerate(entities):
            original = ent.get("original", "")
            aliases = ent.get("aliases", [])
            etype = ent.get("entity_type", "poi_category")

            search_terms = [original] + aliases
            logger.info("[2.%d] Entity: '%s' (type=%s, aliases=%s)", i + 1, original, etype, aliases)

            query_emb = all_embeddings[i].tolist()
            query_emb_str = f"[{','.join(map(str, query_emb))}]"

            try:
                with conn.cursor() as cur:
                    # Build dynamic text similarity expression
                    if len(search_terms) == 1:
                        text_sim_sql = "similarity(raw_value, %s)"
                    else:
                        sim_clauses = ", ".join(["similarity(raw_value, %s)"] * len(search_terms))
                        text_sim_sql = f"GREATEST({sim_clauses})"

                    # Dual-path recall: vector HNSW + exact alias match
                    unified_sql = f"""
                        WITH candidates AS (
                            (
                                SELECT
                                    raw_value, source_table, source_column, entity_type,
                                    (embedding <=> %s::vector) as vec_distance,
                                    {text_sim_sql} as text_sim
                                FROM value_embeddings
                                ORDER BY embedding <=> %s::vector
                                LIMIT 30
                            )
                            UNION
                            (
                                SELECT
                                    raw_value, source_table, source_column, entity_type,
                                    (embedding <=> %s::vector) as vec_distance,
                                    {text_sim_sql} as text_sim
                                FROM value_embeddings
                                WHERE raw_value = ANY(%s)
                                LIMIT 10
                            )
                        )
                        SELECT
                            raw_value, source_table, source_column, entity_type,
                            vec_distance, text_sim,
                            (0.7 * GREATEST(1.0 - vec_distance, 0.0) + 0.3 * text_sim +
                             CASE WHEN entity_type = %s THEN 0.05 ELSE 0.0 END) as hybrid_score
                        FROM candidates
                        ORDER BY hybrid_score DESC
                        LIMIT 5
                    """

                    params = [
                        # CTE path A (vector)
                        query_emb_str,
                        *search_terms,
                        query_emb_str,
                        # CTE path B (exact alias)
                        query_emb_str,
                        *search_terms,
                        search_terms,
                        # outer scoring
                        etype
                    ]

                    cur.execute(unified_sql, params)
                    candidates = cur.fetchall()

            except Exception as e:
                logger.error("[2.%d] DB query failed: %s", i + 1, e)
                candidates = []

            # LLM rerank when top scores are close
            if candidates and len(candidates) > 1:
                top1_score = candidates[0][6]
                top2_score = candidates[1][6]

                if abs(top1_score - top2_score) < 0.10:
                    logger.info("[2.%d] Scores close (%.3f vs %.3f), triggering LLM rerank",
                                i + 1, top1_score, top2_score)
                    candidates_str = "\n".join([
                        f"{idx + 1}. {c[0]} (source: {c[1]}.{c[2]}, db_type: {c[3]}, score: {c[6]:.3f})"
                        for idx, c in enumerate(candidates[:3])
                    ])

                    rerank_prompt = RERANK_TEMPLATE.format(
                        original=original,
                        candidates_str=candidates_str,
                        max_choice=min(3, len(candidates))
                    )
                    try:
                        resp = llm.invoke([HumanMessage(content=rerank_prompt)])
                        choice = int(resp.content.strip())
                        if 1 <= choice <= len(candidates):
                            selected = candidates.pop(choice - 1)
                            candidates.insert(0, selected)
                            logger.info("[2.%d] LLM rerank chose: %d", i + 1, choice)
                    except Exception as e:
                        logger.warning("[2.%d] LLM rerank failed: %s", i + 1, e)

            # Final decision
            if candidates and candidates[0][6] > 0.5:
                best = {
                    "original": original,
                    "canonical": candidates[0][0],
                    "table": candidates[0][1],
                    "column": candidates[0][2],
                    "entity_type": candidates[0][3],
                    "confidence": candidates[0][6]
                }
                logger.info("[2.%d] Mapped: '%s' -> '%s' (score=%.3f, vec_dist=%.3f, text_sim=%.3f)",
                            i + 1, original, best["canonical"],
                            candidates[0][6], candidates[0][4], candidates[0][5])
            else:
                best = {"original": original, "canonical": original, "confidence": 0.5}
                logger.info("[2.%d] No match, fallback to original: '%s'", i + 1, original)

            grounded.append(best)

    logger.info("[2] Dynamic Grounding done (%.2fs), %d entities processed",
                time.time() - start, len(grounded))
    return {"grounded_entities": grounded}


def schema_retriever(state: AgentState):
    """Node 3: Retrieve relevant schema docs via vector search."""
    logger.info("[3] Schema Retriever start")
    start = time.time()

    vectorstore = get_schema_vectorstore()
    docs = vectorstore.similarity_search(state['query'], k=8)

    schema_str = "\n\n".join([doc.page_content for doc in docs])

    logger.info("[3] Retrieved %d schema docs", len(docs))
    logger.info("[3] Schema Retriever done (%.2fs)", time.time() - start)
    return {"relevant_schema": schema_str}


def sql_planner_generator(state: AgentState):
    """Node 4: Plan + generate SQL in a single LLM call (merged planner & generator)."""
    logger.info("[4] SQL Planner+Generator start")
    start = time.time()

    grounded_entities = state.get("grounded_entities", [])
    grounded_str = json.dumps(grounded_entities, ensure_ascii=False)

    if not grounded_entities:
        logger.warning("[4] No grounded entities, SQL generation may be inaccurate")

    system_prompt = SQL_GENERATION_SYSTEM_TEMPLATE.format(
        relevant_schema=state.get('relevant_schema', ''),
        grounded_entities=grounded_str
    )

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['query'])
    ])
    sql = strip_sql_markdown(resp.content)

    logger.info("[4] SQL Planner+Generator done (%.2fs)", time.time() - start)
    logger.info("[4] Generated SQL:\n%s", sql)

    return {"sql": sql}


def sql_reviewer(state: AgentState):
    """Node 5: Validate, execute SQL, and auto-fix on error."""
    logger.info("[5] SQL Reviewer start")
    start = time.time()
    sql = state.get("sql", "")

    if not sql:
        logger.error("[5] SQL is empty")
        return {"error": "SQL 为空", "final_sql": ""}

    sql_to_execute = strip_sql_markdown(sql)
    if not sql_to_execute:
        logger.error("[5] SQL empty after cleanup")
        return {"error": "清理后 SQL 为空", "final_sql": ""}

    if not is_safe_sql(sql_to_execute):
        logger.warning("[5] SQL safety check failed")
        return {"error": "生成的 SQL 不安全", "final_sql": sql_to_execute}

    try:
        logger.info("[5] SQL safety check passed, executing...")
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = '10s'; {sql_to_execute}")
                col_names = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()

        logger.info("[5] SQL executed OK, %d rows (%.2fs)", len(rows), time.time() - start)
        return {
            "final_sql": sql_to_execute,
            "error": None,
            "query_results": rows,
            "column_names": col_names,
        }

    except Exception as e:
        logger.error("[5] SQL execution failed: %s", e)
        logger.error("[5] SQL was:\n%s", sql_to_execute)

        # Auto-fix via LLM
        fix_prompt = SQL_FIX_TEMPLATE.format(sql=sql_to_execute, error=str(e))
        logger.info("[5] Fix prompt:\n%s", fix_prompt)

        fixed_raw = llm.invoke([HumanMessage(content=fix_prompt)]).content.strip()
        logger.info("[5] Fixed raw:\n%s", fixed_raw)

        fixed = strip_sql_markdown(fixed_raw)
        logger.info("[5] Auto-fixed SQL:\n%s", fixed)
        return {"error": str(e), "sql": fixed}


# ====================== LangGraph DAG Workflow ======================
# Topology:
#                 ┌→ extractor → grounding ───┐
# START → fork → │                             ├→ sql_planner_generator → reviewer → END
#                 └→ schema_retriever ────────┘

def _fork(state: AgentState):
    """Pass-through node that fans out to parallel branches."""
    return {}


def _merge(state: AgentState):
    """Pass-through node that joins parallel branches."""
    return {}


workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("fork", _fork)
workflow.add_node("extractor", entity_extractor)
workflow.add_node("grounding", dynamic_grounding)
workflow.add_node("retriever", schema_retriever)
workflow.add_node("merge", _merge)
workflow.add_node("generator", sql_planner_generator)
workflow.add_node("reviewer", sql_reviewer)

# Edges: fan-out from fork
workflow.set_entry_point("fork")
workflow.add_edge("fork", "extractor")
workflow.add_edge("fork", "retriever")

# Entity branch: extractor → grounding → merge
workflow.add_edge("extractor", "grounding")
workflow.add_edge("grounding", "merge")

# Schema branch: retriever → merge
workflow.add_edge("retriever", "merge")

# After merge: generate → review → end
workflow.add_edge("merge", "generator")
workflow.add_edge("generator", "reviewer")
workflow.add_edge("reviewer", END)

graph = workflow.compile()


async def run_text2geosql(query: str) -> dict:
    """Public entry point. Returns dict with sql, query_results, column_names, error."""
    logger.info("=" * 70)
    logger.info("[Text2GeoSQL] Pipeline start")
    logger.info("  Query: %s", query)
    overall_start = time.time()

    try:
        result = await graph.ainvoke({"query": query})

        sql = result.get("final_sql") or result.get("sql") or ""
        error = result.get("error")
        elapsed = time.time() - overall_start

        if error:
            logger.warning("[Text2GeoSQL] Completed with error: %s", error)
        else:
            logger.info("[Text2GeoSQL] Pipeline success (%.2fs)", elapsed)
        logger.info("  Final SQL: %s", sql[:300] if sql else "None")
        logger.info("=" * 70)

        return {
            "sql": sql,
            "error": error,
            "query_results": result.get("query_results"),
            "column_names": result.get("column_names"),
        }

    except Exception as e:
        elapsed = time.time() - overall_start
        logger.error("[Text2GeoSQL] Pipeline error (%.2fs): %s", elapsed, e, exc_info=True)
        logger.info("=" * 70)
        raise
