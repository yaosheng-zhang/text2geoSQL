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
from app.schema_rag import get_schema_vectorstore, get_schema_summary, has_spatial_data
from app.utils import is_safe_sql
from app.few_shot import search_similar_examples
from app.prompts import (
    ENTITY_EXTRACTION_TEMPLATE,
    RERANK_TEMPLATE,
    build_sql_generation_prompt,
    build_sql_fix_prompt,
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


# ====================== Helpers ======================
def strip_sql_markdown(raw: str) -> str:
    """Remove ```sql ... ``` wrappers from LLM output."""
    sql = raw.strip()
    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql:
        sql = sql.split("```")[1].split("```")[0].strip()
    return sql


def _classify_error(err_msg: str) -> str:
    """粗略将 PostgreSQL 错误分类，用于帮助 LLM 聚焦修复方向。"""
    e = err_msg.lower()
    if "column" in e and ("does not exist" in e or "unknown" in e):
        return "列名错误"
    if "relation" in e and "does not exist" in e:
        return "表名错误"
    if "function" in e and "does not exist" in e:
        return "函数不存在"
    if "operator does not exist" in e or "invalid input syntax" in e or "cast" in e:
        return "类型不匹配"
    if "syntax error" in e:
        return "语法错误"
    if "join" in e:
        return "JOIN 错误"
    return "未分类错误"


# ====================== Agent Nodes ======================

def entity_extractor(state: dict):
    """Node 1: Schema-aware 实体提取。

    注入 schema 摘要，让 LLM 基于真实表/列结构提取实体，而不是凭空猜测。
    """
    logger.info("[1] Entity Extractor (schema-aware) start")
    start = time.time()

    try:
        schema_summary = get_schema_summary(max_cols_per_table=8)
    except Exception as e:
        logger.warning("[1] 获取 schema summary 失败，回退为无上下文模式: %s", e)
        schema_summary = "（暂无 schema 信息）"

    prompt_template = PromptTemplate(
        template=ENTITY_EXTRACTION_TEMPLATE,
        input_variables=["query", "schema_summary"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()}
    )

    prompt = prompt_template.format(query=state['query'], schema_summary=schema_summary)
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

    # groundable 类型：value/keyword 需要匹配具体值；status_condition 也要匹配枚举值
    # （如"已完成" → status='completed'）；numeric_condition / date_condition 跳过
    GROUNDABLE_TYPES = ("value", "keyword", "status_condition")
    groundable = [e for e in entities if e.get("entity_type") in GROUNDABLE_TYPES]
    if not groundable:
        logger.info("[2] 所有实体均为数值/日期条件，跳过值匹配")
        return {"grounded_entities": entities}

    originals = [ent.get("original", "") for ent in groundable]
    all_embeddings = bge_embedding_model.encode(originals, normalize_embeddings=True, batch_size=32)
    logger.info("[2] Batch encoded %d entity embeddings", len(originals))

    emb_map = {groundable[i].get("original", ""): all_embeddings[i] for i in range(len(groundable))}

    with get_connection() as conn:
        for i, ent in enumerate(entities):
            original = ent.get("original", "")
            etype = ent.get("entity_type", "value")

            # 非 groundable 类型实体不做值匹配，原样保留
            if etype not in GROUNDABLE_TYPES:
                grounded.append({
                    "original": original,
                    "canonical": original,
                    "entity_type": etype,
                    "confidence": 1.0,
                })
                continue

            aliases = ent.get("aliases", [])
            search_terms = [original] + aliases
            logger.info("[2.%d] Entity: '%s' (type=%s, aliases=%s)", i + 1, original, etype, aliases)

            query_emb = emb_map[original].tolist()
            query_emb_str = f"[{','.join(map(str, query_emb))}]"

            try:
                with conn.cursor() as cur:
                    if len(search_terms) == 1:
                        text_sim_sql = "similarity(raw_value, %s)"
                    else:
                        sim_clauses = ", ".join(["similarity(raw_value, %s)"] * len(search_terms))
                        text_sim_sql = f"GREATEST({sim_clauses})"

                    unified_sql = f"""
                        WITH candidates AS (
                            (
                                SELECT
                                    raw_value, source_table, source_column,
                                    (embedding <=> %s::vector) as vec_distance,
                                    {text_sim_sql} as text_sim
                                FROM value_embeddings
                                ORDER BY embedding <=> %s::vector
                                LIMIT 30
                            )
                            UNION
                            (
                                SELECT
                                    raw_value, source_table, source_column,
                                    (embedding <=> %s::vector) as vec_distance,
                                    {text_sim_sql} as text_sim
                                FROM value_embeddings
                                WHERE raw_value = ANY(%s)
                                LIMIT 10
                            )
                        )
                        SELECT
                            raw_value, source_table, source_column,
                            vec_distance, text_sim,
                            (0.7 * GREATEST(1.0 - vec_distance, 0.0) + 0.3 * text_sim) as hybrid_score
                        FROM candidates
                        ORDER BY hybrid_score DESC
                        LIMIT 5
                    """

                    params = [
                        query_emb_str, *search_terms, query_emb_str,
                        query_emb_str, *search_terms, search_terms,
                    ]

                    cur.execute(unified_sql, params)
                    candidates = cur.fetchall()

            except Exception as e:
                logger.error("[2.%d] DB query failed: %s", i + 1, e)
                candidates = []

            # LLM rerank 当 top-1/2 分数接近
            if candidates and len(candidates) > 1:
                top1_score = candidates[0][5]
                top2_score = candidates[1][5]

                if abs(top1_score - top2_score) < 0.10:
                    logger.info("[2.%d] Scores close (%.3f vs %.3f), triggering LLM rerank",
                                i + 1, top1_score, top2_score)
                    candidates_str = "\n".join([
                        f"{idx + 1}. {c[0]} (source: {c[1]}.{c[2]}, score: {c[5]:.3f})"
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

            if candidates and candidates[0][5] > 0.5:
                best = {
                    "original": original,
                    "canonical": candidates[0][0],
                    "table": candidates[0][1],
                    "column": candidates[0][2],
                    "entity_type": etype,
                    "confidence": candidates[0][5]
                }
                logger.info("[2.%d] Mapped: '%s' -> '%s' (score=%.3f, vec_dist=%.3f, text_sim=%.3f)",
                            i + 1, original, best["canonical"],
                            candidates[0][5], candidates[0][3], candidates[0][4])
            else:
                best = {"original": original, "canonical": original, "entity_type": etype, "confidence": 0.5}
                logger.info("[2.%d] No match, fallback to original: '%s'", i + 1, original)

            grounded.append(best)

    logger.info("[2] Dynamic Grounding done (%.2fs), %d entities processed",
                time.time() - start, len(grounded))
    return {"grounded_entities": grounded}


def schema_retriever(state: AgentState):
    """Node 3: Entity-enriched schema retrieval。

    把已 grounding 到的表名/列名/canonical 值拼接到 query 后，提高检索召回率。
    """
    logger.info("[3] Schema Retriever start")
    start = time.time()

    vectorstore = get_schema_vectorstore()

    # Entity enrichment: 把 grounded entities 中的 table/column/canonical 拼到 query
    enrichment_tokens = []
    for ent in state.get("grounded_entities", []) or []:
        if ent.get("table"):
            enrichment_tokens.append(ent["table"])
        if ent.get("column"):
            enrichment_tokens.append(ent["column"])
        if ent.get("canonical") and ent.get("canonical") != ent.get("original"):
            enrichment_tokens.append(str(ent["canonical"]))

    enriched_query = state['query']
    if enrichment_tokens:
        enriched_query = f"{state['query']} {' '.join(set(enrichment_tokens))}"
        logger.info("[3] Enriched query tokens: %s", list(set(enrichment_tokens))[:10])

    docs = vectorstore.similarity_search(enriched_query, k=15)
    schema_str = "\n\n".join([doc.page_content for doc in docs])

    # 打印检索到的文档内容
    for idx, doc in enumerate(docs):
        logger.info("=== 检索到文档 [%d] (table=%s, type=%s) ===\n%s\n=== 文档 [%d] 结束 ===",
                    idx, doc.metadata.get("table"), doc.metadata.get("type"), doc.page_content, idx)

    logger.info("[3] Retrieved %d schema docs", len(docs))
    logger.info("[3] Schema Retriever done (%.2fs)", time.time() - start)
    return {"relevant_schema": schema_str}


def sql_planner_generator(state: AgentState):
    """Node 4: Plan + generate SQL，注入历史 few-shot 示例（Vanna 风格闭环）。"""
    logger.info("[4] SQL Planner+Generator start")
    start = time.time()

    grounded_entities = state.get("grounded_entities", [])
    grounded_str = json.dumps(grounded_entities, ensure_ascii=False)

    if not grounded_entities:
        logger.warning("[4] No grounded entities, SQL generation may be inaccurate")

    # 检索相似历史查询
    few_shots = search_similar_examples(state['query'], top_k=3, min_similarity=0.8)

    # 检测数据库是否有空间数据（决定是否注入 PostGIS 规则）
    try:
        spatial = has_spatial_data()
    except Exception:
        spatial = False

    system_prompt = build_sql_generation_prompt(
        relevant_schema=state.get('relevant_schema', ''),
        grounded_entities=grounded_str,
        has_spatial=spatial,
        few_shots=few_shots,
    )

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['query'])
    ])
    sql = strip_sql_markdown(resp.content)

    logger.info("[4] SQL Planner+Generator done (%.2fs, has_spatial=%s, few_shots=%d)",
                time.time() - start, spatial, len(few_shots))
    logger.info("[4] Generated SQL:\n%s", sql)

    return {"sql": sql}


def sql_reviewer(state: AgentState):
    """Node 5: EXPLAIN 预校验 + 迭代修复（最多 2 轮）+ 实际执行。"""
    logger.info("[5] SQL Reviewer start")
    start = time.time()
    sql = state.get("sql", "")

    if not sql:
        logger.error("[5] SQL is empty")
        return {"error": "SQL 为空", "final_sql": ""}

    current_sql = strip_sql_markdown(sql)
    if not current_sql:
        return {"error": "清理后 SQL 为空", "final_sql": ""}

    if not is_safe_sql(current_sql):
        logger.warning("[5] SQL safety check failed")
        return {"error": "生成的 SQL 不安全", "final_sql": current_sql}

    try:
        spatial = has_spatial_data()
    except Exception:
        spatial = False

    relevant_schema = state.get("relevant_schema", "")
    error_history: List[str] = []
    MAX_FIX_ROUNDS = 2

    for attempt in range(MAX_FIX_ROUNDS + 1):  # 1 次初始 + 最多 2 次修复
        # === 1. EXPLAIN 预校验（语法 + schema 有效性）===
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '5s'")
                    cur.execute(f"EXPLAIN {current_sql}")
                    cur.fetchall()
            logger.info("[5.%d] EXPLAIN passed", attempt)
        except Exception as e:
            err_msg = f"[{_classify_error(str(e))}] {str(e)}"
            logger.warning("[5.%d] EXPLAIN failed: %s", attempt, err_msg[:200])

            if attempt >= MAX_FIX_ROUNDS:
                logger.error("[5] 已达最大修复轮次，放弃")
                return {"error": str(e), "final_sql": current_sql}

            error_history.insert(0, err_msg)
            current_sql = _fix_sql(current_sql, error_history, relevant_schema, spatial)
            if not current_sql or not is_safe_sql(current_sql):
                return {"error": "修复后的 SQL 无效或不安全",
                        "final_sql": current_sql or "",
                        "error_history": error_history}
            continue  # 继续下一轮 EXPLAIN

        # === 2. 实际执行 ===
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '10s'")
                    cur.execute(current_sql)
                    col_names = [desc[0] for desc in cur.description] if cur.description else []
                    rows = cur.fetchall()

            logger.info("[5] SQL executed OK, %d rows (%.2fs, attempts=%d)",
                        len(rows), time.time() - start, attempt + 1)
            return {
                "final_sql": current_sql,
                "error": None,
                "query_results": rows,
                "column_names": col_names,
            }

        except Exception as e:
            err_msg = f"[{_classify_error(str(e))}] {str(e)}"
            logger.error("[5.%d] 执行失败: %s", attempt, err_msg[:200])

            if attempt >= MAX_FIX_ROUNDS:
                logger.error("[5] 已达最大修复轮次，放弃")
                return {"error": str(e), "final_sql": current_sql}

            error_history.insert(0, err_msg)
            current_sql = _fix_sql(current_sql, error_history, relevant_schema, spatial)
            if not current_sql or not is_safe_sql(current_sql):
                return {"error": "修复后的 SQL 无效或不安全",
                        "final_sql": current_sql or ""}

    # 不应到这里
    return {"error": "异常的修复流程", "final_sql": current_sql}


def _fix_sql(sql: str, error_history: List[str], relevant_schema: str, spatial: bool) -> str:
    """调用 LLM 修复 SQL。"""
    fix_prompt = build_sql_fix_prompt(
        sql=sql,
        error_history=error_history,
        relevant_schema=relevant_schema,
        has_spatial=spatial,
    )
    logger.info("[Fix] Fix prompt (first 500):\n%s", fix_prompt[:500])

    try:
        fixed_raw = llm.invoke([HumanMessage(content=fix_prompt)]).content.strip()
        fixed = strip_sql_markdown(fixed_raw)
        logger.info("[Fix] Fixed SQL:\n%s", fixed)
        return fixed
    except Exception as e:
        logger.error("[Fix] LLM 修复调用失败: %s", e)
        return sql  # 返回原 SQL，让外层判断放弃


# ====================== LangGraph DAG Workflow ======================
# Topology:
#                 ┌→ extractor → grounding ───┐
# START → fork → │                             ├→ sql_planner_generator → reviewer → END
#                 └→ schema_retriever ────────┘
#
# 注意：schema_retriever 现在会用 grounded_entities 做 enrichment，但 grounding 也写该 key，
# 所以图结构保持不变（retriever 读的是 merge 之后的状态）

def _fork(state: AgentState):
    return {}


def _merge(state: AgentState):
    return {}


workflow = StateGraph(AgentState)

workflow.add_node("fork", _fork)
workflow.add_node("extractor", entity_extractor)
workflow.add_node("grounding", dynamic_grounding)
workflow.add_node("retriever", schema_retriever)
workflow.add_node("merge", _merge)
workflow.add_node("generator", sql_planner_generator)
workflow.add_node("reviewer", sql_reviewer)

workflow.set_entry_point("fork")
workflow.add_edge("fork", "extractor")
workflow.add_edge("extractor", "grounding")
workflow.add_edge("grounding", "retriever")  # 改为串行，让 retriever 能用 grounded_entities
workflow.add_edge("retriever", "generator")
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
