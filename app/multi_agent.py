from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, List, Optional
import psycopg
import json
import logging
import time
import re
from app.config import llm, bge_embedding_model, DB_URL
from app.schema_rag import get_schema_vectorstore
from app.utils import is_safe_sql

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    query: str
    grounded_entities: List[dict]
    relevant_schema: str
    logical_plan: str
    sql: str
    final_sql: str
    error: Optional[str]

# ==================== Agent 节点（带详细日志） ====================
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
# 定义实体结构
class ExtractedEntity(BaseModel):
    original: str = Field(description="用户查询中出现的原始文本")
    entity_type: str = Field(description="实体类型: ward, poi_category, poi_name, rating, distance, infrastructure_name, infrastructure_type")
    aliases: List[str] = Field(default=[], description="可能的别名、简称、全称（用于提高匹配率）")

class EntityList(BaseModel):
    entities: List[ExtractedEntity] = Field(description="提取出的实体列表")

json_parser = JsonOutputParser(pydantic_object=EntityList)

def entity_extractor(state: dict):
    logger.info("【1】实体提取 Agent 开始")
    start = time.time()

    prompt_template = PromptTemplate(
    template="""你是一个极其严谨、精确的**数据库实体提取专家**，专门服务于 Text-to-SQL 任务。

    任务：
    从用户查询中提取所有可能用于构造 SQL WHERE、JOIN、GROUP BY、ORDER BY、HAVING 等子句的关键元素，包括但不限于：
    - 具体的实体值（人名、产品名、地点、ID、名称等）
    - 类别/类型（产品类别、建筑类型、部门、状态等）
    - 时间/日期条件
    - 状态条件（已竣工、在建、已完成、活跃等）
    - 基础设施、建筑、POI 等领域相关概念
    - 任何可能对应数据库表中列（column）或值的词语

    **允许提取的实体类型（entity_type）**（必须从以下列表中选择最合适的）：
    - table_name：表名或表别称
    - column_name：列名或字段描述
    - poi_name：具体兴趣点/名称（如东京铁塔、某个产品名、客户名）
    - poi_category：类别（如餐厅、建筑、高楼、电子产品）
    - infrastructure_name：基础设施名称
    - infrastructure_type：基础设施类型（如建筑、桥梁、道路）
    - ward：行政区/区域
    - person_name：人名
    - id_value：ID、编号、代码
    - numeric_condition：数值条件（如高度>100米、价格>5000）
    - date_condition：日期/时间条件
    - status_condition：状态条件（如已竣工、已完成、在售）
    - other_condition：其他过滤条件
    - keyword：其他重要关键词（兜底类型）

    **输出要求**：
    - 必须严格返回合法的 JSON，不要添加任何解释、markdown 或额外文字。
    - 为每个实体生成 1-4 个合理的 aliases（别名、简称、全称、英文、同义词），用于后续模糊匹配和 SQL 值查找。
    - 如果查询是条件描述（如“已竣工且高度超过100米的建筑”），也要提取：
      - “建筑” → infrastructure_type 或 poi_category
      - “高度超过100米” → numeric_condition
      - “已竣工” → status_condition
    - 即使没有非常具体的名称，也要提取类别和条件，**禁止返回空列表**。

    查询: {query}

    {format_instructions}

    **示例输出格式**（仅供参考，不要复制）：
    {{
      "entities": [
        {{
          "original": "东京铁塔",
          "entity_type": "poi_name",
          "aliases": ["东京塔", "Tokyo Tower", "铁塔"]
        }},
        {{
          "original": "高度超过100米",
          "entity_type": "numeric_condition",
          "aliases": [">100米", "height > 100", "100米以上"]
        }},
        {{
          "original": "已竣工",
          "entity_type": "status_condition",
          "aliases": ["completed", "已完成", "竣工"]
        }},
        {{
          "original": "建筑",
          "entity_type": "infrastructure_type",
          "aliases": ["building", "大楼", "高楼"]
        }}
      ]
    }}
    """,
        input_variables=["query"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()}
    )

    prompt = prompt_template.format(query=state['query'])
    
    resp = llm.invoke([HumanMessage(content=prompt)])
    raw_content = resp.content.strip()

    logger.info(f"1.2 LLM 原始输出 (前500字): {raw_content}")

    try:
        parsed = json_parser.parse(raw_content)
        entities = parsed.get("entities", [])
        logger.info(f"1.3 实体提取成功 → 共 {len(entities)} 个实体")
        if entities:
            for ent in entities[:3]:
                logger.info(f"     - {ent.get('original')} ({ent.get('entity_type')})")
    except Exception as e:
        logger.error(f"1.3 Parser 解析失败: {e}")
        logger.error(f"     原始输出: {raw_content}")
        entities = []

    logger.info(f"【1】实体提取 Agent 完成 ({time.time() - start:.2f}s)")
    return {"grounded_entities": entities}


def dynamic_grounding(state: AgentState):
    logger.info("【2】动态 Grounding Agent 开始（一元化真·混合搜索）")
    start = time.time()
    grounded = []
    entities = state.get("grounded_entities", [])

    logger.info("2.1 需要处理的实体数量: %d", len(entities))

    for i, ent in enumerate(entities, 1):
        original = ent.get("original", "")
        aliases = ent.get("aliases", [])
        etype = ent.get("entity_type", "poi_category")

        # 搜索词列表：原始词 + 所有别名
        search_terms = [original] + aliases
        logger.info("2.2.%d 处理实体: '%s' (type=%s, 别名: %s)", i, original, etype, aliases)

        # 🚀 核心改动 1：坚决不做向量平均！
        # 只使用用户原始输入提取向量，保障最纯粹的语义不被幻觉别名稀释
        query_emb = bge_embedding_model.encode(original, normalize_embeddings=True).tolist()
        query_emb_str = f"[{','.join(map(str, query_emb))}]"

        try:
            with psycopg.connect(DB_URL) as conn:
                with conn.cursor() as cur:
                    
                    # 动态构建文本相似度打分逻辑
                    if len(search_terms) == 1:
                        text_sim_sql = "similarity(raw_value, %s)"
                    else:
                        sim_clauses = ", ".join(["similarity(raw_value, %s)"] * len(search_terms))
                        text_sim_sql = f"GREATEST({sim_clauses})"

                    # 🚀 核心改动 2：双路召回 CTE 架构 (合并为一条 SQL 解决所有问题)
                    # 路线 A: 向量索引极速搜索 (寻找语义相近)
                    # 路线 B: 别名字面精准命中直通车 (保证你推理的别名 100% 进候选池)
                    unified_sql = f"""
                        WITH candidates AS (
                            -- 路线 A：纯向量召回 (HNSW索引, 容错率强)
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
                            -- 路线 B：精确/别名召回直通车 (哪怕向量偏了，只要别名对上就必抓进来)
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
                            -- 🚀 核心改动 3：科学的平滑计分公式
                            -- 向量为主(0.7) + 文本为辅(0.3) + 同类型奖励加成(0.05)
                            (0.7 * (1 - vec_distance) + 0.3 * text_sim + 
                             CASE WHEN entity_type = %s THEN 0.05 ELSE 0.0 END) as hybrid_score
                        FROM candidates
                        ORDER BY hybrid_score DESC
                        LIMIT 5
                    """
                    
                    # 严谨的参数绑定顺序（千万不能乱）
                    params = [
                        # === CTE路线 A (向量) ===
                        query_emb_str,
                        *search_terms,
                        query_emb_str,
                        # === CTE路线 B (文本直通车) ===
                        query_emb_str,
                        *search_terms,
                        search_terms,   # ANY(%s) 会自动把 Python List 转为 Postgres 数组
                        # === 外层打分 ===
                        etype
                    ]
                    
                    cur.execute(unified_sql, params)
                    candidates = cur.fetchall()

        except Exception as e:
            logger.error("2.3 数据库查询失败: %s", e)
            candidates = []

        # ==================== LLM 重排（处理最后一点歧义） ====================
        if candidates and len(candidates) > 1:
            top1_score = candidates[0][6]
            top2_score = candidates[1][6]

            if abs(top1_score - top2_score) < 0.10: # 放宽差异容忍度
                logger.info("2.4.%d 分数接近 (%.3f vs %.3f)，启动 LLM 重排...", i, top1_score, top2_score)
                candidates_str = "\n".join([
                    f"{idx+1}. {c[0]} (来源: {c[1]}.{c[2]}, 库内类型: {c[3]}, 综合分: {c[6]:.3f})"
                    for idx, c in enumerate(candidates[:3])
                ])

                rerank_prompt = f"""用户原始查询提及: "{original}"
我们找出了以下候选数据库标准命名:
{candidates_str}

判断哪一个最符合用户意图？只返回数字编号（1-3），无需解释。都不行返回 0。"""
                try:
                    resp = llm.invoke([HumanMessage(content=rerank_prompt)])
                    choice = int(resp.content.strip())
                    if 1 <= choice <= len(candidates):
                        # 把选中的提升到第一位
                        selected_candidate = candidates.pop(choice - 1)
                        candidates.insert(0, selected_candidate)
                        logger.info("2.4.%d LLM 重排完成，强制选择: %d", i, choice)
                except Exception as e:
                    logger.warning("2.4.%d LLM 重排失败: %s", i, e)

        # ==================== 最终决策 ====================
        if candidates and candidates[0][6] > 0.5:
            best = {
                "original": original,
                "canonical": candidates[0][0],
                "table": candidates[0][1],
                "column": candidates[0][2],
                "entity_type": candidates[0][3],
                "confidence": candidates[0][6]
            }
            logger.info("2.5.%d ✓ 映射成功: '%s' → '%s' (得分: %.3f | 向量距: %.3f | 文本相似: %.3f | DB类型: %s)",
                        i, original, best["canonical"], candidates[0][6], candidates[0][4], candidates[0][5], candidates[0][3])
        else:
            best = {"original": original, "canonical": original, "confidence": 0.5}
            logger.info("2.5.%d ⚠ 映射失败，退化为原始值: '%s'", i, original)

        grounded.append(best)

    logger.info("【2】动态 Grounding Agent 完成 (%.2fs)，共处理 %d 个实体", time.time() - start, len(grounded))
    return {"grounded_entities": grounded}


def schema_retriever(state: AgentState):
    logger.info("【3】Schema RAG 检索 Agent 开始")
    start = time.time()

    vectorstore = get_schema_vectorstore()
    docs = vectorstore.similarity_search(state['query'], k=8)

    schema_str = "\n\n".join([doc.page_content for doc in docs])

    logger.info("3.1 检索到 %d 个 Schema 文档", len(docs))
    logger.info("3.2 Schema 内容预览:\n%s", schema_str)

    logger.info("【3】Schema RAG 检索 Agent 完成 (%.2fs)", time.time() - start)
    return {"relevant_schema": schema_str}


def logic_planner(state: AgentState):
    logger.info("【4】逻辑规划 Agent 开始")
    start = time.time()

    grounded_entities = state.get("grounded_entities", [])
    grounded_str = json.dumps(grounded_entities, ensure_ascii=False)

    if not grounded_entities:
        logger.warning("⚠️  没有 grounded 实体，规划可能不准确")
        logger.warning("   原始查询: %s", state['query'])

    prompt = f"""你是空间数据库查询规划专家。
已知 Schema:
{state['relevant_schema']}

已 grounding 的实体:
{grounded_str}

用户查询: {state['query']}

请给出清晰的查询计划，包括需要用到的表、JOIN方式、过滤条件等。"""

    resp = llm.invoke([SystemMessage(content="你是一个专业的 PostGIS 查询规划助手。"), HumanMessage(content=prompt)])

    logger.info("4.1 逻辑规划完成 (%.2fs)，规划长度 %d 字符", time.time() - start, len(resp.content))
    logger.info("4.2 规划内容:\n%s", resp.content)

    return {"logical_plan": resp.content}


def sql_generator(state: AgentState):
    logger.info("【5】SQL 生成 Agent 开始")
    start = time.time()

    grounded_entities = state.get("grounded_entities", [])
    grounded_str = json.dumps(grounded_entities, ensure_ascii=False)

    if not grounded_entities:
        logger.warning("⚠️  没有 grounded 实体，SQL 生成可能不准确")

    system_prompt = f"""你是 PostGIS 专家，只能生成 SELECT 语句。
Schema:
{state['relevant_schema']}

Grounded Entities:
{grounded_str}

Logical Plan:
{state['logical_plan']}

严格只返回一条有效的 SQL 语句，不要任何解释，不要 ```sql 标记。"""

    resp = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=state['query'])])
    sql = resp.content.strip()

    if "```sql" in sql:
        sql = sql.split("```sql")[1].split("```")[0].strip()
    elif "```" in sql:
        sql = sql.split("```")[1].strip()

    logger.info("5.1 SQL 生成完成 (%.2fs)", time.time() - start)
    logger.info("5.2 生成的 SQL:\n%s", sql)

    return {"sql": sql}


def sql_reviewer(state: AgentState):
    logger.info("【6】SQL 审查 & 执行 Agent 开始")
    start = time.time()
    sql = state.get("sql", "")

    if not sql:
        logger.error("❌ SQL 为空，无法执行")
        return {"error": "SQL 为空", "final_sql": ""}

    # 先清理可能存在的 markdown 代码块
    sql_to_execute = re.sub(r'^```(?:sql)?\s*', '', sql, flags=re.IGNORECASE)
    sql_to_execute = re.sub(r'\s*```$', '', sql_to_execute, flags=re.IGNORECASE).strip()

    if not sql_to_execute:
        logger.error("❌ 清理后 SQL 为空")
        return {"error": "清理后 SQL 为空", "final_sql": ""}

    if not is_safe_sql(sql_to_execute):
        logger.warning("❌ SQL 安全检查失败")
        logger.warning("   SQL: %s", sql_to_execute)
        return {"error": "生成的 SQL 不安全", "final_sql": sql_to_execute}

    try:
        logger.info("✅ SQL 安全检查通过，开始执行...")
        with psycopg.connect(DB_URL, options="-c statement_timeout=10000") as conn:
            with conn.cursor() as cur:
                cur.execute(sql_to_execute)
                rows = cur.fetchall()[:10]

        logger.info("✅ SQL 执行成功，返回 %d 行数据 (%.2fs)", len(rows), time.time() - start)
        return {"final_sql": sql_to_execute, "error": None}

    except Exception as e:
        logger.error("❌ SQL 执行失败: %s", e)
        logger.error("   原始 SQL:\n%s", sql_to_execute)

        # ==================== 通用修复 Prompt ====================
        fix_prompt = f"""你是一个经验丰富的 PostgreSQL + PostGIS 数据库专家。

**原始 SQL：**
```sql
{sql_to_execute}
执行报错信息：
{str(e)}
请分析错误原因并修复SQL。
修复时请：

保留原始查询的业务意图和主要结构（SELECT 字段、JOIN、WHERE 等）
修复类型不匹配、函数调用、语法等问题（例如 GeoJSON 需要加 ::text）
只返回修正后的完整 SQL 语句，不要任何解释、不要 ```sql 标记、不要额外文字。

修复后的 SQL：
"""
        logger.info("fix_prompt:\n",fix_prompt)
        fixed_raw = llm.invoke([HumanMessage(content=fix_prompt)]).content.strip()
        logger.info("fix_prompt:\n",fixed_raw)
        fixed = re.sub(r'^(?:sql)?\s*', '', fixed_raw, flags=re.IGNORECASE)         
        fixed = re.sub(r'\s*$', '', fixed, flags=re.IGNORECASE).strip()
        logger.info("🔧 尝试自动修复后的 SQL:\n%s", fixed)
        return {"error": str(e), "sql": fixed}


# ==================== 构建 LangGraph 工作流 ====================
workflow = StateGraph(AgentState)

workflow.add_node("extractor", entity_extractor)
workflow.add_node("grounding", dynamic_grounding)
workflow.add_node("retriever", schema_retriever)
workflow.add_node("planner", logic_planner)
workflow.add_node("generator", sql_generator)
workflow.add_node("reviewer", sql_reviewer)

workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "grounding")
workflow.add_edge("grounding", "retriever")
workflow.add_edge("retriever", "planner")
workflow.add_edge("planner", "generator")
workflow.add_edge("generator", "reviewer")
workflow.add_edge("reviewer", END)

graph = workflow.compile()

async def run_text2geosql(query: str):
    """对外统一调用接口"""
    logger.info("=" * 70)
    logger.info("【Text2GeoSQL 完整流程开始】")
    logger.info("   查询: %s", query)
    overall_start = time.time()

    try:
        result = await graph.ainvoke({"query": query})
        sql = result.get("final_sql") or result.get("sql") or ""
        error = result.get("error")

        elapsed = time.time() - overall_start

        if error:
            logger.warning("❌ 流程完成但包含错误: %s", error)
            logger.warning("   生成的 SQL: %s", sql if sql else "无")
        else:
            logger.info("✅ 【Text2GeoSQL 完整流程成功结束】")
            logger.info("   耗时: %.2fs", elapsed)
            logger.info("   最终 SQL: %s", sql if sql else "无")

        logger.info("=" * 70)
        return sql

    except Exception as e:
        elapsed = time.time() - overall_start
        logger.error("❌ 【Text2GeoSQL 流程异常】 耗时 %.2fs", elapsed)
        logger.error("   错误: %s", e, exc_info=True)
        logger.info("=" * 70)
        raise