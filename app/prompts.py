# app/prompts.py - 通用 Text2SQL Prompt 模板
#
# 设计原则：
# 1. 所有模板都与具体业务领域解耦（不硬编码地理/PostGIS/产品等领域词汇）
# 2. 通过 build_* 函数动态组装 prompt，按需注入 dialect / few-shot / spatial 规则
# 3. 空间数据库（PostGIS）相关内容仅在 has_spatial=True 时追加

# ====================== 1. 实体提取（Schema-aware） ======================

ENTITY_EXTRACTION_TEMPLATE = """你是一个严谨的 Text-to-SQL **实体提取专家**。

任务：从用户查询中提取所有可能用于构造 SQL WHERE / JOIN / GROUP BY / ORDER BY / HAVING 子句的关键元素。

**参考数据库 Schema 摘要**（用于对齐实体与真实表名/列名）：
{schema_summary}

**实体类型（entity_type）**（只从以下 5 种通用类型中选择）：
- `value`：具体的值或名称（产品名、人名、地名、ID、类别名等，任何可能匹配某列值的内容）
- `numeric_condition`：数值条件（如 "高于 100"、"销量前 10"）
- `date_condition`：时间/日期条件（如 "2024 年之后"、"最近一周"）
- `status_condition`：状态/枚举条件（如 "已完成"、"在售"、"启用"）
- `keyword`：兜底关键词（其他重要词语）

**输出要求**：
- 严格返回合法 JSON，不要任何解释或 markdown 标记
- 为每个 `value` 类型实体生成 1-4 个 aliases（别名、简称、全称、同义词、英文），用于后续值匹配
- 其他类型的 aliases 可以为空列表
- 如果查询中同时包含具体值和条件（如"价格超过100元的已下架商品"），都要提取出来
- **禁止返回空列表**，至少要提取一个 keyword

**查询**: {query}

{format_instructions}

**示例**（仅供格式参考，不要照搬内容）：
{{
  "entities": [
    {{"original": "苹果手机", "entity_type": "value", "aliases": ["iPhone", "Apple phone", "苹果"]}},
    {{"original": "价格超过 1000 元", "entity_type": "numeric_condition", "aliases": ["price > 1000", ">1000元"]}},
    {{"original": "已下架", "entity_type": "status_condition", "aliases": ["offline", "下架", "停售"]}}
  ]
}}
"""


# ====================== 2. LLM Rerank（保持通用） ======================

RERANK_TEMPLATE = """用户原始查询提及: "{original}"
我们找出了以下候选数据库标准命名:
{candidates_str}

判断哪一个最符合用户意图？只返回数字编号（1-{max_choice}），无需解释。都不行返回 0。"""


# ====================== 3. SQL 生成 - 基础模板 + 动态组装 ======================

_SQL_GENERATION_BASE = """你是一个经验丰富的 **SQL 数据库查询专家**，同时是一个严谨的查询规划者。

请先在心里规划查询策略（需要哪些表、JOIN 方式、过滤条件、聚合函数），然后生成最终的 SQL。

**数据库 Schema:**
{relevant_schema}

**已匹配的实体（Grounded Entities，用于 WHERE 过滤值的规范化）:**
{grounded_entities}
{few_shot_block}{spatial_block}
**通用规则：**
1. 只生成单条 SELECT 语句（允许使用 WITH/CTE 和子查询）
2. 严格按照 Schema 中的 `Relationships (JOIN Paths)` 进行 JOIN，不要凭空创造 JOIN 关系
3. WHERE 条件的值使用 Grounded Entities 中的 `canonical` 值（而非用户原始输入）
4. 使用标准 SQL 语法（PostgreSQL 方言），保持查询简洁清晰
5. 只返回最终 SQL 语句，不要解释、不要 markdown 标记"""


_SPATIAL_BLOCK = """
**空间查询规则**（本数据库包含 geometry/geography 列）：
- 使用 PostGIS 空间函数：ST_Contains、ST_Intersects、ST_DWithin、ST_Distance、ST_Within 等
- 使用空间函数前，确保已 JOIN 包含 GEOMETRY 列的表
- 计算经纬度距离时使用 geography 类型或 ST_DWithin(geom, ..., meters)
- 返回几何数据时注意用 ST_AsText 或 ST_AsGeoJSON 转换可读格式
"""


_FEW_SHOT_PREFIX = """
**参考历史成功查询**（与当前问题相似的历史 question-SQL 对，供参考不要照搬）：
{few_shot_content}
"""


def build_sql_generation_prompt(
    relevant_schema: str,
    grounded_entities: str,
    has_spatial: bool = False,
    few_shots: list = None,
) -> str:
    """动态组装 SQL 生成 system prompt。

    Args:
        relevant_schema: 检索到的相关 schema 文档拼接
        grounded_entities: 已 grounding 的实体 JSON 字符串
        has_spatial: 数据库是否包含空间数据列
        few_shots: 历史相似 query-SQL 对列表 [{'question': ..., 'sql': ...}, ...]
    """
    spatial_block = _SPATIAL_BLOCK if has_spatial else ""

    if few_shots:
        content_lines = []
        for i, fs in enumerate(few_shots, 1):
            content_lines.append(f"示例 {i}:")
            content_lines.append(f"  问题: {fs['question']}")
            content_lines.append(f"  SQL: {fs['sql']}")
        few_shot_block = _FEW_SHOT_PREFIX.format(few_shot_content="\n".join(content_lines))
    else:
        few_shot_block = ""

    return _SQL_GENERATION_BASE.format(
        relevant_schema=relevant_schema,
        grounded_entities=grounded_entities,
        few_shot_block=few_shot_block,
        spatial_block=spatial_block,
    )


# ====================== 4. SQL 修复 - 基础模板 + 动态组装 ======================

_SQL_FIX_BASE = """你是一个经验丰富的 PostgreSQL 数据库专家。请分析执行报错并修复 SQL。

**原始 SQL：**
{sql}

**累积错误信息**（按时间倒序，最新的在最上面）：
{error_history}

**相关 Schema 上下文**（用于校验列名/表名）：
{relevant_schema}
{dialect_hint}
**修复要求：**
1. 保留原始查询的业务意图和主体结构（SELECT 字段、JOIN、WHERE 等）
2. 常见错误类型和修复方向：
   - 列名/表名错误 → 对照 Schema 修正
   - 类型不匹配 → 使用 CAST 或 ::type 显式转换
   - 函数不存在 → 换成标准 PostgreSQL 函数
   - JOIN 条件错误 → 按 Schema 中的 Relationships 修正
3. 只返回修复后的完整 SQL，不要任何解释、markdown 标记或额外文字"""


_DIALECT_HINT_SPATIAL = """
**空间查询提示**：如涉及空间函数（ST_*），注意返回 geometry 时加 ::text 或用 ST_AsText / ST_AsGeoJSON 包装。
"""


def build_sql_fix_prompt(
    sql: str,
    error_history: list,
    relevant_schema: str = "",
    has_spatial: bool = False,
) -> str:
    """动态组装 SQL 修复 prompt。

    Args:
        sql: 待修复的 SQL
        error_history: 累积错误列表，每个元素为 str（最新的在前）
        relevant_schema: 相关 schema 上下文（用于校验列名）
        has_spatial: 是否涉及空间查询
    """
    error_text = "\n".join([f"[轮 {len(error_history) - i}] {err}" for i, err in enumerate(error_history)])
    dialect_hint = _DIALECT_HINT_SPATIAL if has_spatial else ""

    return _SQL_FIX_BASE.format(
        sql=sql,
        error_history=error_text,
        relevant_schema=relevant_schema or "（无额外 schema 上下文）",
        dialect_hint=dialect_hint,
    )


# ====================== 向后兼容（旧代码如果还引用这些名字，提供占位） ======================

# 注意：旧的 SQL_GENERATION_SYSTEM_TEMPLATE 和 SQL_FIX_TEMPLATE 已被上面的
# build_sql_generation_prompt / build_sql_fix_prompt 函数替代。
# multi_agent.py 会改为调用函数形式。
