# app/prompts.py - All prompt templates for the Text2GeoSQL pipeline

# ====================== 1. Entity Extraction ======================

ENTITY_EXTRACTION_TEMPLATE = """你是一个极其严谨、精确的**数据库实体提取专家**，专门服务于 Text-to-SQL 任务。

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
- 如果查询是条件描述（如"已竣工且高度超过100米的建筑"），也要提取：
  - "建筑" → infrastructure_type 或 poi_category
  - "高度超过100米" → numeric_condition
  - "已竣工" → status_condition
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
"""


# ====================== 2. LLM Rerank ======================

RERANK_TEMPLATE = """用户原始查询提及: "{original}"
我们找出了以下候选数据库标准命名:
{candidates_str}

判断哪一个最符合用户意图？只返回数字编号（1-{max_choice}），无需解释。都不行返回 0。"""


# ====================== 3. SQL Generation (merged planner + generator) ======================

SQL_GENERATION_SYSTEM_TEMPLATE = """你是 PostGIS 空间数据库查询专家，同时也是一个出色的查询规划者。

请基于以下信息，先在心里规划查询策略（需要哪些表、JOIN 方式、过滤条件），然后直接生成最终的 SQL。

**数据库 Schema:**
{relevant_schema}

**已匹配的实体（Grounded Entities）:**
{grounded_entities}

**规则：**
1. 只生成 SELECT 语句
2. 严格按照 Schema 中的 Relationships (JOIN Paths) 进行 JOIN
3. 使用 Grounded Entities 中的 canonical 值（而非用户原始输入）作为 WHERE 条件值
4. 如涉及空间查询，使用合适的 PostGIS 函数（ST_Contains, ST_Intersects, ST_DWithin, ST_Distance 等）
5. 使用空间函数前，确保已 JOIN 包含 GEOMETRY 列的表
6. 只返回一条有效的 SQL 语句，不要任何解释，不要 ```sql 标记"""


# ====================== 4. SQL Fix ======================

SQL_FIX_TEMPLATE = """你是一个经验丰富的 PostgreSQL + PostGIS 数据库专家。

**原始 SQL：**
{sql}

**执行报错信息：**
{error}

请分析错误原因并修复 SQL。
修复时请：
1. 保留原始查询的业务意图和主要结构（SELECT 字段、JOIN、WHERE 等）
2. 修复类型不匹配、函数调用、语法等问题（例如 GeoJSON 需要加 ::text）
3. 只返回修正后的完整 SQL 语句，不要任何解释、不要 ```sql 标记、不要额外文字。

修复后的 SQL："""
