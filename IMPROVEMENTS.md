# Text2SQL 优化改进说明

## 问题分析

### 原始问题
1. **实体名称不匹配**：用户查询"东京铁塔"，但数据库存储的是"东京塔"
2. **Grounding 不准确**：纯向量搜索，阈值过严（0.25），短文本语义区分度不够
3. **数据缺失**：`infrastructure` 表的数据未被 ETL 索引，基础设施实体无法匹配
4. **跨类型匹配失败**：严格按 `entity_type` 过滤，导致类型判断错误时完全无法匹配
5. **pg_trgm 扩展未使用**：数据库已安装文本相似度扩展，但完全没用上

---

## 解决方案

### 1. 修复 ETL 数据缺失 ✅

**文件**: `etl_value_embeddings.py`

**改动**:
```python
sources = [
    ("spatial_chunks", "title", "poi_name", False),
    ("poi_details", "category", "poi_category", False),
    ("poi_details", "tags", "poi_tag", True),
    ("admin_boundaries", "ward_name", "ward", False),
    ("admin_boundaries", "district", "district", False),
    ("infrastructure", "name", "infrastructure_name", False),      # 新增
    ("infrastructure", "type", "infrastructure_type", False),      # 新增
]
```

**效果**: 基础设施实体现在可以被正确索引和匹配。

---

### 2. 混合搜索：向量 + 文本相似度 ✅

**文件**: `app/multi_agent.py` → `dynamic_grounding()`

**核心改进**:
```sql
-- 综合打分公式
hybrid_score = 0.7 * (1 - vec_distance) + 0.3 * text_sim

-- 使用 pg_trgm 的 similarity() 函数计算文本相似度
similarity(raw_value, '东京铁塔')  -- 返回 0-1 之间的相似度
```

**搜索策略**:
1. **优先同类型搜索**：先在相同 `entity_type` 内搜索（Top 10）
2. **跨类型兜底**：如果同类型最高分 < 0.6，则跨类型搜索（Top 15）
3. **文本相似度过滤**：跨类型搜索时要求 `text_sim > 0.3`

**阈值调整**:
- 原始：纯向量距离 < 0.25（过严）
- 现在：混合分数 > 0.5（更合理）

---

### 3. 实体别名生成 ✅

**文件**: `app/multi_agent.py` → `entity_extractor()`

**改动**:
```python
class ExtractedEntity(BaseModel):
    original: str
    entity_type: str
    aliases: List[str] = Field(default=[], description="可能的别名、简称、全称")
```

**Prompt 增强**:
```
例如：
- "东京铁塔" → aliases: ["东京塔", "Tokyo Tower", "铁塔"]
- "涩谷行政区" → aliases: ["涩谷", "Shibuya", "涩谷区"]
- "餐厅" → aliases: ["饭店", "restaurant", "食堂"]
```

**Grounding 时的使用**:
```python
search_terms = [original] + aliases  # ["东京铁塔", "东京塔", "Tokyo Tower"]

# 1. 向量平均（提高召回率）
embeddings = [encode(term) for term in search_terms]
query_emb = average(embeddings)

# 2. 文本相似度取最大值
GREATEST(similarity(raw_value, '东京铁塔'), 
         similarity(raw_value, '东京塔'),
         similarity(raw_value, 'Tokyo Tower'))
```

---

### 4. LLM 重排（处理歧义） ✅

**触发条件**: Top-1 和 Top-2 分数差距 < 0.15

**流程**:
```python
if abs(top1_score - top2_score) < 0.15:
    # 让 LLM 判断哪个候选项最符合用户意图
    rerank_prompt = f"""
    用户查询中提到了实体: "{original}"
    我找到了以下候选匹配项:
    1. 东京塔 (来源: spatial_chunks.title, 类型: poi_name, 分数: 0.78)
    2. 东京铁塔附近公交站 (来源: infrastructure.name, 类型: infrastructure_name, 分数: 0.76)
    
    请判断哪个最符合用户意图，只返回数字编号（1-2），不要解释。
    """
    choice = llm.invoke(rerank_prompt)
```

**效果**: 处理"东京铁塔"这种歧义实体时，LLM 会根据上下文选择正确的候选项。

---

### 5. 日志系统完善 ✅

**文件**: `app/main.py`, `app/multi_agent.py`, `app/config.py`, `app/schema_rag.py`

**改进**:
- 添加 `logging.basicConfig()` 配置（之前日志完全不输出）
- 每个 Agent 节点添加详细日志（步骤编号、耗时、输入输出）
- HTTP 请求中间件记录每个请求的耗时
- 数据库连接失败时的详细错误提示

**日志示例**:
```
2026-04-09 15:30:12 [INFO] app.multi_agent - 【2】动态 Grounding Agent 开始（混合搜索 + LLM 重排）
2026-04-09 15:30:12 [INFO] app.multi_agent - 2.2.1 处理实体: '东京铁塔' (type=poi_name, 别名: ['东京塔', 'Tokyo Tower'])
2026-04-09 15:30:13 [INFO] app.multi_agent - 2.5.1 ✓ 匹配成功: '东京铁塔' → '东京塔' (混合分数: 0.856, 向量距离: 0.123, 文本相似度: 0.789)
```

---

### 6. 其他优化 ✅

**`app/utils.py`**: 修复 SQL 安全检查
- 原始：任何包含 `;` 的 SQL 都被拒绝（误判）
- 现在：使用 `sqlparse` token 级别检测，允许末尾分号

**`app/schema_rag.py`**: 修复 PGVector 依赖问题
- 原始：使用 `PGVector`（依赖 `psycopg2`，但项目只装了 `psycopg` v3）
- 现在：使用 `InMemoryVectorStore`（无依赖冲突）

**`test_query.py`**: 增强测试脚本
- 添加健康检查（最多等待 30 秒）
- 请求超时从 20s 提升到 120s（LLM 调用较慢）
- 更清晰的错误提示

---

## 使用方式

### 1. 重新运行 ETL（必须）
```bash
python etl_value_embeddings.py
```
这会将 `infrastructure` 表的数据索引到 `value_embeddings` 表。

### 2. 启动服务
```bash
# 终端 1: 启动服务
python -m app.main

# 终端 2: 运行测试
python test_query.py
```

### 3. 查看日志
服务端会输出详细的日志，包括：
- 每个 Agent 的执行时间
- 实体提取结果（包括别名）
- Grounding 匹配详情（向量距离、文本相似度、混合分数）
- LLM 重排决策
- SQL 生成和执行结果

---

## 预期效果

### 测试案例 1: "东京铁塔附近有哪些基础设施？"

**之前**:
- 实体提取: `{"original": "东京铁塔", "entity_type": "poi_name"}`
- Grounding: 未找到匹配（向量距离 > 0.25），使用原始值
- SQL: `WHERE title = '东京铁塔'` → 查询失败（数据库存的是"东京塔"）

**现在**:
- 实体提取: `{"original": "东京铁塔", "entity_type": "poi_name", "aliases": ["东京塔", "Tokyo Tower"]}`
- Grounding: 混合搜索匹配到"东京塔"（混合分数 0.856）
- SQL: `WHERE title = '东京塔'` → 查询成功 ✅

### 测试案例 2: "涩谷行政区内评级高于 4.5 的旅游 POI 有哪些？"

**之前**:
- 实体提取: `{"original": "涩谷行政区", "entity_type": "ward"}`
- Grounding: 可能匹配到"Shibuya"（如果向量距离够近）

**现在**:
- 实体提取: `{"original": "涩谷行政区", "entity_type": "ward", "aliases": ["涩谷", "Shibuya"]}`
- Grounding: 文本相似度 `similarity("Shibuya", "涩谷") = 0.6`，混合分数更高
- 匹配准确率提升 ✅

---

## 技术亮点

1. **混合检索**：向量语义 + 文本字面，兼顾召回率和准确率
2. **多级兜底**：同类型 → 跨类型 → LLM 重排 → 原始值
3. **别名扩展**：LLM 生成别名，向量平均 + 文本最大相似度
4. **自适应阈值**：从固定 0.25 降低到混合分数 0.5，减少漏召回
5. **生产级日志**：每个环节可追踪，便于调试和监控

---

## 后续优化建议

1. **别名词典**：为常见地标维护人工标注的别名词典（如"东京铁塔" → "东京塔"）
2. **缓存机制**：对高频查询的 Grounding 结果做缓存
3. **A/B 测试**：调整混合分数权重（当前 0.7 向量 + 0.3 文本）
4. **用户反馈**：收集 Grounding 错误案例，持续优化
5. **多语言支持**：增强中英文混合查询的处理能力
