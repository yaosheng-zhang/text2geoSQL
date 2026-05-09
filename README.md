# Text2GeoSQL

<p align="center">
  <b>自然语言 → 空间 SQL</b><br>
  基于 LangGraph 多 Agent 架构的 Text-to-SQL 引擎，专为 PostgreSQL / PostGIS 空间数据库设计
</p>

<p align="center">
  <a href="#-核心特性">核心特性</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-项目架构">项目架构</a> •
  <a href="#-配置说明">配置说明</a> •
  <a href="#-使用方式">使用方式</a>
</p>

---

## 核心特性

- **LangGraph 多 Agent 工作流** — 实体提取 → Schema 检索 → 动态 Grounding → SQL 生成 → 执行/自修复，全流程可视化追踪
- **混合检索 Grounding** — 向量语义搜索 (`pgvector` + `bge-m3`) 结合 PostgreSQL 文本相似度 (`pg_trgm`)，兼顾召回率与准确率
- **Schema-aware 实体提取** — LLM 基于真实数据库 Schema 提取实体，自动生成别名（如"东京铁塔"→["东京塔", "Tokyo Tower"]）
- **M-Schema 增强表示** — 自动构建含表注释、列示例、物理/隐式外键关系的增强 Schema，提升 LLM 对数据库结构的理解
- **Vanna 风格 Few-Shot 闭环** — 查询成功自动保存 `(question, sql)` 向量对，随使用自然积累，冷启动时自动退化为 Zero-Shot
- **多 LLM 供应商支持** — OpenAI、Azure OpenAI、DeepSeek、Anthropic Claude、智谱 AI、Ollama、vLLM、阿里云百炼，一键切换
- **生产级安全与日志** — Token 级 SQL 安全检查（`sqlparse`）、全链路结构化日志、连接池管理、健康检查
- **PostGIS 原生支持** — 自动检测空间列，注入 `ST_Intersects`、`ST_DWithin` 等空间函数使用规则

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/your-org/text2geoSQL.git
cd text2geoSQL
```

### 2. 启动数据库（PostGIS + pgvector）

```bash
docker-compose up -d postgis
```

或使用现有 PostgreSQL 数据库（需手动安装 `postgis` 和 `pgvector` 扩展）。

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key 和数据库地址
```

### 4. 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5. 运行值嵌入 ETL（首次必需）

```bash
python etl_value_embeddings.py
```

该脚本会自动扫描数据库中的文本列，使用 `bge-m3` 生成向量并写入 `value_embeddings` 表，为后续 Grounding 提供数据基础。

### 6. 启动服务

```bash
python -m app.main
```

服务启动后访问 http://127.0.0.1:8000/docs 查看交互式 API 文档。

### 7. 测试查询

```bash
python test_query.py
```

---

## 项目架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户查询 (NLQ)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  [1] Entity Extractor (Schema-aware)                        │
│      • 基于数据库 Schema 摘要提取实体                         │
│      • 自动生成多语言别名（别名、简称、英文）                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  [2] Dynamic Grounding Agent (混合检索)                      │
│      • 同类型向量搜索 Top-10                                 │
│      • 跨类型兜底搜索（含文本相似度过滤）                      │
│      • 混合打分: 0.7 * 向量 + 0.3 * pg_trgm                  │
│      • 歧义时触发 LLM Rerank                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  [3] SQL Generator                                          │
│      • Schema RAG 检索相关表结构 (Ensemble: BM25 + Vector)   │
│      • 注入 Few-Shot 示例（Vanna 风格向量检索）               │
│      • 动态拼接 PostGIS 空间规则                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  [4] SQL Executor / Fixer                                   │
│      • 执行 SQL，token 级安全检查                             │
│      • 出错时自动分类错误类型，触发 LLM 自修复（最多 2 轮）     │
│      • 成功后自动保存 Few-Shot 示例                          │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

| 层级 | 技术 |
|------|------|
| API 框架 | FastAPI + Uvicorn |
| 工作流编排 | LangGraph |
| LLM 接入 | LangChain（多供应商统一抽象） |
| 向量检索 | `pgvector` + `InMemoryVectorStore` + `bge-m3` |
| 文本检索 | `pg_trgm` + `BM25Retriever` |
| 数据库 | PostgreSQL 16 + PostGIS 3.5 |
| Embedding | BAAI/bge-m3 (1024-dim) |

---

## 配置说明

所有配置通过环境变量或 `.env` 文件管理：

### 数据库配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `DB_URL` | PostgreSQL 连接字符串 | `postgresql://postgres:postgres123@localhost:5432/spatial_kb` |

### LLM 配置

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM_PROVIDER` | 供应商类型 | `openai` / `deepseek` / `ollama` / `anthropic` / ... |
| `MODEL_NAME` | 模型名称 | `gpt-4o` / `deepseek-chat` / `qwen2.5:7b` |
| `LLM_API_KEY` | API Key | `sk-xxx` |
| `LLM_BASE_URL` | API 基础地址 | `https://api.openai.com/v1` |
| `LLM_TEMPERATURE` | 温度参数 | `0.0` |

### Embedding 配置

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `EMBEDDING_MODEL_PATH` | 本地 bge-m3 模型路径 | `BAAI/bge-m3`（自动从 HuggingFace 下载） |
| `EMBEDDING_DIM` | 向量维度 | `1024` |
| `ETL_BATCH_SIZE` | ETL 批处理大小 | `512` |

详细配置示例请参考 [`.env.example`](.env.example)。

---

## 使用方式

### HTTP API

**POST /query** — 自然语言查询

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "涩谷行政区内评级高于 4.5 的旅游 POI 有哪些？"}'
```

**响应示例：**

```json
{
  "sql": "SELECT sp.name, sp.rating, sp.category FROM spatial_chunks sp JOIN admin_boundaries ab ON ST_Within(sp.geom, ab.geom) WHERE ab.ward_name = '涩谷' AND sp.rating > 4.5 AND sp.category = '旅游'",
  "results": [
    {
      "title": "涩谷天空",
      "content": "rating: 4.8, category: 旅游",
      "metadata": { "name": "涩谷天空", "rating": 4.8, "category": "旅游" }
    }
  ],
  "execution_time_ms": 1450
}
```

**GET /health** — 健康检查

```bash
curl http://127.0.0.1:8000/health
```

### 作为库使用

```python
from app.multi_agent import run_text2geosql

result = run_text2geosql("东京铁塔附近 500 米内有哪些地铁站？")
print(result["sql"])
# SELECT ... ST_DWithin(...) ...
```

### M-Schema 示例

```python
from example import EnhancedSchemaEngine

engine = EnhancedSchemaEngine("postgresql://...")
mschema = engine.build()
print(mschema.to_enhanced_mschema())
```

---

## 项目结构

```
text2geoSQL/
├── app/
│   ├── main.py              # FastAPI 服务入口与生命周期管理
│   ├── multi_agent.py       # LangGraph 多 Agent 工作流核心
│   ├── schema_rag.py        # M-Schema 构建与 Schema RAG 检索
│   ├── few_shot.py          # Vanna 风格 Few-Shot 闭环学习
│   ├── llm_provider.py      # 多供应商 LLM 工厂
│   ├── config.py            # 全局配置与 Embedding 初始化
│   ├── db.py                # 数据库连接池（psycopg-pool）
│   ├── prompts.py           # 通用 Prompt 模板（领域解耦）
│   └── utils.py             # SQL 安全检查等工具函数
├── etl_value_embeddings.py  # 值嵌入 ETL（自动扫描 + 向量化）
├── example.py               # M-Schema 增强示例与向量检索演示
├── test_query.py            # 带健康检查的集成测试脚本
├── docker-compose.yml       # PostGIS + pgvector 一键编排
├── Dockerfile               # 多阶段构建的 PostGIS/pgvector 镜像
├── requirements.txt         # Python 依赖
└── .env.example             # 环境变量模板
```

---

## 核心优化与改进

本项目在经典 Text2SQL  pipeline 基础上进行了以下关键优化（详见 [`IMPROVEMENTS.md`](IMPROVEMENTS.md)）：

1. **混合 Grounding 检索** — 从纯向量搜索（阈值 0.25，漏召率高）升级为向量 + `pg_trgm` 混合打分（0.7/0.3 权重），匹配准确率显著提升
2. **别名生成与扩展** — LLM 自动为实体生成 1-4 个别名，Grounding 时向量取平均、文本相似度取最大，有效解决"东京铁塔"≠"东京塔"的问题
3. **LLM 重排消歧** — Top-1 与 Top-2 分数差距 < 0.15 时，触发 LLM 二次判断，避免歧义实体误匹配
4. **自适应阈值** — 从固定向量距离阈值改为混合分数阈值（> 0.5），减少漏召回
5. **Schema RAG 预热** — 服务启动时后台预热 Schema 向量库，避免首次查询卡顿
6. **Ensemble Retriever** — `BM25Retriever` + `InMemoryVectorStore` 组合检索，无需 `psycopg2` 依赖

---

## 相关概念

### M-Schema

M-Schema 是一种面向 LLM 的数据库结构表示格式，由本项目在经典 Schema 基础上增强：

- 表/列注释注入
- 列示例值（Sample Values）
- 物理外键 + 隐式外键自动推断（基于命名规范，如 `user_id` → `users.id`）
- JOIN 路径自动生成

### Vanna 风格 Few-Shot

区别于传统静态 Few-Shot，本系统采用动态闭环学习：

- **存储**：查询成功执行后，自动将 `(question, sql, embedding)` 存入 `sql_examples` 表
- **检索**：新查询通过向量相似度检索最相关的历史案例，注入 Prompt
- **演化**：随使用自然积累，系统越用越准，冷启动时自动退化为 Zero-Shot

---

## 许可

[MIT](LICENSE)

---

<p align="center">
  Built with LangGraph + FastAPI + PostGIS
</p>
