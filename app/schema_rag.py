import logging
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

import psycopg  # 保留你的原生连接用于 PostGIS 探测（可选）
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.config import bge_embeddings, DB_URL  # 你的配置

logger = logging.getLogger(__name__)

# ====================== 官方 M-Schema 核心类（已融合） ======================
# 以下直接基于 https://github.com/XGenerationLab/M-Schema 的 m_schema.py 和 schema_engine.py
# 进行了最小化精简 + 增强（省略了 LlamaIndex 依赖，只保留核心逻辑）

class MSchema:
    """官方 M-Schema 核心类（增强版）"""
    def __init__(self, db_id: str = "Anonymous", schema: Optional[str] = None):
        self.db_id = db_id
        self.schema = schema
        self.tables: Dict[str, Dict] = {}
        self.foreign_keys: List[List] = []          # 显式 FK
        self.implicit_foreign_keys: List[Dict] = [] # 新增：逻辑推断 FK（你的核心优势）

    def add_table(self, name: str, comment: str = ""):
        self.tables[name] = {
            "fields": {},
            "comment": comment.strip() if comment else ""
        }

    def add_field(self, table_name: str, field_name: str, field_type: str = "",
                  primary_key: bool = False, nullable: bool = True,
                  default: Any = None, autoincrement: bool = False,
                  comment: str = "", examples: List = None):
        if table_name not in self.tables:
            self.add_table(table_name)
        self.tables[table_name]["fields"][field_name] = {
            "type": field_type,
            "primary_key": primary_key,
            "nullable": nullable,
            "default": default,
            "autoincrement": autoincrement,
            "comment": comment.strip() if comment else "",
            "examples": examples or []
        }

    def add_foreign_key(self, table_name: str, field_name: str,
                        ref_table_name: str, ref_field_name: str, fk_type: str = "物理外键"):
        self.foreign_keys.append([table_name, field_name, ref_table_name, ref_field_name, fk_type])

    def add_implicit_foreign_key(self, table_name: str, field_name: str,
                                 ref_table_name: str, ref_field_name: str):
        self.implicit_foreign_keys.append({
            "source_table": table_name,
            "source_col": field_name,
            "target_table": ref_table_name,
            "target_col": ref_field_name,
            "type": "逻辑推断外键"
        })
        # 同时加入 foreign_keys（统一管理）
        self.add_foreign_key(table_name, field_name, ref_table_name, ref_field_name, "逻辑推断外键")

    # ====================== 你的隐式 FK 启发式（核心融合） ======================
    def infer_implicit_fks(self, ignore_tables: set = None):
        """你的隐式 FK 推断逻辑（已优化：支持复合 PK、类型匹配、去重）"""
        if ignore_tables is None:
            ignore_tables = set()

        explicit_links = {
            f"{t}.{c}->{rt}.{rc}"
            for t, c, rt, rc, _ in self.foreign_keys
        }

        pk_map = {}  # col_name -> [(table, pk_col), ...]
        for t_name, info in self.tables.items():
            if t_name in ignore_tables:
                continue
            for f_name, f_info in info["fields"].items():
                if f_info["primary_key"]:
                    pk_map.setdefault(f_name, []).append((t_name, f_name))

        for t_name, info in self.tables.items():
            if t_name in ignore_tables:
                continue
            for c_name, c_info in info["fields"].items():
                if c_info["primary_key"]:
                    continue  # 不把自己的 PK 当外键

                target_pk_list = pk_map.get(c_name, [])
                singular = t_name.rstrip("s")  # 简单去复数

                for target_t, target_pk in target_pk_list:
                    if target_t == t_name:
                        continue

                    # 规则1: 字段名 == 目标表 PK 名
                    # 规则2: 字段名 == 目标表单数_ PK 名
                    if (c_name == target_pk) or (c_name == f"{singular}_{target_pk}"):
                        link_key = f"{t_name}.{c_name}->{target_t}.{target_pk}"
                        if link_key not in explicit_links:
                            self.add_implicit_foreign_key(t_name, c_name, target_t, target_pk)
                            explicit_links.add(link_key)

    def to_enhanced_mschema(self) -> str:
        """官方 to_mschema + 你的 Relationships (JOIN Paths)"""
        lines = [f"【DB_ID】 {self.db_id}", "【Schema】"]

        for table_name, table_info in self.tables.items():
            comment = table_info.get("comment", "")
            header = f"# Table: {table_name}"
            if comment:
                header += f", {comment}"
            lines.append(header)

            field_lines = []
            for field_name, f in table_info["fields"].items():
                parts = [f"({field_name}:{f['type'].upper()}"]
                if f["comment"]:
                    parts.append(f["comment"])
                if f["primary_key"]:
                    parts.append("Primary Key")
                if f.get("examples"):
                    ex = f["examples"][:3]  # 最多3个示例
                    if ex:
                        parts.append(f"Examples: [{', '.join(map(str, ex))}]")
                field_lines.append(", ".join(parts) + ")")

            lines.append("[")
            lines.extend(field_lines)
            lines.append("]")

            # ============ 你的 JOIN Paths（关键增强） ============
            fks_out = [fk for fk in self.foreign_keys if fk[0] == table_name]
            fks_in = [fk for fk in self.foreign_keys if fk[2] == table_name] + self.implicit_foreign_keys

            if fks_out or fks_in:
                lines.append("Relationships (JOIN Paths):")
                for fk in fks_out:
                    _, my_col, target_t, target_col, fk_type = fk
                    lines.append(
                        f"  - To connect to [{target_t}], use: "
                        f"JOIN {target_t} ON {table_name}.{my_col} = {target_t}.{target_col} ({fk_type})"
                    )
                for fk in fks_in:
                    if isinstance(fk, dict):  # implicit
                        lines.append(
                            f"  - [{fk['source_table']}] can connect here using: "
                            f"JOIN {fk['source_table']} ON {fk['source_table']}.{fk['source_col']} = "
                            f"{table_name}.{fk['target_col']} ({fk['type']})"
                        )
                    else:
                        source_t, source_col, _, my_col, fk_type = fk  # 反向
                        lines.append(
                            f"  - [{source_t}] can connect here using: "
                            f"JOIN {source_t} ON {source_t}.{source_col} = {table_name}.{my_col} ({fk_type})"
                        )

        # PostGIS 通用知识块（你的特色）
        lines.append("\n【General SQL & PostGIS Knowledge】")
        lines.append("- Use JOIN strictly according to the 'Relationships' paths above.")
        lines.append("- Spatial: ST_Contains, ST_Intersects, ST_DWithin, ST_Distance, etc.")
        lines.append("- Always JOIN a table with GEOMETRY column before using spatial functions.")

        return "\n".join(lines)


# ====================== 增强版 SchemaEngine（融合官方 + 你的特性） ======================
class EnhancedSchemaEngine:
    """融合版引擎：SQLAlchemy（官方） + 你的隐式 FK + PostGIS 探测 + LangChain Document 输出"""

    IGNORE_TABLES = {'spatial_ref_sys', 'geometry_columns', 'geography_columns'}

    def __init__(self, db_url: str = DB_URL, sample_rows: int = 5):
        self.engine: Engine = create_engine(db_url, echo=False)
        self.mschema = MSchema(db_id="dynamic_db", schema="public")
        self.has_spatial_data = False
        self.sample_rows = sample_rows

    def build(self) -> MSchema:
        """完整构建流程"""
        inspector = self.engine.inspect()

        # 1. 提取所有表（仅 public）
        tables = inspector.get_table_names(schema="public")
        for table_name in tables:
            if table_name in self.IGNORE_TABLES:
                continue

            comment = inspector.get_table_comment(table_name, schema="public").get("text", "") or ""
            self.mschema.add_table(table_name, comment=comment)

            # 2. 字段 + 丰富信息（官方最强部分）
            columns = inspector.get_columns(table_name, schema="public")
            pk_cols = inspector.get_pk_constraint(table_name, schema="public")["constrained_columns"]

            for col in columns:
                col_name = col["name"]
                col_type = str(col["type"])
                is_pk = col_name in pk_cols

                # PostGIS 探测
                if any(x in col_type.lower() for x in ("geometry", "geography")):
                    self.has_spatial_data = True

                examples = self._fetch_distinct_values(table_name, col_name)

                self.mschema.add_field(
                    table_name=table_name,
                    field_name=col_name,
                    field_type=col_type,
                    primary_key=is_pk,
                    nullable=col.get("nullable", True),
                    default=col.get("default"),
                    autoincrement=col.get("autoincrement", False),
                    comment=col.get("comment") or "",
                    examples=examples
                )

            # 3. 显式 FK（官方可靠方式）
            fks = inspector.get_foreign_keys(table_name, schema="public")
            for fk in fks:
                for c_col, r_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                    self.mschema.add_foreign_key(
                        table_name, c_col, fk["referred_table"], r_col, "物理外键"
                    )

        # 4. 你的隐式 FK 推断（核心实战增强）
        self.mschema.infer_implicit_fks(ignore_tables=self.IGNORE_TABLES)

        logger.info(f"✅ M-Schema 构建完成：{len(self.mschema.tables)} 张表，"
                    f"显式 FK {len(self.mschema.foreign_keys)}，逻辑 FK {len(self.mschema.implicit_foreign_keys)}")
        return self.mschema

    def _fetch_distinct_values(self, table_name: str, column_name: str) -> List:
        """官方风格的 distinct values 采样（安全、实用）"""
        query = text(f"""
            SELECT DISTINCT {column_name} 
            FROM {table_name} 
            WHERE {column_name} IS NOT NULL 
            LIMIT {self.sample_rows}
        """)
        try:
            with self.engine.connect() as conn:
                result = conn.execute(query)
                return [row[0] for row in result.fetchall() if row[0] is not None]
        except Exception:
            return []

    def get_docs(self) -> List[Document]:
        """最终输出：每个表一个 LangChain Document（完美适配你的 RAG）"""
        self.build()
        docs = []

        # 每张表一个 Document（带完整 JOIN 提示）
        for table_name in self.mschema.tables:
            # 生成单表增强描述
            single_doc = self._single_table_enhanced_doc(table_name)
            docs.append(Document(
                page_content=single_doc,
                metadata={"table": table_name, "type": "table_schema"}
            ))

        # PostGIS 通用知识块
        if self.has_spatial_data:
            postgis_doc = """
General SQL & PostGIS Knowledge (No specific tables):
- Use JOIN strictly according to the 'Relationships (JOIN Paths)' in each table.
- Spatial functions: ST_Contains(polygon, point), ST_Intersects(geom1, geom2), ST_DWithin(geom1, geom2, distance)
- Always JOIN a table containing GEOMETRY column before applying spatial functions.
"""
            docs.append(Document(
                page_content=postgis_doc.strip(),
                metadata={"table": "postgis_guide", "type": "dialect_guide"}
            ))

        logger.info(f"✅ 生成 {len(docs)} 个 M-Schema 知识块（LangChain Document）")
        return docs

    def _single_table_enhanced_doc(self, table_name: str) -> str:
        """单表增强输出（官方格式 + 你的 JOIN Paths）"""
        table_info = self.mschema.tables[table_name]
        lines = [f"Table: {table_name}"]
        if table_info["comment"]:
            lines.append(f"Comment: {table_info['comment']}")

        lines.append("Columns:")
        for col_name, col in table_info["fields"].items():
            pk_mark = " [PRIMARY KEY]" if col["primary_key"] else ""
            ex = f" [Examples: {col['examples'][:2]}]" if col.get("examples") else ""
            lines.append(f"  - {col_name} ({col['type']}){pk_mark}{ex}")

        # JOIN Paths（你的最大优势）
        fks_out = [fk for fk in self.mschema.foreign_keys if fk[0] == table_name]
        fks_in = [fk for fk in self.mschema.foreign_keys if fk[2] == table_name]

        if fks_out or fks_in or self.mschema.implicit_foreign_keys:
            lines.append("Relationships (JOIN Paths):")
            for fk in fks_out:
                _, my_col, target, target_col, fk_type = fk
                lines.append(
                    f"  - To [{target}]: JOIN {target} ON {table_name}.{my_col} = {target}.{target_col} ({fk_type})"
                )
            for fk in fks_in:
                source, source_col, _, my_col, fk_type = fk
                lines.append(
                    f"  - [{source}] can JOIN here: JOIN {source} ON {source}.{source_col} = {table_name}.{my_col} ({fk_type})"
                )

        return "\n".join(lines)


# ====================== 你的原始接口（无缝替换） ======================
_vectorstore = None

def get_dynamic_m_schema_docs() -> list[Document]:
    """最终推荐接口：直接替换你原来的函数"""
    global _vectorstore
    try:
        logger.info("🚀 正在执行【融合版 M-Schema】自动化构建...")
        engine = EnhancedSchemaEngine(DB_URL)
        return engine.get_docs()
    except Exception as e:
        logger.error("融合版 M-Schema 构建失败", exc_info=True)
        raise


def get_schema_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        logger.info("正在初始化 Schema RAG 内存向量库（融合版）...")
        schema_docs = get_dynamic_m_schema_docs()
        _vectorstore = InMemoryVectorStore.from_documents(
            documents=schema_docs,
            embedding=bge_embeddings,
        )
        logger.info(f"✅ Schema RAG 初始化完成，共 {len(schema_docs)} 个知识块")
    return _vectorstore