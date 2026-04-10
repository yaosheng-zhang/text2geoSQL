#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
example.py - 融合版 M-Schema 完整演示示例（已修复 SQLAlchemy 2.0+ inspect 问题）
"""

import logging
from typing import Any, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================== 导入 ======================
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

# 你的配置（请确保 DB_URL 已正确配置）
from app.config import bge_embeddings, DB_URL

# ====================== MSchema 类（保持不变） ======================
class MSchema:
    def __init__(self, db_id: str = "demo_db"):
        self.db_id = db_id
        self.tables: dict = {}
        self.foreign_keys: List = []
        self.implicit_foreign_keys: List[dict] = []

    def add_table(self, name: str, comment: str = ""):
        self.tables[name] = {"fields": {}, "comment": comment or ""}

    def add_field(self, table_name: str, field_name: str, field_type: str = "",
                  primary_key: bool = False, nullable: bool = True,
                  default: Any = None, comment: str = "", examples: List = None):
        if table_name not in self.tables:
            self.add_table(table_name)
        self.tables[table_name]["fields"][field_name] = {
            "type": field_type,
            "primary_key": primary_key,
            "nullable": nullable,
            "default": default,
            "comment": comment or "",
            "examples": examples or []
        }

    def add_foreign_key(self, table: str, col: str, ref_table: str, ref_col: str, fk_type: str = "物理外键"):
        self.foreign_keys.append([table, col, ref_table, ref_col, fk_type])

    def add_implicit_foreign_key(self, table: str, col: str, ref_table: str, ref_col: str):
        self.add_foreign_key(table, col, ref_table, ref_col, "逻辑推断外键")
        self.implicit_foreign_keys.append({
            "source_table": table, "source_col": col,
            "target_table": ref_table, "target_col": ref_col,
            "type": "逻辑推断外键"
        })

    def infer_implicit_fks(self):
        explicit_links = {f"{t}.{c}->{rt}.{rc}" for t, c, rt, rc, _ in self.foreign_keys}
        from collections import defaultdict
        pk_map = defaultdict(list)
        for t_name, info in self.tables.items():
            for f_name, f in info["fields"].items():
                if f["primary_key"]:
                    pk_map[f_name].append((t_name, f_name))

        for t_name, info in self.tables.items():
            for c_name, c_info in info["fields"].items():
                if c_info["primary_key"]:
                    continue
                for target_t, target_pk in pk_map.get(c_name, []):
                    if target_t == t_name:
                        continue
                    singular = t_name.rstrip('s')
                    if (c_name == target_pk) or (c_name == f"{singular}_{target_pk}"):
                        link = f"{t_name}.{c_name}->{target_t}.{target_pk}"
                        if link not in explicit_links:
                            self.add_implicit_foreign_key(t_name, c_name, target_t, target_pk)
                            explicit_links.add(link)

    def to_enhanced_mschema(self) -> str:
        lines = [f"【DB_ID】 {self.db_id}", "【Schema】"]
        for table_name, t_info in self.tables.items():
            comment_part = f", {t_info['comment']}" if t_info['comment'] else ""
            lines.append(f"# Table: {table_name}{comment_part}")
            lines.append("[")
            for f_name, f in t_info["fields"].items():
                parts = [f"{f_name}:{f['type'].upper()}"]
                if f["primary_key"]:
                    parts.append("Primary Key")
                if f["comment"]:
                    parts.append(f["comment"])
                if f["examples"]:
                    parts.append(f"Examples: {f['examples'][:3]}")
                lines.append("  " + ", ".join(parts))
            lines.append("]")

            # JOIN Paths
            out_fks = [fk for fk in self.foreign_keys if fk[0] == table_name]
            in_fks = [fk for fk in self.foreign_keys if fk[2] == table_name]
            if out_fks or in_fks:
                lines.append("Relationships (JOIN Paths):")
                for fk in out_fks:
                    lines.append(f"  - To [{fk[2]}]: JOIN {fk[2]} ON {table_name}.{fk[1]} = {fk[2]}.{fk[3]} ({fk[4]})")
                for fk in in_fks:
                    lines.append(f"  - [{fk[0]}] can JOIN here: JOIN {fk[0]} ON {fk[0]}.{fk[1]} = {table_name}.{fk[3]} ({fk[4]})")

        lines.append("\n【General Knowledge】 Use JOIN only according to Relationships above.")
        return "\n".join(lines)


# ====================== 修复后的 EnhancedSchemaEngine ======================
class EnhancedSchemaEngine:
    IGNORE_TABLES = {'spatial_ref_sys', 'geometry_columns', 'geography_columns'}

    def __init__(self, db_url: str = DB_URL, sample_rows: int = 5):
        self.engine: Engine = create_engine(db_url, echo=False)
        self.mschema = MSchema(db_id="production_db")
        self.has_spatial = False
        self.sample_rows = sample_rows

    def build(self):
        # === 关键修复：使用 sqlalchemy.inspect(engine) ===
        inspector = inspect(self.engine)

        tables = [t for t in inspector.get_table_names(schema="public") 
                  if t not in self.IGNORE_TABLES]

        for table in tables:
            comment_dict = inspector.get_table_comment(table, schema="public")
            comment = comment_dict.get("text", "") if isinstance(comment_dict, dict) else ""

            self.mschema.add_table(table, comment)

            columns = inspector.get_columns(table, schema="public")
            pk_constraint = inspector.get_pk_constraint(table, schema="public")
            pk_cols = set(pk_constraint.get("constrained_columns", []))

            for col in columns:
                col_name = col["name"]
                col_type = str(col["type"])
                is_pk = col_name in pk_cols

                if any(geo in col_type.lower() for geo in ("geometry", "geography")):
                    self.has_spatial = True

                examples = self._get_examples(table, col_name)
                self.mschema.add_field(
                    table_name=table,
                    field_name=col_name,
                    field_type=col_type,
                    primary_key=is_pk,
                    nullable=col.get("nullable", True),
                    default=col.get("default"),
                    comment=col.get("comment", ""),
                    examples=examples
                )

            # 显式 Foreign Keys
            for fk in inspector.get_foreign_keys(table, schema="public"):
                for c_col, r_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                    self.mschema.add_foreign_key(table, c_col, fk["referred_table"], r_col, "物理外键")

        self.mschema.infer_implicit_fks()
        logger.info(f"✅ M-Schema 构建完成：{len(self.mschema.tables)} 张表 | "
                    f"显式FK: {len(self.mschema.foreign_keys)} | 逻辑FK: {len(self.mschema.implicit_foreign_keys)}")
        return self.mschema

    def _get_examples(self, table: str, column: str) -> List:
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT {self.sample_rows}"
                ))
                return [str(row[0]) for row in result if row[0] is not None]
        except Exception:
            return []

    def get_docs(self) -> List[Document]:
        self.build()
        docs = []
        for table_name in self.mschema.tables:
            content = self.mschema.to_enhanced_mschema()  # 简化处理，实际可优化为单表输出
            docs.append(Document(
                page_content=content,
                metadata={"table": table_name, "type": "table_schema"}
            ))

        if self.has_spatial:
            docs.append(Document(
                page_content="PostGIS 通用知识：使用 ST_Intersects、ST_DWithin 等空间函数前，请确保已 JOIN 包含 geometry 的表。",
                metadata={"table": "postgis_guide"}
            ))
        return docs


# ====================== 主函数 ======================
def main():
    logger.info("=== 融合版 M-Schema 示例开始（已修复 inspect） ===")

    if not DB_URL:
        logger.error("请在 app/config.py 中配置正确的 DB_URL")
        return

    engine = EnhancedSchemaEngine(DB_URL)
    mschema = engine.build()

    print("\n" + "="*80)
    print("【完整增强版 M-Schema 输出】")
    print("="*80)
    print(mschema.to_enhanced_mschema()[:2000] + "\n...（省略部分内容）")

    docs = engine.get_docs()
    print(f"\n✅ 成功生成 {len(docs)} 个 LangChain Document")

    # 向量检索演示
    vectorstore = InMemoryVectorStore.from_documents(docs, embedding=bge_embeddings)
    results = vectorstore.similarity_search("用户订单相关的表和 JOIN 方式", k=2)

    print("\n🔍 向量检索测试结果：")
    for i, doc in enumerate(results, 1):
        print(f"\n--- 结果 {i} (Table: {doc.metadata.get('table')}) ---")
        print(doc.page_content[:600] + "..." if len(doc.page_content) > 600 else doc.page_content)

    logger.info("=== 示例运行完成 ===")


if __name__ == "__main__":
    main()