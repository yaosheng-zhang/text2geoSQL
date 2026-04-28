import logging
import os
from typing import Any, Dict, List, Optional, Set
import warnings

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from app.config import bge_embeddings, DB_URL

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Did not recognize type 'vector'")
warnings.filterwarnings("ignore", category=Warning, module="sqlalchemy")


# ====================== MSchema 核心类 ======================
class MSchema:
    def __init__(self, db_id: str = "Anonymous", schema: Optional[str] = None):
        self.db_id = db_id
        self.schema = schema
        self.tables: Dict[str, Dict] = {}
        self.foreign_keys: List[List] = []
        self.implicit_foreign_keys: List[Dict] = []

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
        self.add_foreign_key(table_name, field_name, ref_table_name, ref_field_name, "逻辑推断外键")

    def infer_implicit_fks(self, ignore_tables: set = None):
        if ignore_tables is None:
            ignore_tables = set()

        explicit_links = {f"{t}.{c}->{rt}.{rc}" for t, c, rt, rc, _ in self.foreign_keys}

        pk_map = {}
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
                    continue

                target_pk_list = pk_map.get(c_name, [])
                singular = t_name.rstrip("s")

                for target_t, target_pk in target_pk_list:
                    if target_t == t_name:
                        continue
                    if (c_name == target_pk) or (c_name == f"{singular}_{target_pk}"):
                        link_key = f"{t_name}.{c_name}->{target_t}.{target_pk}"
                        if link_key not in explicit_links:
                            self.add_implicit_foreign_key(t_name, c_name, target_t, target_pk)
                            explicit_links.add(link_key)

    def to_enhanced_mschema(self, has_spatial: bool = False) -> str:
        """转换为完整 M-Schema 文本。

        Args:
            has_spatial: 若为 True 则在末尾追加 PostGIS 通用知识；否则只追加通用 SQL 知识
        """
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
                    ex = f["examples"][:3]
                    if ex:
                        parts.append(f"Examples: [{', '.join(map(str, ex))}]")
                field_lines.append(", ".join(parts) + ")")

            lines.append("[")
            lines.extend(field_lines)
            lines.append("]")

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
                    if isinstance(fk, dict):
                        lines.append(
                            f"  - [{fk['source_table']}] can connect here using: "
                            f"JOIN {fk['source_table']} ON {fk['source_table']}.{fk['source_col']} = "
                            f"{table_name}.{fk['target_col']} ({fk['type']})"
                        )
                    else:
                        source_t, source_col, _, my_col, fk_type = fk
                        lines.append(
                            f"  - [{source_t}] can connect here using: "
                            f"JOIN {source_t} ON {source_t}.{source_col} = {table_name}.{my_col} ({fk_type})"
                        )

        lines.append("\n【General SQL Knowledge】")
        lines.append("- Use JOIN strictly according to the 'Relationships' paths above.")
        lines.append("- Use standard PostgreSQL syntax (CTE / WITH / subqueries are allowed).")

        if has_spatial:
            lines.append("\n【PostGIS Spatial Knowledge】")
            lines.append("- Spatial functions: ST_Contains, ST_Intersects, ST_DWithin, ST_Distance, ST_Within.")
            lines.append("- Always JOIN a table with GEOMETRY column before applying spatial functions.")
            lines.append("- For geometry output use ST_AsText() or ST_AsGeoJSON() to get readable text.")

        return "\n".join(lines)


# ====================== IGNORE_TABLES 动态构建 ======================

_INTERNAL_TABLES = {"value_embeddings", "sql_examples"}  # 本项目内部表，固定忽略
_POSTGIS_SYSTEM_TABLES = {"spatial_ref_sys", "geometry_columns", "geography_columns"}


def _detect_postgis(engine: Engine) -> bool:
    """检测数据库是否启用了 PostGIS 扩展。"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT 1 FROM pg_extension WHERE extname = 'postgis' LIMIT 1"
            ))
            return result.fetchone() is not None
    except Exception as e:
        logger.debug("检测 PostGIS 扩展失败（忽略）: %s", e)
        return False


def _build_ignore_tables(engine: Engine) -> Set[str]:
    """动态构建要忽略的表集合。

    规则：
    - 始终忽略本项目内部表（value_embeddings, sql_examples）
    - 若检测到 PostGIS 扩展，追加 PostGIS 系统表
    - 用户可通过 .env 的 IGNORE_TABLES=t1,t2,t3 扩展
    """
    ignore = set(_INTERNAL_TABLES)

    if _detect_postgis(engine):
        ignore.update(_POSTGIS_SYSTEM_TABLES)
        logger.info("检测到 PostGIS 扩展，已追加空间系统表到忽略列表")

    extra = os.getenv("IGNORE_TABLES", "").strip()
    if extra:
        user_ignore = {t.strip() for t in extra.split(",") if t.strip()}
        ignore.update(user_ignore)
        logger.info("用户自定义忽略表: %s", user_ignore)

    return ignore


# ====================== EnhancedSchemaEngine ======================
class EnhancedSchemaEngine:
    def __init__(self, db_url: str = DB_URL, sample_rows: int = 5):
        self.engine: Engine = create_engine(db_url, echo=False, pool_pre_ping=True)
        self.mschema = MSchema(db_id="dynamic_db", schema="public")
        self.has_spatial_data = False
        self.sample_rows = sample_rows
        self.ignore_tables = _build_ignore_tables(self.engine)

    def build(self) -> MSchema:
        inspector = inspect(self.engine)

        tables = inspector.get_table_names(schema="public")
        for table_name in tables:
            if table_name in self.ignore_tables:
                continue

            comment_dict = inspector.get_table_comment(table_name, schema="public")
            comment = comment_dict.get("text", "") if isinstance(comment_dict, dict) else ""

            self.mschema.add_table(table_name, comment=comment)

            columns = inspector.get_columns(table_name, schema="public")
            pk_constraint = inspector.get_pk_constraint(table_name, schema="public")
            pk_cols = pk_constraint.get("constrained_columns", [])

            for col in columns:
                col_name = col["name"]
                col_type = str(col["type"])
                is_pk = col_name in pk_cols

                if any(x in col_type.lower() for x in ("geometry", "geography")):
                    self.has_spatial_data = True

                examples = []  # will be fetched in batch below

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

            fks = inspector.get_foreign_keys(table_name, schema="public")
            for fk in fks:
                for c_col, r_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                    self.mschema.add_foreign_key(
                        table_name, c_col, fk["referred_table"], r_col, "物理外键"
                    )

        # Batch fetch all distinct values with a single connection
        self._batch_fetch_examples()

        self.mschema.infer_implicit_fks(ignore_tables=self.ignore_tables)

        logger.info("M-Schema 构建完成: %d 张表, 显式 FK %d, 逻辑 FK %d, 空间数据: %s",
                     len(self.mschema.tables),
                     len(self.mschema.foreign_keys),
                     len(self.mschema.implicit_foreign_keys),
                     self.has_spatial_data)
        return self.mschema

    def _batch_fetch_examples(self):
           """
           极速且安全的数据采样：
           1. 使用 SELECT * 避免全表扫描
           2. 过滤掉无意义的大字段类型
           3. 对过长的数据样本进行截断，防止撑爆 Embedding 模型
           """
           # 设定一个样例的最大字符长度，超过则截断
           MAX_EXAMPLE_LENGTH = 50 
           
           with self.engine.connect() as conn:
               for table_name, table_info in self.mschema.tables.items():
                   try:
                       # 1. 整表取前几行，避免按列查
                       query = text(f'SELECT * FROM "{table_name}" LIMIT :limit')
                       result = conn.execute(query, {"limit": self.sample_rows}).mappings().all()
    
                       for col_name, col_info in table_info["fields"].items():
                           col_type = col_info["type"].lower()
                           
                           # 2. 【类型过滤】：跳过这些又长又没必要做样例的字段
                           if any(x in col_type for x in ("geometry", "geography", "bytea", "blob", "json")):
                               col_info["examples"] = []
                               continue
                            
                           # 3. 【暴力截断】：安全提取数据
                           examples_set = set()
                           for row in result:
                               val = row.get(col_name)
                               if val is not None:
                                   # 转为字符串处理
                                   val_str = str(val).strip()
                                   # 如果字符串太长，直接截断
                                   if len(val_str) > MAX_EXAMPLE_LENGTH:
                                       val_str = val_str[:MAX_EXAMPLE_LENGTH] + "..."
                                   # 去除空字符串的情况
                                   if val_str:
                                       examples_set.add(val_str)
                                       
                           col_info["examples"] = list(examples_set)
    
                   except Exception as e:
                       logger.warning(f"提取表 {table_name} 样例数据失败 (跳过): {e}")
                       for col_info in table_info["fields"].values():
                           if "examples" not in col_info:
                               col_info["examples"] = []
    
    def get_docs(self) -> List[Document]:
        self.build()
        docs = []

        for table_name in self.mschema.tables:
            single_doc = self._single_table_enhanced_doc(table_name)
            docs.append(Document(
                page_content=single_doc,
                metadata={"table": table_name, "type": "table_schema"}
            ))

        if self.has_spatial_data:
            postgis_doc = """
PostGIS Spatial Knowledge (no specific tables):
- Use JOIN strictly according to the 'Relationships (JOIN Paths)' in each table.
- Spatial functions: ST_Contains, ST_Intersects, ST_DWithin, ST_Distance, ST_Within, etc.
- Always JOIN a table containing a GEOMETRY column before applying spatial functions.
- For geometry output, wrap with ST_AsText() or ST_AsGeoJSON() for readable results.
"""
            docs.append(Document(
                page_content=postgis_doc.strip(),
                metadata={"table": "postgis_guide", "type": "dialect_guide"}
            ))

        logger.info("生成 %d 个 M-Schema 知识块", len(docs))
        return docs

    def _single_table_enhanced_doc(self, table_name: str) -> str:
        table_info = self.mschema.tables[table_name]
        lines = [f"Table: {table_name}"]
        if table_info.get("comment"):
            lines.append(f"Comment: {table_info['comment']}")

        lines.append("Columns:")
        for col_name, col in table_info["fields"].items():
            pk_mark = " [PRIMARY KEY]" if col["primary_key"] else ""
            ex = f" [Examples: {col.get('examples', [])[:2]}]" if col.get("examples") else ""
            lines.append(f"  - {col_name} ({col['type']}){pk_mark}{ex}")

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


# ====================== 对外接口 ======================
_vectorstore = None
_schema_docs_cache = None
_engine_cache: Optional[EnhancedSchemaEngine] = None


def _get_engine() -> EnhancedSchemaEngine:
    """获取或创建全局 EnhancedSchemaEngine 单例。"""
    global _engine_cache
    if _engine_cache is None:
        _engine_cache = EnhancedSchemaEngine(DB_URL)
    return _engine_cache


def get_dynamic_m_schema_docs() -> list[Document]:
    global _schema_docs_cache

    if _schema_docs_cache is not None:
        logger.info("使用缓存的 Schema 文档 (%d 个知识块)", len(_schema_docs_cache))
        return _schema_docs_cache

    try:
        logger.info("正在执行 M-Schema 自动化构建...")
        engine = _get_engine()
        _schema_docs_cache = engine.get_docs()
        logger.info("M-Schema 文档缓存完成")
        return _schema_docs_cache
    except Exception as e:
        logger.error("M-Schema 构建失败: %s", e, exc_info=True)
        raise


def get_schema_vectorstore():
    global _vectorstore

    if _vectorstore is not None:
        logger.info("使用缓存的 Schema 向量库")
        return _vectorstore

    logger.info("正在初始化 Schema RAG 内存向量库...")
    import time
    start = time.time()

    try:
        schema_docs = get_dynamic_m_schema_docs()
        logger.info("开始向量化 %d 个 Schema 文档...", len(schema_docs))

        # 打印待向量化的文档内容
        for idx, doc in enumerate(schema_docs):
            logger.info("=== 待向量化文档 [%d] ===\n%s\n=== 文档 [%d] 结束 ===", idx, doc.page_content, idx)


        _vectorstore = InMemoryVectorStore.from_documents(
            documents=schema_docs,
            embedding=bge_embeddings,
        )

        elapsed = time.time() - start
        logger.info("Schema RAG 初始化完成 (%.2fs), 共 %d 个知识块", elapsed, len(schema_docs))
        return _vectorstore
    except Exception as e:
        logger.error("Schema RAG 初始化失败: %s", e, exc_info=True)
        raise


def has_spatial_data() -> bool:
    """返回当前数据库是否包含 geometry/geography 列。

    用于动态决定是否在 prompt 中注入 PostGIS 规则。
    """
    engine = _get_engine()
    if not engine.mschema.tables:  # 尚未构建
        engine.build()
    return engine.has_spatial_data


def get_schema_summary(max_cols_per_table: int = 6) -> str:
    """返回紧凑的 schema 摘要，供 schema-aware 实体提取使用。

    格式示例：
        Tables:
        - products (产品表): id, name, category, price, status
        - orders (订单表): order_id, customer_id, amount, order_date
        - ...

    Args:
        max_cols_per_table: 每张表最多显示多少列（优先显示主键 + 非外键列）
    """
    engine = _get_engine()
    if not engine.mschema.tables:
        engine.build()

    lines = ["Tables:"]
    for table_name, table_info in engine.mschema.tables.items():
        comment = table_info.get("comment", "")

        # 优先选择主键 + 关键列（过滤掉纯几何/embedding 列）
        cols = []
        for col_name, col in table_info["fields"].items():
            col_type = col.get("type", "").lower()
            # 跳过几何和大字段
            if any(x in col_type for x in ("geometry", "geography", "bytea")):
                continue
            cols.append(col_name)

        cols = cols[:max_cols_per_table]
        col_str = ", ".join(cols)

        header = f"- {table_name}"
        if comment:
            header += f" ({comment})"
        header += f": {col_str}"
        lines.append(header)

    return "\n".join(lines)
