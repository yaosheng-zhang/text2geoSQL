import os
import psycopg
from psycopg import sql
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Set
from dotenv import load_dotenv

load_dotenv()

# ====================== 配置 ======================
DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres123@localhost:5432/city_planning")
BATCH_SIZE = int(os.getenv("ETL_BATCH_SIZE", "512"))
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # bge-m3 维度

# HuggingFace 镜像（可选）
if os.getenv("EMBEDDING_MODEL_PATH"):
    MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("正在加载 Embedding 模型...")
model = SentenceTransformer(MODEL_PATH)

# ==================== 动态配置 ====================
TARGET_SCHEMA = "public"
WHITELIST_TABLES: List[str] = []

# 跳过不适合做值嵌入的列（ID/时间戳/几何/密码等）
BLACKLIST_COLUMNS = {
    "id", "created_at", "updated_at", "geom", "geometry", "location",
    "uuid", "code", "key", "password", "email", "phone", "gid", "embedding"
}

# 本项目内部表（始终忽略）
_INTERNAL_TABLES = {"value_embeddings", "sql_examples"}
# PostGIS 系统表（仅当库启用 PostGIS 时忽略）
_POSTGIS_SYSTEM_TABLES = {"spatial_ref_sys", "geometry_columns", "geography_columns"}

# 只嵌入文本类型列（数值/日期由 LLM 作为 numeric_condition / date_condition 处理）
EMBEDDABLE_TYPES = {
    "character varying", "varchar", "text", "character", "name",
    "_text", "_varchar", "ARRAY"
}


def _detect_postgis(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'postgis' LIMIT 1")
        return cur.fetchone() is not None


def _build_blacklist_tables(conn) -> Set[str]:
    """动态构建忽略表集合。"""
    bl = set(_INTERNAL_TABLES)
    if _detect_postgis(conn):
        bl.update(_POSTGIS_SYSTEM_TABLES)
        logger.info("检测到 PostGIS 扩展，已追加空间系统表到忽略列表")

    extra = os.getenv("IGNORE_TABLES", "").strip()
    if extra:
        user_ignore = {t.strip() for t in extra.split(",") if t.strip()}
        bl.update(user_ignore)
        logger.info(f"用户自定义忽略表: {user_ignore}")

    return bl


def create_value_embeddings_table_if_not_exists(conn):
    """自动创建 value_embeddings 表（不含 entity_type 查询语义列）。

    设计说明：
    - 不再存储 entity_type。LLM 提取的 entity_type 是"查询侧语义"
      （value/numeric_condition/...），而数据表里的值都是 "value" 本质，
      硬塞会导致语义错位
    - source_table + source_column 已足够定位值来源
    """
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            vector_type = f"VECTOR({EMBEDDING_DIM})"
            logger.info("✅ 已启用 pgvector 扩展")
        except Exception as e:
            logger.warning(f"pgvector 不可用，回退使用 FLOAT[]: {e}")
            vector_type = "FLOAT[]"

        # pg_trgm 用于 similarity() 函数
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        except Exception as e:
            logger.warning(f"pg_trgm 不可用: {e}")

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS value_embeddings (
                id SERIAL PRIMARY KEY,
                source_table TEXT NOT NULL,
                source_column TEXT NOT NULL,
                raw_value TEXT NOT NULL,
                embedding {vector_type} NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                CONSTRAINT uk_value_embeddings
                UNIQUE (source_table, source_column, raw_value)
            );
        """)

        # 兼容旧库：若存在老的 entity_type 列则保留，但设为可空
        cur.execute("""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'value_embeddings' AND column_name = 'entity_type'
                ) THEN
                    ALTER TABLE value_embeddings ALTER COLUMN entity_type DROP NOT NULL;
                END IF;
            END$$;
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_value_embeddings_source
            ON value_embeddings (source_table, source_column);
        """)

        # HNSW 向量索引（若 pgvector 支持）
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_value_embeddings_hnsw
                ON value_embeddings USING hnsw (embedding vector_cosine_ops);
            """)
        except Exception as e:
            logger.warning(f"HNSW 索引创建失败（可能 pgvector 版本太旧）: {e}")

        # 支持 pg_trgm 的 trigram 索引（加速 similarity() 查询）
        try:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_value_embeddings_trgm
                ON value_embeddings USING gin (raw_value gin_trgm_ops);
            """)
        except Exception as e:
            logger.warning(f"trigram 索引创建失败: {e}")

        logger.info("✅ value_embeddings 表检查/创建完成")


def get_all_candidate_columns(conn) -> List[Tuple[str, str, str, bool]]:
    """动态获取候选列（排除忽略表和黑名单列）。"""
    blacklist_tables = _build_blacklist_tables(conn)

    with conn.cursor() as cur:
        base_sql = """
            SELECT c.table_name, c.column_name, c.data_type, c.udt_name
            FROM information_schema.columns c
            JOIN information_schema.tables t
                ON c.table_schema = t.table_schema
                AND c.table_name = t.table_name
            WHERE c.table_schema = %s
              AND t.table_type = 'BASE TABLE'
        """
        params = [TARGET_SCHEMA]

        if WHITELIST_TABLES:
            base_sql += " AND c.table_name = ANY(%s)"
            params.append(WHITELIST_TABLES)

        if blacklist_tables:
            base_sql += " AND c.table_name != ALL(%s)"
            params.append(list(blacklist_tables))

        base_sql += " ORDER BY c.table_name, c.ordinal_position;"

        cur.execute(base_sql, params)

        candidates = []
        for row in cur.fetchall():
            table, col, dtype, udt_name = row
            if col.lower() in BLACKLIST_COLUMNS:
                continue
            is_array = dtype.upper() == "ARRAY" or udt_name.startswith("_")
            if dtype.lower() in EMBEDDABLE_TYPES or is_array:
                candidates.append((table, col, dtype, is_array))

        return candidates


def get_distinct_values(conn, table: str, column: str, is_array: bool) -> List[str]:
    """安全获取 DISTINCT 值"""
    with conn.cursor() as cur:
        tb_id = sql.Identifier(table)
        col_id = sql.Identifier(column)

        if is_array:
            query = sql.SQL("""
                SELECT DISTINCT unnest({col}) AS val
                FROM {tb}
                WHERE {col} IS NOT NULL
            """).format(col=col_id, tb=tb_id)
        else:
            query = sql.SQL("""
                SELECT DISTINCT {col} AS val
                FROM {tb}
                WHERE {col} IS NOT NULL
                  AND {col}::text != ''
            """).format(col=col_id, tb=tb_id)

        cur.execute(query)
        return [str(row[0]).strip() for row in cur.fetchall() if row[0] and str(row[0]).strip()]


def get_existing_values_for_column(conn, table: str, column: str) -> Set[str]:
    """按列获取已存在的 raw_value"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT raw_value
            FROM value_embeddings
            WHERE source_table = %s AND source_column = %s
        """, (table, column))
        return {row[0] for row in cur.fetchall()}


def refresh_value_embeddings(incremental: bool = True):
    """主 ETL 函数"""
    try:
        with psycopg.connect(DB_URL, autocommit=False) as conn:
            create_value_embeddings_table_if_not_exists(conn)
            conn.commit()

            if not incremental:
                logger.info("全量模式：清空旧数据...")
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE value_embeddings RESTART IDENTITY;")
                conn.commit()

            logger.info("正在发现数据库中的候选字段...")
            candidates = get_all_candidate_columns(conn)
            logger.info(f"发现 {len(candidates)} 个可嵌入的候选字段")

            total_inserted = 0
            total_skipped = 0

            for table, col, dtype, is_array in candidates:
                logger.info(f"处理 {table}.{col} (type: {dtype}, array: {is_array})")

                try:
                    values = get_distinct_values(conn, table, col, is_array)
                    if not values:
                        logger.info(f"  → 无有效值，跳过")
                        continue

                    existing = get_existing_values_for_column(conn, table, col) if incremental else set()
                    new_values = [v for v in values if v not in existing]
                    total_skipped += (len(values) - len(new_values))

                    if not new_values:
                        logger.info(f"  → 全部已存在，跳过")
                        continue

                    logger.info(f"  → 需要生成 embedding 的新值数量: {len(new_values)}")

                    embeddings = model.encode(
                        new_values,
                        batch_size=BATCH_SIZE,
                        normalize_embeddings=True,
                        show_progress_bar=True
                    )

                    data = [
                        (table, col, val, emb.tolist())
                        for val, emb in zip(new_values, embeddings)
                    ]

                    with conn.cursor() as cur:
                        cur.executemany("""
                            INSERT INTO value_embeddings
                            (source_table, source_column, raw_value, embedding)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (source_table, source_column, raw_value)
                            DO NOTHING
                        """, data)

                    conn.commit()
                    total_inserted += len(data)
                    logger.info(f"  → 已插入 {len(data)} 条新记录")

                except Exception as inner_e:
                    logger.error(f"  ❌ 处理 {table}.{col} 失败: {inner_e}")
                    conn.rollback()
                    continue

            logger.info(f"✅ ETL 执行完成！新增 {total_inserted} 条，跳过 {total_skipped} 条")

    except Exception as e:
        logger.error(f"💥 ETL 严重失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    refresh_value_embeddings(incremental=True)
