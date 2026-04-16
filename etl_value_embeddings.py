import psycopg
from psycopg import sql
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Set

# ====================== 配置 ======================
DB_URL = "postgresql://postgres:postgres123@localhost:5432/city_planning"
BATCH_SIZE = 512                    
MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024                # bge-m3 维度

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("正在加载 Embedding 模型...")
model = SentenceTransformer(MODEL_NAME)

# ==================== 动态配置 ====================
TARGET_SCHEMA = "public"
WHITELIST_TABLES: List[str] = []   

BLACKLIST_COLUMNS = {
    "id", "created_at", "updated_at", "geom", "geometry", "location",
    "uuid", "code", "key", "password", "email", "phone", "gid", "embedding"
}

BLACKLIST_TABLES = {
    "value_embeddings",        # 防止自处理
    "pg_stat_statements",
    # 可继续添加其他系统表或不想处理的表
}

EMBEDDABLE_TYPES = {
    "character varying", "varchar", "text", "character", "name",
    "_text", "_varchar", "ARRAY"
}

def create_value_embeddings_table_if_not_exists(conn):
    """自动创建带 entity_type 的 value_embeddings 表"""
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            vector_type = f"VECTOR({EMBEDDING_DIM})"
            logger.info("✅ 已启用 pgvector 扩展")
        except Exception as e:
            logger.warning(f"pgvector 不可用，回退使用 FLOAT[]: {e}")
            vector_type = "FLOAT[]"

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS value_embeddings (
                id SERIAL PRIMARY KEY,
                source_table TEXT NOT NULL,
                source_column TEXT NOT NULL,
                raw_value TEXT NOT NULL,
                embedding {vector_type} NOT NULL,
                entity_type TEXT NOT NULL,                    -- 新增：表名_字段名
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                CONSTRAINT uk_value_embeddings 
                UNIQUE (source_table, source_column, raw_value)
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_value_embeddings_source 
            ON value_embeddings (source_table, source_column);
            
            CREATE INDEX IF NOT EXISTS idx_value_embeddings_entity 
            ON value_embeddings (entity_type);
        """)

        logger.info("✅ value_embeddings 表检查/创建完成（包含 entity_type）")


def get_all_candidate_columns(conn) -> List[Tuple[str, str, str, bool]]:
    """动态获取候选列 - 排除自身表"""
    with conn.cursor() as cur:
        base_sql = """
            SELECT c.table_name, c.column_name, c.data_type, c.udt_name
            FROM information_schema.columns c
            JOIN information_schema.tables t 
                ON c.table_schema = t.table_schema 
                AND c.table_name = t.table_name
            WHERE c.table_schema = %s
              AND t.table_type = 'BASE TABLE'
              AND c.table_name != 'value_embeddings'
        """
        params = [TARGET_SCHEMA]

        if WHITELIST_TABLES:
            base_sql += " AND c.table_name = ANY(%s)"
            params.append(WHITELIST_TABLES)

        if BLACKLIST_TABLES:
            base_sql += " AND c.table_name != ALL(%s)"
            params.append(list(BLACKLIST_TABLES))

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
            # 创建表
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

                    # 生成 embedding
                    embeddings = model.encode(
                        new_values,
                        batch_size=BATCH_SIZE,
                        normalize_embeddings=True,
                        show_progress_bar=True
                    )

                    # entity_type = table_col （表名_字段名）
                    entity_type = f"{table}_{col}"

                    # 准备数据：增加 entity_type
                    data = [
                        (table, col, val, emb.tolist(), entity_type)
                        for val, emb in zip(new_values, embeddings)
                    ]

                    with conn.cursor() as cur:
                        cur.executemany("""
                            INSERT INTO value_embeddings 
                            (source_table, source_column, raw_value, embedding, entity_type)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (source_table, source_column, raw_value) 
                            DO NOTHING
                        """, data)

                    conn.commit()
                    total_inserted += len(data)
                    logger.info(f"  → 已插入 {len(data)} 条新记录 (entity_type: {entity_type})")

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