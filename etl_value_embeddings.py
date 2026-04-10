import psycopg
from psycopg import sql
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Set

# ====================== 配置 ======================
DB_URL = "postgresql://postgres:postgres123@localhost:5432/city_planning"
BATCH_SIZE = 512                    
MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024                # bge-m3 维度固定为 1024

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("正在加载 Embedding 模型...")
model = SentenceTransformer(MODEL_NAME)

# ==================== 动态配置 ====================
TARGET_SCHEMA = "public"
WHITELIST_TABLES: List[str] = []   
BLACKLIST_COLUMNS = {
    "id", "created_at", "updated_at", "geom", "geometry", "location",
    "uuid", "code", "key", "password", "email", "phone", "gid"
}
EMBEDDABLE_TYPES = {
    "character varying", "varchar", "text", "character", "name",
    "_text", "_varchar", "ARRAY"
}

def create_value_embeddings_table_if_not_exists(conn):
    """自动创建 value_embeddings 表（建议数据库预先安装 pgvector 插件）"""
    with conn.cursor() as cur:
        # 尝试开启 pgvector 支持 (如果数据库支持的话，大大提升后期检索速度)
        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            vector_type = f"VECTOR({EMBEDDING_DIM})"
        except Exception as e:
            logger.warning(f"无法创建 pgvector 扩展，将回退使用 FLOAT[]: {e}")
            conn.rollback() # 捕获异常后回滚当前小事务
            vector_type = "FLOAT[]"
            
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
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_value_embeddings_source 
            ON value_embeddings (source_table, source_column);
        """)
        logger.info("✅ value_embeddings 表检查/创建完成")


def get_all_candidate_columns(conn) -> List[Tuple[str, str, str, bool]]:
    """动态获取所有候选列（修复参数类型问题）"""
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
        
        # 动态拼接白名单，避免 ANY(None) 的 psycopg3 报错
        if WHITELIST_TABLES:
            base_sql += " AND c.table_name = ANY(%s)"
            params.append(WHITELIST_TABLES)
            
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
    """获取 DISTINCT 值（修复 SQL 注入和保留字问题）"""
    with conn.cursor() as cur:
        # 使用 psycopg.sql.Identifier 安全转义表名和列名
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
        # 清理空数据
        return [str(row[0]).strip() for row in cur.fetchall() if row[0] and str(row[0]).strip()]


def get_existing_values_for_column(conn, table: str, column: str) -> Set[str]:
    """【内存优化】按需获取当前字段已存在的 value，防止全局查询撑爆内存"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT raw_value FROM value_embeddings 
            WHERE source_table = %s AND source_column = %s
        """, (table, column))
        return {row[0] for row in cur.fetchall()}


def refresh_value_embeddings(incremental: bool = True):
    """主 ETL 逻辑（修复事务问题）"""
    try:
        # 使用连接，开启手动事务控制
        with psycopg.connect(DB_URL, autocommit=False) as conn:
            
            # 第一步：建表
            create_value_embeddings_table_if_not_exists(conn)
            conn.commit()  # DDL 立即提交

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

            # 第二步：循环处理
            for table, col, dtype, is_array in candidates:
                logger.info(f"处理 {table}.{col} (type: {dtype}, array: {is_array})")
                
                try:
                    values = get_distinct_values(conn, table, col, is_array)
                    if not values:
                        logger.info(f"  → 无有效值，跳过")
                        continue

                    # 按需加载当前列已存在的特征，节省内存
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

                    # 准备插入数据 (如果使用 pgvector，也可以直接存入 list)
                    data = [
                        (table, col, val, emb.tolist())
                        for val, emb in zip(new_values, embeddings)
                    ]

                    # 插入
                    with conn.cursor() as cur:
                        cur.executemany("""
                            INSERT INTO value_embeddings 
                            (source_table, source_column, raw_value, embedding)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (source_table, source_column, raw_value) 
                            DO NOTHING
                        """, data)

                    # 【关键修复】每处理完一个字段就 Commit，释放锁，防止错误引发全局回滚
                    conn.commit()
                    
                    total_inserted += len(data)
                    logger.info(f"  → 已成功插入 {len(data)} 条新记录并提交")
                    
                except Exception as inner_e:
                    # 单个字段报错，只回滚当前字段，不影响其他表
                    logger.error(f"  ❌ 处理 {table}.{col} 失败: {inner_e}")
                    conn.rollback()
                    continue

            logger.info(f"✅ 动态实体向量更新完成！新增 {total_inserted} 条，跳过 {total_skipped} 条")

    except Exception as e:
        logger.error(f"💥 ETL 严重失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    refresh_value_embeddings(incremental=True)