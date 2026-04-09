# etl_value_embeddings.py - 工业级优化版
import psycopg
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# ====================== 配置 ======================
DB_URL = "postgresql://postgres:postgres123@localhost:5432/spatial_kb"
BATCH_SIZE = 512                    # 根据显存调整，推荐 256~1024
MODEL_NAME = "BAAI/bge-m3"          # 中文强，推荐本地部署时换成 bge-m3

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = SentenceTransformer(MODEL_NAME)

def get_distinct_values(conn, table: str, column: str, is_array: bool = False) -> List[str]:
    """安全获取 DISTINCT 值，支持普通列和数组列"""
    with conn.cursor() as cur:
        if is_array:
            sql = f"""
                SELECT DISTINCT unnest({column}) AS val 
                FROM {table} 
                WHERE {column} IS NOT NULL AND array_length({column}, 1) > 0
            """
        else:
            sql = f"""
                SELECT DISTINCT {column} AS val 
                FROM {table} 
                WHERE {column} IS NOT NULL
            """
        cur.execute(sql)
        return [row[0] for row in cur.fetchall() if row[0]]

def refresh_value_embeddings():
    """增量式 + 高性能刷新动态实体向量表"""
    try:
        with psycopg.connect(DB_URL, autocommit=False) as conn:
            with conn.cursor() as cur:
                # 可选：清空（生产建议改成增量更新，这里先保留全量方式）
                logger.info("正在清空旧的 value_embeddings 数据...")
                cur.execute("TRUNCATE TABLE value_embeddings RESTART IDENTITY;")

                sources = [
                    ("spatial_chunks", "title", "poi_name", False),
                    ("poi_details", "category", "poi_category", False),
                    ("poi_details", "tags", "poi_tag", True),           # 数组列
                    ("admin_boundaries", "ward_name", "ward", False),
                    ("admin_boundaries", "district", "district", False),
                    ("infrastructure", "name", "infrastructure_name", False),
                    ("infrastructure", "type", "infrastructure_type", False),
                ]

                total_inserted = 0

                for table, col, etype, is_array in sources:
                    logger.info(f"正在处理 {table}.{col} (type: {etype}) ...")
                    values = get_distinct_values(conn, table, col, is_array)

                    if not values:
                        logger.info(f"  → {table}.{col} 无数据，跳过")
                        continue

                    logger.info(f"  → 发现 {len(values)} 个唯一值，正在生成 embedding...")

                    # 分批 encode，降低内存峰值
                    embeddings = model.encode(
                        values, 
                        batch_size=BATCH_SIZE, 
                        normalize_embeddings=True,
                        show_progress_bar=True
                    )

                    # 准备批量插入数据
                    data: List[Tuple] = [
                        (table, col, val, emb.tolist(), etype)
                        for val, emb in zip(values, embeddings)
                    ]

                    # 使用 executemany 批量插入（性能大幅提升）
                    cur.executemany("""
                        INSERT INTO value_embeddings 
                        (source_table, source_column, raw_value, embedding, entity_type)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, data)

                    total_inserted += len(data)
                    logger.info(f"  → 已插入 {len(data)} 条记录")

                conn.commit()
                logger.info(f"✅ 动态实体向量更新完成！共插入 {total_inserted} 条记录")

    except Exception as e:
        logger.error(f"❌ ETL 执行失败: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    refresh_value_embeddings()