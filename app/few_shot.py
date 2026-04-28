# app/few_shot.py - Vanna 风格 Few-Shot 闭环系统
#
# 核心思路：
# 1. 查询成功执行后，自动将 (question, sql) 对存入 sql_examples 表（带向量）
# 2. 新查询到来时，检索最相似的历史 question-SQL 对，作为 few-shot 注入 prompt
# 3. 冷启动时（无历史数据）退化为 zero-shot，随使用自然积累

import logging
from typing import List, Dict, Optional

from app.db import get_connection
from app.config import bge_embedding_model

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1024  # bge-m3 维度


def ensure_table():
    """启动时调用，确保 sql_examples 表存在。"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 确保 pgvector 扩展可用
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS sql_examples (
                        id SERIAL PRIMARY KEY,
                        question TEXT NOT NULL,
                        sql_text TEXT NOT NULL,
                        embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                        success_count INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT uk_sql_examples UNIQUE (question)
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sql_examples_embedding
                    ON sql_examples USING hnsw (embedding vector_cosine_ops);
                """)

        logger.info("sql_examples 表检查/创建完成")
    except Exception as e:
        logger.warning("sql_examples 表创建失败（few-shot 将以 zero-shot 退化运行）: %s", e)


def save_example(question: str, sql: str):
    """将成功的 question-SQL 对存入向量库。

    如果相同 question 已存在，更新 sql_text 和 success_count。
    """
    if not question or not sql:
        return

    try:
        embedding = bge_embedding_model.encode(
            question, normalize_embeddings=True
        ).tolist()
        emb_str = f"[{','.join(map(str, embedding))}]"

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO sql_examples (question, sql_text, embedding, success_count)
                    VALUES (%s, %s, %s::vector, 1)
                    ON CONFLICT (question)
                    DO UPDATE SET
                        sql_text = EXCLUDED.sql_text,
                        success_count = sql_examples.success_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                """, (question, sql, emb_str))

        logger.info("[FewShot] 已保存成功查询: '%s' (SQL 长度: %d)", question[:60], len(sql))
    except Exception as e:
        logger.warning("[FewShot] 保存失败（不影响主流程）: %s", e)


def search_similar_examples(query: str, top_k: int = 3, min_similarity: float = 0.5) -> List[Dict]:
    """检索与当前 query 最相似的历史 question-SQL 对。

    Args:
        query: 用户查询文本
        top_k: 最多返回多少条
        min_similarity: 最低余弦相似度阈值

    Returns:
        [{'question': ..., 'sql': ..., 'similarity': ...}, ...]
        如果表不存在或无数据，返回空列表（退化为 zero-shot）
    """
    try:
        embedding = bge_embedding_model.encode(
            query, normalize_embeddings=True
        ).tolist()
        emb_str = f"[{','.join(map(str, embedding))}]"

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT question, sql_text,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM sql_examples
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (emb_str, emb_str, min_similarity, emb_str, top_k))

                rows = cur.fetchall()

        results = [
            {"question": row[0], "sql": row[1], "similarity": float(row[2])}
            for row in rows
        ]

        if results:
            logger.info("[FewShot] 检索到 %d 条相似历史查询 (最高相似度: %.3f)",
                        len(results), results[0]["similarity"])
        else:
            logger.info("[FewShot] 无相似历史查询，使用 zero-shot 模式")

        return results

    except Exception as e:
        # 表不存在或其他错误 → 安静退化为 zero-shot
        logger.debug("[FewShot] 检索失败（退化为 zero-shot）: %s", e)
        return []
