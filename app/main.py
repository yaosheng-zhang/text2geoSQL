# app/main.py - 完整功能版
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import psycopg
import json
import logging
import time

# ====================== 日志配置（必须在其他模块导入前设置） ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from app.config import DB_URL
from app.multi_agent import run_text2geosql


# ====================== 生命周期管理 ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动/关闭时的生命周期管理"""
    logger.info("========== 服务启动中 ==========")
    # 检查数据库连接
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        logger.info("数据库连接正常: %s", DB_URL.split("@")[-1])
    except Exception as e:
        logger.error("数据库连接失败: %s", e)
        logger.error("请确认 PostgreSQL 服务已启动，且连接信息正确")

    logger.info("========== 服务启动完成，监听 http://127.0.0.1:8000 ==========")
    yield
    logger.info("========== 服务关闭 ==========")


app = FastAPI(title="空间知识库 - 完整版 (BAAI/bge-m3)", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录每个请求的耗时和状态"""
    start = time.time()
    logger.info(">>> %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info("<<< %s %s -> %d (%.2fs)", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.post("/query")
async def query(request: QueryRequest):
    try:
        logger.info("收到查询: %s", request.query)

        # 调用多 Agent Text2GeoSQL
        sql = await run_text2geosql(request.query)
        logger.info("生成 SQL: %s", sql[:300] if sql else "None")

        if not sql:
            raise HTTPException(500, "未能生成有效 SQL")

        # 执行 SQL
        with psycopg.connect(DB_URL, options="-c statement_timeout=10000") as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                col_names = [desc[0] for desc in cur.description] if cur.description else []

        logger.info("SQL 执行成功，返回 %d 行, 列: %s", len(rows), col_names)

        results = []
        for row in rows[:10]:
                    # 安全解析 metadata（推荐写一个辅助函数）
                    metadata = {}
                    if len(row) > 3 and row[3] is not None:
                        metadata_str = str(row[3]).strip() if isinstance(row[3], (str, bytes)) else ""
                        if metadata_str:
                            try:
                                metadata = json.loads(metadata_str)
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.warning("JSON 解析失败，原始值: %s, 错误: %s", repr(row[3]), e)
                                metadata = {}
        
                    results.append({
                        "title": row[1] if len(row) > 1 else None,      # 根据你的列顺序调整索引
                        "content": str(row[0]) if len(row) > 0 else "",
                        "metadata": metadata
                    })

        logger.info("查询成功，返回 %d 条结果", len(results))
        return {
            "sql": sql,
            "results": results,
            "message": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("查询失败: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.get("/health")
async def health():
    """健康检查 - 同时检测数据库连通性"""
    db_ok = False
    try:
        with psycopg.connect(DB_URL, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_ok = True
    except Exception as e:
        logger.warning("健康检查 - 数据库不可达: %s", e)

    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
        "message": "空间知识库完整版服务正常运行"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)