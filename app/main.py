# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import logging
import time
import sys

# ====================== 日志配置（必须在其他模块导入前设置） ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from app.db import get_connection, close_pool, get_pool
from app.multi_agent import run_text2geosql


# ====================== 生命周期管理 ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info("Service starting...")
    logger.info("=" * 70)

    db_ok = False
    try:
        logger.info("Checking database connection...")
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_ok = True
        logger.info("Database connection OK")
    except Exception as e:
        logger.error("Database connection failed: %s", str(e)[:200])

    if not db_ok:
        logger.warning("Database unavailable, service starting in degraded mode")

    logger.info("=" * 70)
    logger.info("Service ready")
    logger.info("  Listen: http://127.0.0.1:8000")
    logger.info("  Health: http://127.0.0.1:8000/health")
    logger.info("  Query:  POST http://127.0.0.1:8000/query")
    logger.info("=" * 70)

    yield

    close_pool()
    logger.info("=" * 70)
    logger.info("Service stopped")
    logger.info("=" * 70)


app = FastAPI(title="空间知识库 - Text2GeoSQL", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    logger.info(">>> %s %s", request.method, request.url.path)
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info("<<< %s %s -> %d (%.2fs)", request.method, request.url.path, response.status_code, elapsed)
    return response


def _build_results(rows, col_names, max_rows=10):
    """Build structured result list from raw DB rows."""
    results = []
    for row in rows[:max_rows]:
        row_dict = dict(zip(col_names, row))

        # Find best title field
        title = "无标题"
        for target in ['name', 'title', 'id', 'code']:
            found_key = next((k for k in col_names if target in k.lower()), None)
            if found_key:
                title = str(row_dict[found_key])
                break
        else:
            title = str(row[0]) if len(row) > 0 else "无标题"

        # Find best content field
        content = ""
        for target in ['description', 'type', 'status', 'value']:
            found_key = next((k for k in col_names if target in k.lower()), None)
            if found_key:
                content = str(row_dict[found_key])
                break
        else:
            content = str(row[1]) if len(row) > 1 else title

        results.append({
            "title": title,
            "content": content,
            "metadata": row_dict,
        })

    return results


@app.post("/query")
async def query(request: QueryRequest):
    try:
        logger.info("Query received: %s", request.query)

        # Run multi-agent Text2GeoSQL pipeline
        pipeline_result = await run_text2geosql(request.query)

        sql = pipeline_result.get("sql", "")
        error = pipeline_result.get("error")
        logger.info("Generated SQL: %s", sql[:300] if sql else "None")

        if not sql:
            raise HTTPException(500, "未能生成有效 SQL")

        # Use results from pipeline if available (avoid double execution)
        rows = pipeline_result.get("query_results")
        col_names = pipeline_result.get("column_names")

        if rows is not None and col_names is not None:
            logger.info("Using cached results from pipeline (%d rows)", len(rows))
        else:
            # Fallback: execute SQL (e.g., after auto-fix)
            logger.info("Executing SQL (fallback)...")
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SET statement_timeout = '10s'; {sql}")
                    rows = cur.fetchall()
                    col_names = [desc[0] for desc in cur.description] if cur.description else []

        logger.info("SQL result: %d rows, columns: %s", len(rows), col_names)

        results = _build_results(rows, col_names)
        logger.info("Returning %d results", len(results))

        return {
            "sql": sql,
            "results": results,
            "message": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.get("/health")
async def health():
    db_ok = False
    try:
        with get_connection(timeout=3.0) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_ok = True
    except Exception as e:
        logger.warning("Health check - DB unreachable: %s", e)

    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
        "message": "Text2GeoSQL service running"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("=" * 70)
    logger.info("Starting uvicorn...")
    logger.info("=" * 70)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
