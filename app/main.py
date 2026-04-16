# app/main.py - 完整功能版
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import psycopg
import json
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

from app.config import DB_URL
from app.multi_agent import run_text2geosql


# ====================== 生命周期管理 ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动/关闭时的生命周期管理"""
    logger.info("=" * 70)
    logger.info("🚀 服务启动中...")
    logger.info("=" * 70)

    # 检查数据库连接
    db_ok = False
    try:
        logger.info("📋 检查数据库连接...")
        with psycopg.connect(DB_URL, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_ok = True
        db_host = DB_URL.split("@")[-1].split("/")[0]
        logger.info("✅ 数据库连接成功: %s", db_host)
    except psycopg.OperationalError as e:
        logger.error("❌ 数据库连接失败 (OperationalError): %s", str(e)[:200])
        logger.error("   可能原因:")
        logger.error("   1. PostgreSQL 服务未启动")
        logger.error("   2. 连接信息错误 (用户名/密码/主机/端口)")
        logger.error("   3. 防火墙阻止连接")
        logger.error("   DB_URL: %s", DB_URL.split("@")[0] + "@***")
    except Exception as e:
        logger.error("❌ 数据库连接异常: %s", e)

    if not db_ok:
        logger.warning("⚠️  数据库不可用，服务将以降级模式启动")
        logger.warning("   /health 端点将返回 'degraded' 状态")
        logger.warning("   /query 端点可能无法正常工作")

    logger.info("=" * 70)
    logger.info("✅ 服务启动完成")
    logger.info("   监听地址: http://127.0.0.1:8000")
    logger.info("   健康检查: http://127.0.0.1:8000/health")
    logger.info("   查询接口: POST http://127.0.0.1:8000/query")
    logger.info("=" * 70)

    yield

    logger.info("=" * 70)
    logger.info("🛑 服务关闭")
    logger.info("=" * 70)


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
        # 建议：使用 psycopg.rows.dict_row 可以直接获取字典格式的结果
        with psycopg.connect(DB_URL, options="-c statement_timeout=10000") as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                # 获取列名
                col_names = [desc[0] for desc in cur.description] if cur.description else []

        logger.info("SQL 执行成功，返回 %d 行, 列: %s", len(rows), col_names)

        results = []
        for row in rows[:10]:
            # 1. 将行数据与列名合并为字典 (通用解析核心)
            row_dict = dict(zip(col_names, row))
            
            # 2. 尝试从列中寻找最适合做 "title" 的字段
            # 优先级：包含 'name' 的列 > 包含 'id' 的列 > 包含 'code' 的列 > 第一列
            title = "无标题"
            potential_title_keys = ['name', 'title', 'id', 'code']
            for target in potential_title_keys:
                found_key = next((k for k in col_names if target in k.lower()), None)
                if found_key:
                    title = str(row_dict[found_key])
                    break
            else:
                title = str(row[0]) if len(row) > 0 else "无标题"

            # 3. 尝试寻找最适合做 "content" 的字段
            # 优先级：包含 'description' > 包含 'type' > 第二列 > 同 title
            content = ""
            potential_content_keys = ['description', 'type', 'status', 'value']
            for target in potential_content_keys:
                found_key = next((k for k in col_names if target in k.lower()), None)
                if found_key:
                    content = str(row_dict[found_key])
                    break
            else:
                content = str(row[1]) if len(row) > 1 else title

            # 4. 构建结果（将整行 row_dict 作为 metadata）
            # 注意：psycopg 默认会把 JSONB 列解析成 dict，不需要手动 json.loads
            results.append({
                "title": title,
                "content": content,
                "metadata": row_dict  # 包含所有返回的列
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
    logger.info("=" * 70)
    logger.info("📦 启动参数:")
    logger.info("   Host: 127.0.0.1")
    logger.info("   Port: 8000")
    logger.info("   Workers: 1")
    logger.info("=" * 70)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )