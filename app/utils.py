# app/utils.py - 优化后的 SQL 安全检查
import sqlparse
import logging
import re

logger = logging.getLogger(__name__)

def is_safe_sql(sql: str) -> bool:
    """
    优化后的 SQL 安全检查：
    - 允许 SELECT、WITH (CTE)、UNION 等只读查询
    - 严格禁止修改数据库的操作
    - 支持 PostGIS 空间函数
    """
    if not sql or not isinstance(sql, str):
        logger.warning("SQL 安全检查失败: SQL 为空或非字符串")
        return False

    sql_upper = sql.strip().upper()
    sql_clean = sql.strip()

    # 1. 基础检查：必须以 SELECT 或 WITH 开头（允许 CTE）
    if not (sql_upper.startswith("SELECT") or 
            sql_upper.startswith("WITH") or 
            sql_upper.startswith("(")):   # 支持子查询
        logger.warning("SQL 安全检查失败: 非 SELECT/WITH 查询 → %s", sql_clean[:100])
        return False

    # 2. 禁止危险关键词（无论大小写）
    dangerous_keywords = [
        "DELETE", "UPDATE", "INSERT", "DROP", "ALTER", "CREATE", "TRUNCATE",
        "GRANT", "REVOKE", "EXECUTE", "EXEC", "CALL", "PROCEDURE", "FUNCTION"
    ]

    for keyword in dangerous_keywords:
        # 使用词边界避免误匹配（如 "updated" 不应该被拦截）
        if re.search(rf'\b{keyword}\b', sql_upper):
            logger.warning("SQL 安全检查失败: 检测到危险关键词 '%s'", keyword)
            return False

    # 3. 简单语法检查（使用 sqlparse 解析）
    try:
        parsed = sqlparse.parse(sql_clean)
        if not parsed:
            logger.warning("SQL 安全检查失败: 无法解析 SQL")
            return False

        # 检查每个语句类型
        for statement in parsed:
            stmt_type = statement.get_type()
            if stmt_type not in ("SELECT", "UNKNOWN"):   # UNKNOWN 可能是复杂的 CTE
                logger.warning("SQL 安全检查失败: 语句类型不是 SELECT，而是 %s", stmt_type)
                return False

    except Exception as e:
        logger.warning("SQL 安全检查时解析失败，默认拒绝: %s", e)
        return False

    logger.debug("SQL 安全检查通过: %s...", sql_clean[:120])
    return True