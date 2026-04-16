# app/db.py - Database connection pool management
import os
import logging
from contextlib import contextmanager
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()
logger = logging.getLogger(__name__)

DB_URL = os.getenv("DB_URL", "postgresql://postgres:postgres123@localhost:5432/spatial_kb")

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Get or create the global connection pool (lazy singleton)."""
    global _pool
    if _pool is None:
        logger.info("Initializing database connection pool...")
        _pool = ConnectionPool(
            conninfo=DB_URL,
            min_size=2,
            max_size=10,
            kwargs={"autocommit": True},
        )
        logger.info("Database connection pool initialized (min=2, max=10)")
    return _pool


@contextmanager
def get_connection(timeout: float = 5.0):
    """Acquire a connection from the pool.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    pool = get_pool()
    conn = pool.getconn(timeout=timeout)
    try:
        yield conn
    finally:
        pool.putconn(conn)


def close_pool():
    """Shutdown the pool on app exit."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("Database connection pool closed")
