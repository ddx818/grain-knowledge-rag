"""
SQLAlchemy 引擎与会话管理。

双引擎设计：
  sync_engine  → grain_db.py（同步 @tool 上下文）
  async_engine → chat_store.py（FastAPI 异步上下文）
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, DeclarativeBase

# ── 连接信息 ──────────────────────────────────────────────
DB_USER = "root"
DB_PASS = "198417"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "agent_db"

SYNC_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
ASYNC_URL = f"mysql+aiomysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# ── 同步引擎（grain_db 使用） ─────────────────────────────
sync_engine = create_engine(
    SYNC_URL,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    pool_pre_ping=True,
)
SyncSessionLocal = sessionmaker(bind=sync_engine)

# ── 异步引擎（chat_store 使用） ───────────────────────────
async_engine = create_async_engine(
    ASYNC_URL,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
    pool_pre_ping=True,
)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False,
)


# ── 声明式基类 ────────────────────────────────────────────
class Base(DeclarativeBase):
    pass
