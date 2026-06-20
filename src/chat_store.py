"""
对话持久化存储：SQLAlchemy 异步 ORM。
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional

from sqlalchemy import select, delete, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import AsyncSessionLocal, async_engine, Base
from src.models import Conversation, Message, Feedback


async def ensure_tables():
    """创建所有缺失的表（幂等）。"""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ================================================================
# 对话管理
# ================================================================

async def create_conversation(title: str = "新对话", cid: str = None) -> Dict:
    if cid is None:
        cid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    async with AsyncSessionLocal() as session:
        conv = Conversation(id=cid, title=title, created_at=now, updated_at=now)
        session.add(conv)
        await session.commit()
    return {"id": cid, "title": title, "created_at": now, "updated_at": now}


async def list_conversations() -> List[Dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Conversation).order_by(Conversation.updated_at.desc())
        )
        return [c.to_dict() for c in result.scalars().all()]


async def get_conversation(cid: str) -> Optional[Dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Conversation).where(Conversation.id == cid)
        )
        conv = result.scalar_one_or_none()
        return conv.to_dict() if conv else None


async def update_title(cid: str, title: str):
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Conversation)
            .where(Conversation.id == cid)
            .values(title=title)
        )
        await session.commit()


async def delete_conversation(cid: str):
    """删除对话及其所有消息（ORM cascade 自动处理）。"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Conversation).where(Conversation.id == cid)
        )
        conv = result.scalar_one_or_none()
        if conv:
            await session.delete(conv)
            await session.commit()


# ================================================================
# 消息管理
# ================================================================

async def add_message(cid: str, role: str, content: str):
    async with AsyncSessionLocal() as session:
        msg = Message(conversation_id=cid, role=role, content=content)
        session.add(msg)
        # 同步更新对话的 updated_at
        await session.execute(
            update(Conversation)
            .where(Conversation.id == cid)
            .values(updated_at=datetime.now().isoformat())
        )
        await session.commit()


async def get_messages(cid: str, limit: int = 50) -> List[Dict]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message.role, Message.content)
            .where(Message.conversation_id == cid)
            .order_by(Message.id.asc())
            .limit(limit)
        )
        return [{"role": r[0], "content": r[1]} for r in result.all()]


async def add_messages_batch(cid: str, messages: list[dict]):
    """批量写入消息（一次 commit，替代逐条 add_message）。"""
    if not messages:
        return
    async with AsyncSessionLocal() as session:
        for msg in messages:
            session.add(Message(
                conversation_id=cid,
                role=msg["role"],
                content=msg["content"],
            ))
        await session.execute(
            update(Conversation)
            .where(Conversation.id == cid)
            .values(updated_at=datetime.now().isoformat())
        )
        await session.commit()


# ================================================================
# 用户反馈
# ================================================================

async def add_feedback(cid: str, rating: str, comment: str = "",
                      context: str = "") -> int:
    """记录用户反馈。context 为 JSON 格式的对话上下文（消息历史）。"""
    async with AsyncSessionLocal() as session:
        fb = Feedback(
            conversation_id=cid, rating=rating,
            comment=comment, context=context or None,
        )
        session.add(fb)
        await session.commit()
        await session.refresh(fb)
        return fb.id


async def get_feedback_stats() -> Dict:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Feedback.rating, func.count()).group_by(Feedback.rating)
        )
        rows = result.all()
    stats = {"positive": 0, "negative": 0, "total": 0}
    for rating, count in rows:
        stats[rating] = count
    stats["total"] = sum(stats.values())
    return stats
