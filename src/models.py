"""
SQLAlchemy ORM 模型。
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    String, Text, Integer, Float, ForeignKey, TIMESTAMP, Index,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(255), default="新对话")
    created_at: Mapped[str] = mapped_column(String(32))
    updated_at: Mapped[str] = mapped_column(String(32))

    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    rating: Mapped[str] = mapped_column(
        String(10), nullable=False
    )
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP, default=datetime.now
    )


class GrainMonitoring(Base):
    """粮仓监测数据 —— 映射现有表 grain_monitoring。"""

    __tablename__ = "grain_monitoring"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hwdm: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    grain_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    check_date: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    check_time: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    inner_humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    outer_humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    inner_temper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    outer_temper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_temper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    min_temper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_temper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    moisture_content: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    impurity_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    imperfect_grain: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fatty_acid_ester: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unit_weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    production_area: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    algorithm_analysis_conclusion: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
