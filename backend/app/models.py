from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import UniqueConstraint, Index
from sqlmodel import Field, Relationship, SQLModel


class Stock(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    ticker: str = Field(index=True, nullable=False, max_length=32)
    name: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    prices: list["PriceHistory"] = Relationship(back_populates="stock")


class PriceHistory(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("stock_code", "date", name="uix_stock_code_date"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    stock_id: Optional[int] = Field(default=None, foreign_key="stock.id")
    stock_code: str = Field(index=True, nullable=False, max_length=32)
    date: date = Field(index=True)
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    adjusted_close: Optional[float] = None
    volume: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    stock: Optional["Stock"] = Relationship(back_populates="prices")


class NewsItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    stock_code: str = Field(index=True, nullable=False, max_length=32)
    title: str = Field(nullable=False)
    url: Optional[str] = None
    published_at: Optional[date] = Field(index=True)
    content: Optional[str] = None
    source: Optional[str] = None
    sentiment_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, nullable=False, max_length=64)
    checksum: Optional[str] = Field(default=None, max_length=128)
    source: Optional[str] = Field(default=None, max_length=256)
    task: str = Field(default="sentiment")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SentimentScore(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("news_id", "model_version_id", name="uix_news_model"),
        Index("idx_score_stock_date", "stock_code", "published_at"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    news_id: int = Field(foreign_key="newsitem.id", nullable=False)
    model_version_id: int = Field(foreign_key="modelversion.id", nullable=False)
    stock_code: str = Field(index=True, nullable=False, max_length=32)
    published_at: date = Field(index=True)
    score: float = Field(nullable=False)
    confidence: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Signal(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("stock_code", "as_of_date", "model_version_id", name="uix_signal_date_model"),
        Index("idx_signal_rank", "as_of_date", "signal_score"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    stock_code: str = Field(index=True, max_length=32)
    as_of_date: date = Field(index=True)
    model_version_id: int = Field(foreign_key="modelversion.id", nullable=False)
    signal_score: float = Field(nullable=False)
    details: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BacktestResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    model_version_id: int = Field(foreign_key="modelversion.id", nullable=False)
    strategy: str = Field(index=True, max_length=64)
    start_date: date
    end_date: date
    annual_return: float
    max_drawdown: float
    sharpe: Optional[float] = None
    trades: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
