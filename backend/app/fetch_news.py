import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import requests
from sqlmodel import Session, select

from .models import NewsItem


def _load_news_tiingo(code: str, days: int = 45) -> pd.DataFrame:
    api_key = os.getenv("TIINGO_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    url = "https://api.tiingo.com/tiingo/news"
    params = {
        "tickers": code.lower(),
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "limit": 200,
        "token": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return pd.DataFrame()

    rows: List[Dict] = []
    for a in data:
        rows.append({
            "published_at": (a.get("publishedDate") or "")[:10],
            "title": a.get("title") or "",
            "content": a.get("description") or "",
            "source": a.get("source") or "tiingo",
        })
    df = pd.DataFrame(rows)
    df = df[df["title"].astype(str).str.strip() != ""]
    return df


def _load_news_csv_fallback(code: str) -> pd.DataFrame:
    c = code.lower()
    candidates = [
        Path(__file__).resolve().parents[2] / "data" / f"news_{c}.csv",
        Path(__file__).resolve().parents[1] / "data" / f"news_{c}.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        rename_map = {"date": "published_at", "time": "published_at", "正文": "content", "标题": "title", "来源": "source"}
        df = df.rename(columns=rename_map)
        for col in ["published_at", "title", "content", "source"]:
            if col not in df.columns:
                df[col] = ""
        df["published_at"] = df["published_at"].astype(str).str.slice(0, 10)
        df = df[df["title"].astype(str).str.strip() != ""]
        return df
    return pd.DataFrame()


def import_news_from_csv_or_api(code: str, engine) -> int:
    code = code.upper()

    # 1) 在线 Tiingo
    df = _load_news_tiingo(code, days=45)

    # 2) 回退 CSV
    if df.empty:
        df = _load_news_csv_fallback(code)

    if df.empty:
        return 0

    inserted = 0
    with Session(engine) as session:
        for _, r in df.iterrows():
            published_at = str(r.get("published_at", "")).strip()
            title = str(r.get("title", "")).strip()
            content = str(r.get("content", "")).strip()
            source = str(r.get("source", "")).strip() or "unknown"

            if not title:
                continue
            if not published_at:
                published_at = datetime.utcnow().date().isoformat()

            existed = session.exec(
                select(NewsItem).where(
                    NewsItem.stock_code == code,
                    NewsItem.published_at == published_at,
                    NewsItem.title == title,
                )
            ).first()
            if existed:
                continue

            session.add(NewsItem(
                stock_code=code,
                published_at=published_at,
                title=title,
                content=content,
                source=source,
            ))
            inserted += 1

        session.commit()

    return inserted
