"""
fetch_price.py

Provides fetch_and_store_prices(code, period_days, engine) used by backend/app/main.py.

Behavior:
- Uses Tiingo to download daily OHLCV for the requested ticker for the last `period_days` days.
- If Tiingo fails or returns empty, falls back to local CSV: data/prices_<ticker>.csv
- Ensures a Stock record exists (creates/updates basic metadata).
- Inserts or updates PriceHistory rows keyed by (stock_code, date).
- Returns number of rows inserted/updated (inserted + updated).
"""
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf
from sqlmodel import Session, select

from .models import Stock, PriceHistory

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def _download_history_df_tiingo(code: str, period_days: int) -> pd.DataFrame:
    api_key = os.getenv("TIINGO_API_KEY")
    if not api_key:
        logger.warning("TIINGO_API_KEY not set, skipping Tiingo fetch")
        return pd.DataFrame()

    end = datetime.utcnow().date()
    start = end - timedelta(days=period_days)
    url = f"https://api.tiingo.com/tiingo/daily/{code}/prices"
    params = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "token": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            logger.error("Tiingo fetch failed (%s): %s", resp.status_code, resp.text[:200])
            return pd.DataFrame()
        data = resp.json()
    except Exception as exc:
        logger.error("Tiingo request failed for %s: %s", code, exc)
        return pd.DataFrame()

    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adjClose": "adjusted_close",
            "volume": "volume",
        }
    )

    for c in ["date", "open", "high", "low", "close", "adjusted_close", "volume"]:
        if c not in df.columns:
            df[c] = None

    return df[["date", "open", "high", "low", "close", "adjusted_close", "volume"]]

def _load_csv_fallback(code: str) -> pd.DataFrame:
    code_lc = code.lower()
    candidates = [
        Path(__file__).resolve().parents[2] / "data" / f"prices_{code_lc}.csv",
        Path(__file__).resolve().parents[1] / "data" / f"prices_{code_lc}.csv",
    ]
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception as exc:
                logger.error("Failed to read CSV fallback %s: %s", p, exc)
                continue

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.date
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.date

            df = df.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adjusted_close",
                    "Volume": "volume",
                }
            )
            for c in ["date", "open", "high", "low", "close", "adjusted_close", "volume"]:
                if c not in df.columns:
                    df[c] = None

            logger.info("Using CSV fallback: %s", p)
            return df[["date", "open", "high", "low", "close", "adjusted_close", "volume"]]

    logger.warning("No CSV fallback found for %s", code)
    return pd.DataFrame()

def fetch_and_store_prices(code: str, period_days: int, engine) -> int:
    """
    Fetch `period_days` of daily data for `code` and store into DB using provided SQLModel engine.
    Returns: total inserted_or_updated rows count.
    """
    code_uc = code.upper()

    # ✅ 优先 Tiingo
    df = _download_history_df_tiingo(code_uc, period_days)
    if not df.empty:
        logger.info("Data source: tiingo")

    # ✅ Tiingo 失败后回退 CSV
    if df.empty:
        logger.warning("No data fetched for %s via Tiingo, trying CSV fallback", code_uc)
        df = _load_csv_fallback(code_uc)
        if not df.empty:
            logger.info("Data source: csv")

    if df.empty:
        logger.warning("No data available for %s after fallback", code_uc)
        return 0

    inserted = 0
    updated = 0

    with Session(engine) as session:
        stmt = select(Stock).where(Stock.ticker == code_uc)
        stock = session.exec(stmt).one_or_none()
        if not stock:
            name = None
            try:
                info = yf.Ticker(code_uc).info or {}
                name = info.get("shortName") or info.get("longName")
            except Exception:
                name = None
            stock = Stock(ticker=code_uc, name=name)
            session.add(stock)
            session.commit()
            session.refresh(stock)

        for _, row in df.iterrows():
            dt = row["date"]
            stmt2 = select(PriceHistory).where(
                PriceHistory.stock_code == code_uc, PriceHistory.date == dt
            )
            existing = session.exec(stmt2).one_or_none()
            if existing:
                changed = False
                for attr, col in [
                    ("open", "open"),
                    ("high", "high"),
                    ("low", "low"),
                    ("close", "close"),
                    ("adjusted_close", "adjusted_close"),
                    ("volume", "volume"),
                ]:
                    new_val = row[col]
                    if pd.isna(new_val):
                        new_val = None
                    if getattr(existing, attr) != new_val:
                        setattr(
                            existing,
                            attr,
                            int(new_val) if attr == "volume" and new_val is not None else new_val,
                        )
                        changed = True
                if changed:
                    session.add(existing)
                    updated += 1
            else:
                ph = PriceHistory(
                    stock_id=stock.id,
                    stock_code=code_uc,
                    date=dt,
                    open=row["open"] if not pd.isna(row["open"]) else None,
                    high=row["high"] if not pd.isna(row["high"]) else None,
                    low=row["low"] if not pd.isna(row["low"]) else None,
                    close=row["close"] if not pd.isna(row["close"]) else None,
                    adjusted_close=row["adjusted_close"]
                    if not pd.isna(row["adjusted_close"])
                    else None,
                    volume=int(row["volume"]) if not pd.isna(row["volume"]) else None,
                )
                session.add(ph)
                inserted += 1
        session.commit()

    logger.info("fetch_and_store_prices %s: inserted=%d updated=%d", code_uc, inserted, updated)
    return inserted + updated