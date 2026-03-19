-- SQL schema equivalent for stocks and price_history (SQLite/Postgres compatible)
-- Run with psql or sqlite3 accordingly (adjust types if necessary for your DB)

CREATE TABLE IF NOT EXISTS stocks (
  id INTEGER PRIMARY KEY,
  ticker VARCHAR(32) NOT NULL UNIQUE,
  name VARCHAR(255),
  exchange VARCHAR(64),
  currency VARCHAR(16),
  sector VARCHAR(128),
  industry VARCHAR(128),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS price_history (
  id INTEGER PRIMARY KEY,
  stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  open DOUBLE PRECISION,
  high DOUBLE PRECISION,
  low DOUBLE PRECISION,
  close DOUBLE PRECISION,
  adjusted_close DOUBLE PRECISION,
  volume BIGINT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
  UNIQUE (stock_id, date)
);

CREATE INDEX IF NOT EXISTS ix_price_history_date ON price_history(date);
CREATE INDEX IF NOT EXISTS ix_price_history_stock_date ON price_history(stock_id, date);
