# 行情数据采集流程与异常记录（建议文档）

目标
- 从免费数据源（优先 yfinance） 拉取日线数据并写入数据库，至少导入 ≥6 个月的日线数据（示例：AAPL）。

环境与依赖
- Python 3.8+
- requirements:
  - sqlalchemy
  - pandas
  - yfinance
  - sqlite3 (默认) 或 其它数据库驱动 (psycopg2-binary for Postgres)

建议的 requirements 片段（追加到 backend/requirements.txt）:
```
SQLAlchemy>=1.4
pandas>=1.3
yfinance>=0.2.0
```

步骤（命令示例）
1. 初始化数据库（默认 SQLite 文件 data/stocks.db）:
   - python -m backend.app.db_init
2. 拉取并写入示例股票（例如 AAPL）:
   - python -m backend.app.fetch_prices --ticker AAPL --months 6
   - 或指定 DB URL:
     - DATABASE_URL=postgresql://user:pass@host:5432/dbname python -m backend.app.fetch_prices --ticker AAPL --months 6

数据源优先级与回退策略
- 首选：yfinance（免费，覆盖美股、部分港股/国际代码，例如 600519.SS 可能不在 yfinance 完整支持范围）
- 当 yfinance 无法获取（空返回或抛错），回退：
  1. 使用仓库内的 CSV 样本（data/news_aapl.csv 是新闻样本，若有价格 CSV，应放置在 data/prices_*.csv）
  2. 手动准备 CSV，列需包含 Date, Open, High, Low, Close, Adj Close, Volume（字段名或脚本中映射）
- 长期方案：接入稳定的免费/付费 API（AlphaVantage、IEX Cloud、Tiingo、Polygon 等），但需处理 API Key 与限流。

异常与注意事项
- 限流（rate limit）：yfinance 基于雅虎数据抓取，偶发性限制或数据缺失，脚本中会记录警告并跳过为空的数据集。
- 时区 & 交易日：使用 date（无时区）作为交易日键。若需要分钟级别或真实交易时区，请扩展为 timestamp。
- 复权：脚本保留 adjusted_close（Adj Close）用于复权计算。若用于回测，强烈建议使用 adjusted_close。
- 幂等写入：price_history 使用 (stock_id, date) 唯一约束，脚本尝试更新已有记录，避免重复插入。
- 数据质量检查：建议导入后运行简单校验：连续交易日数、异常零值、volume 为零/极端值等。

日志记录
- 脚本使用标准 logging 输出（INFO/WARNING/ERROR）。生产环境建议把日志导向文件或集中式日志系统，并对失败的 ticker 做重试队列/告警。

扩展建议
- 添加 CLI 支持批量导入（多个 ticker 列表），并行控制（注意 API 限流）。
- 增加单元测试：模拟 yfinance 返回的 DataFrame，检测 upsert 行为。
- 添加 Alembic 支持以便后续 schema 演进与迁移。
