# 后端 FastAPI PoC

## 快速启动（本地）

1. 安装依赖

```
python -m venv .myveny
source myveny/Scripts/activate
pip install -r requirements.txt
```

2. 启动服务

```
uvicorn app.main:app --reload
```

3. 测试示例

```
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/admin/fetch_prices/AAPL?period_days=365"
curl -X POST "http://127.0.0.1:8000/admin/import_news/AAPL"
curl -X POST "http://127.0.0.1:8000/admin/sentiment/AAPL"
curl "http://127.0.0.1:8000/prediction/AAPL?window=30&alpha=0.01"
```

---

## 历史行情 API（第3天）

接口：
```
GET /stocks/{code}/history?start=YYYY-MM-DD&end=YYYY-MM-DD
```

示例：
```
curl "http://127.0.0.1:8000/stocks/AAPL/history?start=2023-01-01&end=2023-12-31"
```

参数说明：
- `start` / `end` 必须为 `YYYY-MM-DD` 格式
- 当 `start > end` 时返回 400
- 当无数据时返回 404

---

## Tiingo + CSV 回退说明（优先 Tiingo）

系统会**优先使用 Tiingo 抓取数据**，若 Tiingo 无法获取（例如 API Key 未配置、请求失败），则自动回退到本地 CSV 文件。

### 1) 配置 Tiingo API Key
在系统环境变量中设置：

```
TIINGO_API_KEY=6f9665ce00424b8eee6b597cb5e6234b650a21de
启动后端前命令行配置命令:
export TIINGO_API_KEY=6f9665ce00424b8eee6b597cb5e6234b650a21de
```

### 2) CSV 回退文件
回退文件路径：`data/prices_<ticker>.csv`  
例如：`data/prices_aapl.csv`

CSV 文件字段要求：

```
Date,Open,High,Low,Close,Adj Close,Volume
```

使用方式仍然保持不变：

```
curl -X POST "http://127.0.0.1:8000/admin/fetch_prices/AAPL?period_days=365"
```
