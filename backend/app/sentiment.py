import logging
import os
from functools import lru_cache
from typing import List, Tuple

from snownlp import SnowNLP
from sqlmodel import Session, select
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .models import ModelVersion, NewsItem, SentimentScore

logger = logging.getLogger(__name__)
vader = SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _get_hf_pipeline():
    """
    Lazy-load HuggingFace 情感模型；设置 HF_SENTIMENT_MODEL 来指定模型。
    若下载或加载失败，会抛异常，由上层兜底。
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    model_name = os.getenv("HF_SENTIMENT_MODEL", "hfl/chinese-roberta-wwm-ext-large")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return pipeline("text-classification", model=mdl, tokenizer=tok, top_k=None)


def _score_hf(texts: List[str]) -> List[Tuple[float, float]]:
    """
    使用 HF pipeline，返回 (score, confidence) 列表。
    假设 label1/pos 表示正向；如模型标签不同需适配。
    """
    clf = _get_hf_pipeline()
    outputs = clf(texts, truncation=True, max_length=256)
    results = []
    for out in outputs:
        pos = next((x for x in out if "1" in x["label"] or "pos" in x["label"].lower()), out[0])
        score = float(pos["score"])
        results.append((score, score))
    return results


def _score_snownlp(text: str) -> Tuple[float, float]:
    s = SnowNLP(text)
    v = float(s.sentiments)  # 0~1
    return v, 1.0


def _score_vader(text: str) -> Tuple[float, float]:
    vs = vader.polarity_scores(text)
    score = (vs["compound"] + 1) / 2
    return score, abs(vs["compound"])


# NEW: 默认模型版本自动创建
def get_or_create_default_model_version(engine) -> ModelVersion:  # CHANGED
    model_name = os.getenv("HF_SENTIMENT_MODEL", "hfl/chinese-roberta-wwm-ext-large")
    with Session(engine) as session:
        mv = session.exec(select(ModelVersion).where(ModelVersion.name == model_name)).one_or_none()
        if mv:
            return mv
        mv = ModelVersion(name=model_name, source="huggingface", task="sentiment")
        session.add(mv)
        session.commit()
        session.refresh(mv)
        return mv


def batch_compute_sentiment(news: List[NewsItem], model_version: ModelVersion, engine) -> int:
    """
    对新闻列表批量打分并写入 SentimentScore，兼容旧字段 sentiment_score。
    兜底顺序：HF -> SnowNLP -> VADER。
    """
    texts = [f"{n.title or ''}\n{n.content or ''}".strip() for n in news]
    scores: List[Tuple[float, float]] = []

    try:
        scores = _score_hf(texts)
    except Exception as e:  # noqa: BLE001
        logger.warning("HF model failed, fallback to SnowNLP/VADER: %s", e)
        for t in texts:
            try:
                scores.append(_score_snownlp(t))
            except Exception:
                scores.append(_score_vader(t))

    with Session(engine) as session:
        for n, (score, conf) in zip(news, scores):
            sc = SentimentScore(
                news_id=n.id,
                model_version_id=model_version.id,
                stock_code=n.stock_code,
                published_at=n.published_at or n.created_at.date(),
                score=score,
                confidence=conf,
            )
            n.sentiment_score = score  # 旧接口兼容
            session.add(sc)
            session.add(n)
        session.commit()
    return len(news)


def compute_sentiment_for_code(code: str, model_version: ModelVersion, engine) -> int:
    """兼容 admin 调用：对指定股票全部新闻打分。"""
    with Session(engine) as session:
        items = session.exec(select(NewsItem).where(NewsItem.stock_code == code.upper())).all()
    if not items:
        return 0
    return batch_compute_sentiment(items, model_version, engine)


# NEW: 兼容 main.py 调用入口
def compute_sentiment_for_news(code: str, engine) -> int:  # CHANGED
    model_version = get_or_create_default_model_version(engine)
    return compute_sentiment_for_code(code, model_version, engine)