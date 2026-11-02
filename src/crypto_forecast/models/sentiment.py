"""Crypto Sentiment Analysis Engine & RSS News Ingestion Pipeline.

Combines:
1. CryptoBertClassifier: RoBERTa-base sequence classification wrapper for
   'ElKulako/cryptobert' with lazy loading, hardware device placement (CUDA/MPS/CPU),
   FP16 execution on CUDA, dynamic batching, and OOM recovery.
2. HeuristicSentimentAnalyzer: Embedded lexicon-based analyzer for offline
   testing and environments where heavy model weights or PyTorch are absent.
3. Keyless RSS News Ingestion: Real-time public news extraction from CoinTelegraph
   and CoinDesk, with browser User-Agent headers, standard-library XML parsing,
   HTML sanitization, UTC timestamp normalization, and multi-source deduplication.
4. Polarity & Decay Dynamics: Softmax polarity scoring s = p_bullish - p_bearish,
   hourly exponential decay with alpha=0.95 (half-life ~13.51h), continuous time-series
   reindexing, and horizon-attenuated directional conviction drift adjustments.
"""

from __future__ import annotations

import difflib
import email.utils
import html
import logging
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
import requests

from crypto_forecast.config import (
    CRYPTOBERT_MAX_LENGTH,
    DEFAULT_BACKOFF_BASE,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_BACKOFF_JITTER,
    DEFAULT_BACKOFF_MAX,
    DEFAULT_MAX_NEWS_ARTICLES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RSS_FEEDS,
    DEFAULT_RSS_HEADERS,
    DEFAULT_SENTIMENT_GAMMA_0,
    DEFAULT_SENTIMENT_LAMBDA,
    HORIZON_TO_HOURS,
    SENTIMENT_DECAY_ALPHA,
    SENTIMENT_LABELS,
    SENTIMENT_MODEL_ID,
    TITLE_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Exceptions
# ==============================================================================


class SentimentError(Exception):
    """Base exception for all sentiment analysis and news ingestion failures."""


class SentimentModelError(SentimentError):
    """Raised when the sentiment model fails to load and fallback is disabled."""


class SentimentInferenceError(SentimentError):
    """Raised when sentiment inference encounters an unrecoverable error."""


class RSSFeedError(SentimentError):
    """Raised when RSS feeds cannot be retrieved or parsed."""


# ==============================================================================
# 1. Offline Heuristic Lexicon Fallback
# ==============================================================================


class HeuristicSentimentAnalyzer:
    """Lightweight rule-based sentiment classifier for offline or test environments.

    Uses a crypto-specific keyword lexicon and negation detection to derive
    bounded polarity in [-1.0, 1.0] and valid probability distributions summing to 1.0.
    """

    BULLISH_KEYWORDS: Final[set[str]] = {
        "surge", "surges", "surging", "surged",
        "rally", "rallies", "rallying", "rallied",
        "jump", "jumps", "jumping", "jumped",
        "pump", "pumps", "pumping", "pumped",
        "bull", "bullish", "bulls",
        "breakout", "ath", "all-time high", "all time high",
        "high", "record", "soar", "soaring", "soars",
        "gain", "gains", "gaining", "gained",
        "profit", "inflow", "inflows", "accumulate", "accumulation",
        "buy", "buying", "buys", "bought",
        "partnership", "partner", "partnerships", "upgrade", "approved", "approval",
        "bounce", "rebound", "green", "milestone", "adoption", "skyrocket",
    }

    BEARISH_KEYWORDS: Final[set[str]] = {
        "crash", "crashes", "crashing", "crashed",
        "plunge", "plunges", "plunging", "plunged",
        "drop", "drops", "dropping", "dropped",
        "dump", "dumps", "dumping", "dumped",
        "bear", "bearish", "bears",
        "breakdown", "fall", "falls", "falling", "fell",
        "hack", "hacks", "hacked", "hacking",
        "exploit", "exploited", "scam", "stolen", "theft",
        "halt", "halts", "halted", "halting",
        "suspend", "suspends", "suspended",
        "ban", "bans", "banned",
        "lawsuit", "sue", "sued", "suing",
        "sec", "investigation", "subpoena",
        "insolvent", "insolvency", "bankrupt", "bankruptcy",
        "liquidation", "liquidated", "liquidations",
        "outflow", "outflows", "sell", "selling", "sells", "sold",
        "fear", "fud", "red", "collapse", "crisis",
    }

    NEGATIONS: Final[set[str]] = {
        "not", "no", "never", "without", "hardly", "barely", "neither",
    }

    def predict_single(self, text: str | None) -> dict[str, float | str]:
        """Analyzes a single text string returning probabilities and polarity."""
        if not text or not str(text).strip():
            return {
                "bearish": 0.0,
                "neutral": 1.0,
                "bullish": 0.0,
                "polarity": 0.0,
                "label": "Neutral",
            }

        # Tokenize cleanly into lowercase alphanumeric tokens
        words = re.findall(r"\b[a-z0-9\-']+\b", str(text).lower())
        if not words:
            return {
                "bearish": 0.0,
                "neutral": 1.0,
                "bullish": 0.0,
                "polarity": 0.0,
                "label": "Neutral",
            }

        pos_count = 0
        neg_count = 0

        for i, word in enumerate(words):
            is_negated = (i > 0 and words[i - 1] in self.NEGATIONS)
            if word in self.BULLISH_KEYWORDS:
                if is_negated:
                    neg_count += 1
                else:
                    pos_count += 1
            elif word in self.BEARISH_KEYWORDS:
                if is_negated:
                    pos_count += 1
                else:
                    neg_count += 1

        total_sentiment_words = pos_count + neg_count
        if total_sentiment_words == 0:
            return {
                "bearish": 0.0,
                "neutral": 1.0,
                "bullish": 0.0,
                "polarity": 0.0,
                "label": "Neutral",
            }

        raw_polarity = (pos_count - neg_count) / float(total_sentiment_words)
        polarity = float(np.clip(raw_polarity, -1.0, 1.0))

        if polarity > 0.05:
            bullish = float(0.333 + 0.667 * polarity)
            bearish = float(0.333 * (1.0 - polarity))
            neutral = float(max(0.0, 1.0 - bullish - bearish))
            label = "Bullish"
        elif polarity < -0.05:
            bearish = float(0.333 + 0.667 * abs(polarity))
            bullish = float(0.333 * (1.0 - abs(polarity)))
            neutral = float(max(0.0, 1.0 - bullish - bearish))
            label = "Bearish"
        else:
            neutral = 1.0
            bearish = 0.0
            bullish = 0.0
            polarity = 0.0
            label = "Neutral"

        total_p = bearish + neutral + bullish
        if total_p <= 0.0:
            total_p = 1.0
            neutral = 1.0

        return {
            "bearish": round(bearish / total_p, 6),
            "neutral": round(neutral / total_p, 6),
            "bullish": round(bullish / total_p, 6),
            "polarity": round(polarity, 6),
            "label": label,
        }

    def predict(self, texts: Sequence[str]) -> list[dict[str, float | str]]:
        """Analyzes a sequence of texts."""
        return [self.predict_single(t) for t in texts]


# ==============================================================================
# 2. CryptoBERT Model Wrapper (RoBERTa Sequence Classifier)
# ==============================================================================


class CryptoBertClassifier:
    """Wrapper for Hugging Face ElKulako/cryptobert sequence classifier.

    Features:
    - Lazy loading: Weights not downloaded or initialized until first call.
    - Hardware device placement: CUDA, MPS, or CPU.
    - FP16 precision on CUDA for Colab T4 acceleration and memory efficiency.
    - Mini-batch inference to prevent latency/memory spikes.
    - Dynamic padding and max sequence length truncation (default 128).
    - CUDA OOM automatic mitigation with batch halving and CPU fallback.
    - Seamless heuristic fallback when Hugging Face Hub or torch is unavailable.
    """

    def __init__(
        self,
        model_id: str = SENTIMENT_MODEL_ID,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = CRYPTOBERT_MAX_LENGTH,
        use_fp16: bool = True,
        fallback_to_heuristic: bool = True,
    ) -> None:
        self.model_id: Final[str] = model_id
        self.batch_size: int = max(1, int(batch_size))
        self.max_length: int = max(16, int(max_length))
        self.use_fp16: bool = bool(use_fp16)
        self.fallback_to_heuristic: bool = bool(fallback_to_heuristic)

        self._explicit_device_str: str | None = device
        self._device: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_loaded: bool = False
        self._is_fallback_active: bool = False
        self._heuristic_fallback: HeuristicSentimentAnalyzer | None = None

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model or fallback has been initialized."""
        return self._is_loaded

    @property
    def is_fallback_active(self) -> bool:
        """Returns True if the classifier is running under heuristic fallback."""
        return self._is_fallback_active

    @property
    def device(self) -> Any:
        """Returns the active torch device, or 'cpu' if in fallback."""
        return self._device if self._device is not None else "cpu"

    def _init_fallback(self) -> None:
        """Initializes the offline heuristic fallback."""
        if self._heuristic_fallback is None:
            self._heuristic_fallback = HeuristicSentimentAnalyzer()
        self._is_fallback_active = True
        self._is_loaded = True

    def _resolve_device(self, torch_module: Any) -> Any:
        """Resolves target execution device based on hardware availability."""
        if self._explicit_device_str is not None:
            return torch_module.device(self._explicit_device_str)
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")

    def load(self) -> None:
        """Loads tokenizer and sequence classification model into memory.

        Postponed until first predict() call to prevent heavy imports or network
        delays on module import.
        """
        if self._is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._device = self._resolve_device(torch)
            logger.info("Initializing CryptoBERT model '%s' on %s...", self.model_id, self._device)

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            dtype = torch.float16 if (self.use_fp16 and self._device.type == "cuda") else torch.float32

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
            )
            self._model.to(self._device)
            self._model.eval()

            self._is_loaded = True
            self._is_fallback_active = False
            logger.info("CryptoBERT loaded successfully on %s (dtype=%s)", self._device, dtype)

        except Exception as exc:
            logger.warning(
                "Unable to load Hugging Face CryptoBERT model '%s': %s",
                self.model_id,
                exc,
            )
            if self.fallback_to_heuristic:
                logger.info("Activating offline heuristic sentiment fallback engine.")
                self._init_fallback()
            else:
                raise SentimentModelError(
                    f"Failed to load CryptoBERT model '{self.model_id}': {exc}"
                ) from exc

    def _predict_neural_batch(self, batch_texts: list[str]) -> list[dict[str, float | str]]:
        """Executes a single mini-batch forward pass through RoBERTa."""
        import torch

        cleaned_texts: list[str] = []
        is_blank: list[bool] = []
        for t in batch_texts:
            if not t or not str(t).strip():
                is_blank.append(True)
                cleaned_texts.append("neutral market")
            else:
                is_blank.append(False)
                cleaned_texts.append(str(t))

        inputs = self._tokenizer(
            cleaned_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        batch_results: list[dict[str, float | str]] = []
        for idx, row in enumerate(probs):
            if is_blank[idx]:
                batch_results.append({
                    "bearish": 0.0,
                    "neutral": 1.0,
                    "bullish": 0.0,
                    "polarity": 0.0,
                    "label": "Neutral",
                })
                continue

            p_bear = float(row[0])
            p_neut = float(row[1])
            p_bull = float(row[2])

            total = p_bear + p_neut + p_bull
            if total > 0.0:
                p_bear /= total
                p_neut /= total
                p_bull /= total

            polarity = float(np.clip(p_bull - p_bear, -1.0, 1.0))
            best_class = int(np.argmax([p_bear, p_neut, p_bull]))
            label = SENTIMENT_LABELS.get(best_class, "Neutral")

            batch_results.append({
                "bearish": round(p_bear, 6),
                "neutral": round(p_neut, 6),
                "bullish": round(p_bull, 6),
                "polarity": round(polarity, 6),
                "label": label,
            })

        return batch_results

    def _recover_oom(self, batch_texts: list[str], exc: Exception) -> list[dict[str, float | str]]:
        """Handles CUDA Out-Of-Memory exceptions by halving batch size or falling back to CPU."""
        import torch

        logger.warning("CUDA OOM encountered during sentiment batch. Clearing cache...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(batch_texts) > 1:
            mid = len(batch_texts) // 2
            logger.info("Retrying with split batch sizes: %d and %d", mid, len(batch_texts) - mid)
            res1 = self._predict_with_oom_protection(batch_texts[:mid])
            res2 = self._predict_with_oom_protection(batch_texts[mid:])
            return res1 + res2

        # Single item still OOMs: migrate model to CPU
        logger.warning("Single item OOM on GPU. Migrating CryptoBERT model to CPU...")
        self._device = torch.device("cpu")
        self._model.to(self._device)
        return self._predict_neural_batch(batch_texts)

    def _predict_with_oom_protection(self, batch_texts: list[str]) -> list[dict[str, float | str]]:
        """Wraps batch execution with OOM interception."""
        try:
            return self._predict_neural_batch(batch_texts)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return self._recover_oom(batch_texts, e)
            raise

    def predict(self, texts: str | Sequence[str]) -> list[dict[str, float | str]]:
        """Extracts sentiment probabilities and polarity for input texts.

        Parameters
        ----------
        texts : str or sequence of str
            Crypto news headlines or social texts.

        Returns
        -------
        list of dict
            Standardized dict per text:
            - 'bearish': float in [0.0, 1.0]
            - 'neutral': float in [0.0, 1.0]
            - 'bullish': float in [0.0, 1.0]
            - 'polarity': float in [-1.0, 1.0]
            - 'label': str ('Bearish', 'Neutral', 'Bullish')
        """
        if isinstance(texts, str):
            text_list = [texts]
        elif isinstance(texts, (list, tuple, np.ndarray)):
            text_list = [str(t) if t is not None else "" for t in texts]
        else:
            raise TypeError(f"Expected str or sequence of str, got {type(texts).__name__}")

        if not text_list:
            return []

        if not self._is_loaded:
            self.load()

        if self._is_fallback_active:
            assert self._heuristic_fallback is not None
            return self._heuristic_fallback.predict(text_list)

        results: list[dict[str, float | str]] = []
        for i in range(0, len(text_list), self.batch_size):
            batch = text_list[i : i + self.batch_size]
            batch_res = self._predict_with_oom_protection(batch)
            results.extend(batch_res)

        return results


# Module Singleton
_GLOBAL_SENTIMENT_CLASSIFIER: CryptoBertClassifier | None = None


def get_default_classifier() -> CryptoBertClassifier:
    """Returns the singleton CryptoBertClassifier instance."""
    global _GLOBAL_SENTIMENT_CLASSIFIER
    if _GLOBAL_SENTIMENT_CLASSIFIER is None:
        _GLOBAL_SENTIMENT_CLASSIFIER = CryptoBertClassifier()
    return _GLOBAL_SENTIMENT_CLASSIFIER


def analyze_sentiment(
    texts: str | Sequence[str],
    classifier: CryptoBertClassifier | None = None,
) -> list[dict[str, float | str]]:
    """Analyzes sentiment using CryptoBERT, returning probabilities and polarity.

    Conforms to the PROJECT.md M2 <-> M3 Interface Contract:
    Returns list of {'bearish': float, 'neutral': float, 'bullish': float, 'polarity': float, 'label': str}.
    """
    if classifier is None:
        classifier = get_default_classifier()
    return classifier.predict(texts)


# ==============================================================================
# 3. Keyless Public News RSS Ingestion & Cleaning
# ==============================================================================


def clean_html_text(raw_text: str | None) -> str:
    """Strips HTML tags, removes script/style blocks, unescapes entities, and collapses whitespace."""
    if not raw_text:
        return ""

    text = str(raw_text).replace("<![CDATA[", "").replace("]]>", "")

    # Strip script, style, and svg tags including internal content
    text = re.sub(
        r"<(script|style|svg)[^>]*>.*?</\1>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Unescape HTML entities (&amp;, &quot;, &#39;, &lt;, &gt;, etc.)
    text = html.unescape(text)

    # Normalize whitespace and strip leading/trailing spaces
    return re.sub(r"\s+", " ", text).strip()


def normalize_rss_date(date_str: str | None) -> pd.Timestamp:
    """Normalizes RFC 822/2822 or ISO-8601 publication dates into a UTC pd.Timestamp."""
    if not date_str or not str(date_str).strip():
        return pd.Timestamp.now(tz="UTC")

    clean_date = str(date_str).strip()

    # Strategy 1: pd.to_datetime handles RFC 2822, RFC 822, ISO 8601 natively
    try:
        ts = pd.to_datetime(clean_date, utc=True)
        if not pd.isna(ts):
            return ts
    except (ValueError, TypeError, pd.errors.ParserError):
        pass

    # Strategy 2: email.utils.parsedate_to_datetime standard library fallback
    try:
        dt_val = email.utils.parsedate_to_datetime(clean_date)
        if dt_val.tzinfo is None:
            return pd.Timestamp(dt_val, tz="UTC")
        return pd.Timestamp(dt_val).tz_convert("UTC")
    except (ValueError, TypeError, IndexError):
        pass

    logger.warning("Failed to parse publication date '%s', defaulting to now(UTC).", date_str)
    return pd.Timestamp.now(tz="UTC")


def clean_article_url(url: str | None) -> str:
    """Strips tracking query parameters (utm_*, ref, etc.) and trailing slashes."""
    if not url:
        return ""
    clean = str(url).strip()
    if "?" in clean:
        clean = clean.split("?")[0]
    return clean.rstrip("/")


def compute_title_similarity(title_a: str, title_b: str) -> float:
    """Calculates SequenceMatcher ratio between two normalized article titles."""
    clean_a = re.sub(r"[^\w\s]", "", str(title_a).lower()).strip()
    clean_b = re.sub(r"[^\w\s]", "", str(title_b).lower()).strip()
    return difflib.SequenceMatcher(None, clean_a, clean_b).ratio()


def _find_item_text(item: ET.Element, tag_name: str) -> str:
    """Extracts trimmed text from an XML element, handling namespaces and local tag matching."""
    el = item.find(tag_name)
    if el is not None and el.text:
        return el.text.strip()

    target = tag_name.lower()
    for child in item:
        local_tag = child.tag.split("}")[-1].lower() if "}" in child.tag else child.tag.lower()
        if local_tag == target and child.text:
            return child.text.strip()
    return ""


def parse_rss_xml(
    xml_content: str | bytes,
    source_name: str = "Unknown",
) -> list[dict[str, Any]]:
    """Parses RSS 2.0 or Atom XML payload and extracts structured, sanitized news items.

    Returns:
    [
        {
            'title': str,
            'published': pd.Timestamp (UTC),
            'source': str,
            'description': str,
            'link': str
        },
        ...
    ]
    """
    if not xml_content:
        return []

    if isinstance(xml_content, bytes):
        xml_str = xml_content.decode("utf-8", errors="replace")
    else:
        xml_str = str(xml_content)

    # Repair bare unescaped ampersands common in RSS feeds
    xml_str = re.sub(r"&(?!(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);)", "&amp;", xml_str)

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as exc:
        logger.warning("Failed to parse RSS XML from %s: %s", source_name, exc)
        return []

    items = root.findall(".//item")
    if not items:
        items = [
            c for c in root.iter()
            if (c.tag.split("}")[-1] if "}" in c.tag else c.tag).lower() in ("item", "entry")
        ]

    parsed_articles: list[dict[str, Any]] = []

    for item in items:
        raw_title = _find_item_text(item, "title")
        raw_desc = _find_item_text(item, "description")
        if not raw_desc:
            raw_desc = _find_item_text(item, "summary") or _find_item_text(item, "encoded")
        raw_date = _find_item_text(item, "pubDate") or _find_item_text(item, "updated")
        raw_link = _find_item_text(item, "link")

        clean_title = clean_html_text(raw_title)
        clean_desc = clean_html_text(raw_desc)

        # Fallback to first 100 characters of description if title is empty
        if not clean_title and clean_desc:
            clean_title = clean_desc[:100] + "..." if len(clean_desc) > 100 else clean_desc

        if not clean_title:
            continue

        pub_ts = normalize_rss_date(raw_date)
        canonical_link = clean_article_url(raw_link)

        parsed_articles.append({
            "title": clean_title,
            "published": pub_ts,
            "source": source_name,
            "description": clean_desc,
            "link": canonical_link,
        })

    return parsed_articles


def deduplicate_articles(
    articles: list[dict[str, Any]],
    similarity_threshold: float = TITLE_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    """Deduplicates articles using canonical URL matching and title similarity."""
    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for art in articles:
        norm_url = clean_article_url(art.get("link", "")).lower()

        # Rule 1: Exact URL match (stripped of query params)
        if norm_url and norm_url in seen_urls:
            continue

        # Rule 2: Title similarity check against already accepted articles
        art_title = art.get("title", "")
        is_duplicate_title = False
        for prev in deduped:
            prev_title = prev.get("title", "")
            if compute_title_similarity(art_title, prev_title) >= similarity_threshold:
                is_duplicate_title = True
                break

        if is_duplicate_title:
            continue

        if norm_url:
            seen_urls.add(norm_url)
        deduped.append(art)

    return deduped


def fetch_single_feed(
    source_name: str,
    url: str,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetches and parses a single RSS feed with retry backoff and desktop User-Agent."""
    req_session = session or requests.Session()

    for attempt in range(max_retries):
        try:
            resp = req_session.get(
                url,
                headers=DEFAULT_RSS_HEADERS,
                timeout=timeout,
            )

            if resp.status_code == 200:
                return parse_rss_xml(resp.content, source_name=source_name)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                sleep_sec = (
                    float(retry_after)
                    if retry_after
                    else min(
                        DEFAULT_BACKOFF_MAX,
                        DEFAULT_BACKOFF_BASE * (DEFAULT_BACKOFF_FACTOR**attempt),
                    )
                    + random.uniform(0.0, DEFAULT_BACKOFF_JITTER)
                )
                logger.warning(
                    "Rate limited on %s (HTTP 429). Retrying in %.2fs (attempt %d/%d)",
                    source_name,
                    sleep_sec,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_sec)
                continue

            if resp.status_code == 403:
                logger.warning("HTTP 403 Forbidden on %s. Bot protection blocked request.", source_name)
                return []

            sleep_sec = min(
                DEFAULT_BACKOFF_MAX,
                DEFAULT_BACKOFF_BASE * (DEFAULT_BACKOFF_FACTOR**attempt),
            )
            logger.warning("HTTP %d from %s. Retrying in %.2fs", resp.status_code, source_name, sleep_sec)
            time.sleep(sleep_sec)

        except (requests.Timeout, requests.ConnectionError) as exc:
            sleep_sec = min(
                DEFAULT_BACKOFF_MAX,
                DEFAULT_BACKOFF_BASE * (DEFAULT_BACKOFF_FACTOR**attempt),
            )
            logger.warning(
                "Network failure fetching %s (%s). Retrying in %.2fs",
                source_name,
                exc,
                sleep_sec,
            )
            time.sleep(sleep_sec)
        except (requests.RequestException, OSError) as exc:
            logger.warning("Unexpected error fetching %s: %s", source_name, exc)
            break

    logger.warning("Exhausted %d retries fetching RSS feed from %s.", max_retries, source_name)
    return []


def fetch_crypto_news(
    sources: dict[str, str] | list[str] | None = None,
    max_articles: int = DEFAULT_MAX_NEWS_ARTICLES,
    raise_on_error: bool = False,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Unified multi-source public crypto news ingestion and deduplication engine.

    Conforms to PROJECT.md interface contract:
    Returns list of {'title': str, 'published': pd.Timestamp, 'source': str,
                     'description': str, 'link': str}
    sorted in descending order by publication timestamp (newest first).
    """
    feeds_to_fetch: dict[str, str] = {}
    if sources is None:
        feeds_to_fetch = dict(DEFAULT_RSS_FEEDS)
    elif isinstance(sources, dict):
        feeds_to_fetch = dict(sources)
    elif isinstance(sources, (list, tuple)):
        for s in sources:
            if s in DEFAULT_RSS_FEEDS:
                feeds_to_fetch[s] = DEFAULT_RSS_FEEDS[s]
            else:
                logger.warning("Unknown RSS feed name '%s', skipping.", s)

    if not feeds_to_fetch:
        feeds_to_fetch = dict(DEFAULT_RSS_FEEDS)

    all_articles: list[dict[str, Any]] = []
    feed_errors: list[str] = []

    for source_name, url in feeds_to_fetch.items():
        try:
            articles = fetch_single_feed(
                source_name=source_name,
                url=url,
                session=session,
            )
            if articles:
                all_articles.extend(articles)
            else:
                feed_errors.append(f"{source_name}: 0 articles parsed")
        except (requests.RequestException, OSError, ET.ParseError) as exc:
            logger.warning("Error harvesting feed %s: %s", source_name, exc)
            feed_errors.append(f"{source_name}: {exc}")

    if not all_articles:
        if raise_on_error:
            raise RSSFeedError(f"Failed to fetch crypto news from all sources: {feed_errors}")
        logger.warning("No articles harvested from feeds: %s", feed_errors)
        return []

    deduped = deduplicate_articles(all_articles, similarity_threshold=TITLE_SIMILARITY_THRESHOLD)
    deduped.sort(key=lambda x: x["published"], reverse=True)
    return deduped[: max(1, max_articles)]


# Direct alias adhering to exact PROJECT.md Interface Contract name
fetch_latest_crypto_news = fetch_crypto_news


# ==============================================================================
# 4. Polarity, Decay Dynamics & Continuous Time-Series Alignment
# ==============================================================================


def compute_sentiment_polarity(bullish_prob: float, bearish_prob: float) -> float:
    """Calculates bounded polarity score in [-1.0, +1.0] from softmax class probabilities.

    Formula: s = p_bullish - p_bearish
    """
    pol = float(bullish_prob) - float(bearish_prob)
    return float(np.clip(pol, -1.0, 1.0))


def compute_hourly_decay(
    current_sentiment: float,
    elapsed_hours: float = 1.0,
    alpha: float = SENTIMENT_DECAY_ALPHA,
) -> float:
    """Applies exponential time decay: S_t = S_0 * (alpha ^ elapsed_hours).

    With alpha=0.95, half-life is approximately 13.5134 hours.
    """
    if elapsed_hours <= 0.0 or abs(current_sentiment) < 1e-7:
        return float(current_sentiment)
    decayed = float(current_sentiment) * (alpha ** float(elapsed_hours))
    if abs(decayed) < 1e-7:
        return 0.0
    return decayed


def get_sentiment_series(
    news_items: Sequence[dict[str, Any]],
    target_timestamps: pd.DatetimeIndex | Sequence[Any] | pd.Series,
    decay_factor: float = SENTIMENT_DECAY_ALPHA,
) -> pd.Series:
    """Aggregates discrete news items into an hourly continuous polarity series aligned

    with target_timestamps. When multiple articles arrive in hour t, computes mean polarity:
        S_bar_t = (1 / M_t) * sum(p_bullish,j - p_bearish,j)
    When M_t = 0 articles arrive in hour t, applies exponential decay:
        S_bar_t = alpha * S_bar_{t-1}
    """
    if len(target_timestamps) == 0:
        return pd.Series(
            [], dtype=np.float64, index=target_timestamps, name="sentiment_polarity"
        )

    target_idx = pd.DatetimeIndex(target_timestamps)

    # Normalize target timestamps to UTC for alignment
    if target_idx.tz is None:
        target_idx_utc = target_idx.tz_localize("UTC")
    else:
        target_idx_utc = target_idx.tz_convert("UTC")

    # Bucket news items into hourly bins (flooring to 1-hour candle start)
    hourly_news: dict[pd.Timestamp, list[float]] = {}
    for item in news_items:
        pub_raw = item.get("published", item.get("pub_date"))
        if pub_raw is None:
            continue
        try:
            pub_ts = pd.to_datetime(pub_raw)
            if pub_ts.tzinfo is None:
                pub_ts = pub_ts.tz_localize("UTC")
            else:
                pub_ts = pub_ts.tz_convert("UTC")
        except (ValueError, TypeError, pd.errors.ParserError):
            continue

        hour_key = pub_ts.floor("h")

        if "polarity" in item:
            pol = float(item["polarity"])
        elif "bullish" in item and "bearish" in item:
            pol = float(item["bullish"]) - float(item["bearish"])
        else:
            pol = 0.0
        pol = float(np.clip(pol, -1.0, 1.0))
        hourly_news.setdefault(hour_key, []).append(pol)

    # Precompute mean polarity per active news hour
    hourly_mean = {k: float(np.mean(v)) for k, v in hourly_news.items()}

    # Check monotonicity
    is_monotonic = target_idx_utc.is_monotonic_increasing
    if is_monotonic:
        sorted_idx = target_idx_utc
        sort_order = None
    else:
        sort_order = np.argsort(target_idx_utc)
        sorted_idx = target_idx_utc[sort_order]

    # Check for prior news items that occurred before sorted_idx[0]
    prior_news_hours = [k for k in sorted(hourly_mean.keys()) if k < sorted_idx[0]]
    current_sentiment = 0.0
    if prior_news_hours:
        last_prior_h = prior_news_hours[0]
        current_sentiment = hourly_mean[last_prior_h]
        for prev_k in prior_news_hours[1:]:
            delta_h = (prev_k - last_prior_h).total_seconds() / 3600.0
            current_sentiment = current_sentiment * (decay_factor**delta_h)
            current_sentiment = hourly_mean[prev_k]
            last_prior_h = prev_k
        delta_to_start = (sorted_idx[0] - last_prior_h).total_seconds() / 3600.0
        current_sentiment = current_sentiment * (decay_factor**delta_to_start)

    values = np.zeros(len(sorted_idx), dtype=np.float64)
    last_ts: pd.Timestamp | None = None

    for i, ts in enumerate(sorted_idx):
        if ts in hourly_mean:
            current_sentiment = hourly_mean[ts]
        else:
            if last_ts is not None:
                delta_hours = max(0.0, (ts - last_ts).total_seconds() / 3600.0)
                current_sentiment = current_sentiment * (decay_factor**delta_hours)
        if abs(current_sentiment) < 1e-7:
            current_sentiment = 0.0
        values[i] = current_sentiment
        last_ts = ts

    # Restore original sorting if target_timestamps was unsorted
    if not is_monotonic and sort_order is not None:
        orig_values = np.empty_like(values)
        orig_values[sort_order] = values
        values = orig_values

    return pd.Series(values, index=target_timestamps, name="sentiment_polarity")


# ==============================================================================
# 5. Directional Conviction Modifier & Horizon Attenuation
# ==============================================================================


def compute_horizon_gamma(
    horizon: str | int,
    gamma_0: float = DEFAULT_SENTIMENT_GAMMA_0,
    decay_rate: float = DEFAULT_SENTIMENT_LAMBDA,
    mode: str = "exponential",
) -> float:
    """Computes horizon attenuation factor gamma_h.

    Decays monotonically from ~0.02 at 1h down to ~0.002 at 30d.
    Modes:
      - 'exponential': gamma_h = gamma_0 * exp(-decay_rate * h)
      - 'sqrt': gamma_h = gamma_0 / sqrt(h)
    """
    if isinstance(horizon, str):
        h_hours = HORIZON_TO_HOURS.get(horizon, 1)
    else:
        h_hours = max(1, int(horizon))

    if mode == "sqrt":
        return float(gamma_0 / math.sqrt(h_hours))
    return float(gamma_0 * math.exp(-decay_rate * h_hours))


def apply_sentiment_drift(
    predictions: dict[str, Any],
    sentiment_polarity: float,
    gamma_0: float = DEFAULT_SENTIMENT_GAMMA_0,
    decay_rate: float = DEFAULT_SENTIMENT_LAMBDA,
    gamma_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Applies sentiment directional conviction adjustment to multi-horizon predictions:

    P_hat_adj = P_hat * (1 + gamma_h * S_bar)
    Updates predicted prices, confidence intervals (P10, P50, P90), predicted change,
    predicted return %, and directional tags ('BULLISH', 'BEARISH', 'NEUTRAL').
    """
    pol = float(np.clip(sentiment_polarity, -1.0, 1.0))
    adjusted: dict[str, Any] = {}

    for h, pred in predictions.items():
        if gamma_map is not None and h in gamma_map:
            gamma_h = float(gamma_map[h])
        else:
            step_h = getattr(pred, "step_hours", HORIZON_TO_HOURS.get(h, 1))
            gamma_h = compute_horizon_gamma(step_h, gamma_0, decay_rate)

        drift = gamma_h * pol
        multiplier = 1.0 + drift

        # Support both dataclass and dictionary objects
        if isinstance(pred, dict):
            curr_price = float(pred.get("current_price", 0.0))
            pred_price = float(pred.get("predicted_price", 0.0))
            p10 = float(pred.get("confidence_p10", pred_price * 0.95))
            p50 = float(pred.get("confidence_p50", pred_price))
            p90 = float(pred.get("confidence_p90", pred_price * 1.05))
        else:
            curr_price = float(getattr(pred, "current_price", 0.0))
            pred_price = float(getattr(pred, "predicted_price", 0.0))
            p10 = float(getattr(pred, "confidence_p10", pred_price * 0.95))
            p50 = float(getattr(pred, "confidence_p50", pred_price))
            p90 = float(getattr(pred, "confidence_p90", pred_price * 1.05))

        adj_price = pred_price * multiplier
        adj_p10 = p10 * multiplier
        adj_p50 = p50 * multiplier
        adj_p90 = p90 * multiplier

        adj_change = adj_price - curr_price
        adj_return = (adj_change / curr_price) * 100.0 if curr_price > 0.0 else 0.0

        if adj_change > 1e-6:
            adj_dir = "BULLISH"
        elif adj_change < -1e-6:
            adj_dir = "BEARISH"
        else:
            adj_dir = "NEUTRAL"

        if hasattr(pred, "__dataclass_fields__"):
            from dataclasses import replace

            adjusted[h] = replace(
                pred,
                predicted_price=adj_price,
                predicted_change=adj_change,
                predicted_return_pct=adj_return,
                direction=adj_dir,
                confidence_p10=adj_p10,
                confidence_p50=adj_p50,
                confidence_p90=adj_p90,
            )
        elif isinstance(pred, dict):
            item = dict(pred)
            item.update({
                "predicted_price": adj_price,
                "predicted_change": adj_change,
                "predicted_return_pct": adj_return,
                "direction": adj_dir,
                "confidence_p10": adj_p10,
                "confidence_p50": adj_p50,
                "confidence_p90": adj_p90,
            })
            adjusted[h] = item
        else:
            # Generic object with attributes
            item = dict(pred.__dict__)
            item.update({
                "predicted_price": adj_price,
                "predicted_change": adj_change,
                "predicted_return_pct": adj_return,
                "direction": adj_dir,
                "confidence_p10": adj_p10,
                "confidence_p50": adj_p50,
                "confidence_p90": adj_p90,
            })
            adjusted[h] = item

    return adjusted


def apply_trajectory_sentiment_drift(
    point_trajectory: np.ndarray,
    quantile_trajectory: np.ndarray,
    sentiment_polarity: float,
    gamma_0: float = DEFAULT_SENTIMENT_GAMMA_0,
    decay_rate: float = DEFAULT_SENTIMENT_LAMBDA,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies horizon-decayed drift across a continuous trajectory of steps (e.g. 720h)."""
    pol = float(np.clip(sentiment_polarity, -1.0, 1.0))
    n_steps = len(point_trajectory)
    steps = np.arange(1, n_steps + 1)
    gammas = gamma_0 * np.exp(-decay_rate * steps)
    multipliers = 1.0 + (gammas * pol)

    adj_points = point_trajectory * multipliers
    adj_quantiles = quantile_trajectory * multipliers[:, np.newaxis]
    return adj_points, adj_quantiles
