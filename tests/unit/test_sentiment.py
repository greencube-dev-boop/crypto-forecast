"""Comprehensive Unit Test Suite for CryptoBERT Sentiment Engine & News Ingestion.

Covers:
- Group 1: Model Wrapper Initialization, Parameter Clamping, Lazy Loading & Fallbacks
- Group 2: Heuristic Lexicon Classifier (Crypto vocabulary, negations, probabilities, bounds)
- Group 3: Mocked RoBERTa Neural Pipeline (Tokenization, device placement, FP16, OOM recovery)
- Group 4: Text Cleaning, HTML Stripping, Date Normalization & Title Similarity
- Group 5: RSS XML Parsing, CDATA Blocks, Ampersand Repair & Deduplication
- Group 6: Network Ingestion, Retries, 429 Handling, 403 Bot-Protection & Fault Tolerance
- Group 7: Polarity Scoring & Analytical Exponential Time-Decay (Exact Half-life ~13.51h)
- Group 8: Hourly Continuous Time-Series Reindexing, Idle Decays & Aggregations
- Group 9: Multi-Horizon Conviction Drift Modifier, Gamma Monotonicity & Quantile Immunity
- Group 10: Trajectory Drift Modification & Interface Contract Compliance
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_forecast.config import (
    CRYPTOBERT_MAX_LENGTH,
    DEFAULT_HORIZONS,
    SENTIMENT_MODEL_ID,
)
from crypto_forecast.models.sentiment import (
    CryptoBertClassifier,
    HeuristicSentimentAnalyzer,
    RSSFeedError,
    SentimentModelError,
    analyze_sentiment,
    apply_sentiment_drift,
    apply_trajectory_sentiment_drift,
    clean_article_url,
    clean_html_text,
    compute_horizon_gamma,
    compute_hourly_decay,
    compute_sentiment_polarity,
    compute_title_similarity,
    deduplicate_articles,
    fetch_crypto_news,
    fetch_latest_crypto_news,
    fetch_single_feed,
    get_default_classifier,
    get_sentiment_series,
    normalize_rss_date,
    parse_rss_xml,
)


# Helper Dataclass to emulate HorizonPrediction from Milestone 3
@dataclass
class DummyHorizonPrediction:
    horizon: str
    step_hours: int
    predicted_price: float
    current_price: float
    predicted_change: float
    predicted_return_pct: float
    direction: str
    confidence_p10: float
    confidence_p50: float
    confidence_p90: float


# ==============================================================================
# GROUP 1: Classifier Initialization, Lazy Loading & Fallbacks
# ==============================================================================


class TestClassifierInitAndLazyLoading:
    """Verifies configuration, parameter clamping, and lazy initialization."""

    def test_default_init_parameters(self):
        """Verifies default parameters match config constants and model is not loaded."""
        clf = CryptoBertClassifier()
        assert clf.model_id == SENTIMENT_MODEL_ID
        assert clf.batch_size == 32
        assert clf.max_length == CRYPTOBERT_MAX_LENGTH
        assert clf.use_fp16 is True
        assert clf.fallback_to_heuristic is True
        assert clf.is_loaded is False
        assert clf.is_fallback_active is False

    def test_custom_parameters_and_clamping(self):
        """Verifies parameter bounds clamping (batch_size >= 1, max_length >= 16)."""
        clf = CryptoBertClassifier(
            model_id="custom/cryptobert",
            device="cpu",
            batch_size=-10,
            max_length=5,
            use_fp16=False,
            fallback_to_heuristic=False,
        )
        assert clf.model_id == "custom/cryptobert"
        assert clf.batch_size == 1
        assert clf.max_length == 16
        assert clf.use_fp16 is False
        assert clf.fallback_to_heuristic is False
        assert clf.is_loaded is False

    def test_lazy_fallback_activation_when_torch_missing(self):
        """Verifies that if torch/transformers cannot be loaded, fallback activates cleanly."""
        clf = CryptoBertClassifier(fallback_to_heuristic=True)
        # Calling predict triggers lazy load()
        res = clf.predict(["Bitcoin surges after ETF approval"])
        assert clf.is_loaded is True
        assert len(res) == 1
        assert res[0]["label"] == "Bullish"
        assert res[0]["polarity"] > 0.0

    def test_strict_mode_raises_when_fallback_disabled(self):
        """Verifies SentimentModelError is raised if fallback_to_heuristic=False and model fails."""
        clf = CryptoBertClassifier(
            model_id="nonexistent-model-id",
            fallback_to_heuristic=False,
        )
        with (
            patch.dict(sys.modules, {"torch": None}),
            pytest.raises(SentimentModelError),
        ):
            clf.load()


# ==============================================================================
# GROUP 2: Heuristic Sentiment Lexicon Analyzer
# ==============================================================================


class TestHeuristicSentimentAnalyzer:
    """Verifies offline rule-based lexicon classifier and invariant guarantees."""

    @pytest.fixture
    def analyzer(self) -> HeuristicSentimentAnalyzer:
        return HeuristicSentimentAnalyzer()

    def test_output_schema_and_types(self, analyzer: HeuristicSentimentAnalyzer):
        """Validates all required keys exist and have exact types."""
        results = analyzer.predict(["Bitcoin records all-time high"])
        assert len(results) == 1
        item = results[0]
        assert set(item.keys()) == {"bearish", "neutral", "bullish", "polarity", "label"}
        assert isinstance(item["bearish"], float)
        assert isinstance(item["neutral"], float)
        assert isinstance(item["bullish"], float)
        assert isinstance(item["polarity"], float)
        assert isinstance(item["label"], str)

    def test_probability_distribution_sum_to_one(self, analyzer: HeuristicSentimentAnalyzer):
        """Verifies probabilities are in [0, 1] and sum exactly to 1.0 within float tolerance."""
        samples = [
            "Massive rally across crypto markets as institutional inflows surge",
            "Catastrophic exchange hack and crash causes panic selloff",
            "Federal Reserve meeting concluded with neutral rate decision",
            "",
            "   ",
        ]
        results = analyzer.predict(samples)
        for res in results:
            assert 0.0 <= res["bearish"] <= 1.0
            assert 0.0 <= res["neutral"] <= 1.0
            assert 0.0 <= res["bullish"] <= 1.0
            total = res["bearish"] + res["neutral"] + res["bullish"]
            assert pytest.approx(total, abs=1e-5) == 1.0

    def test_polarity_bounds_and_symmetry(self, analyzer: HeuristicSentimentAnalyzer):
        """Verifies polarity is within [-1.0, 1.0] and directional concordance with labels."""
        bull_res = analyzer.predict_single("Surge rally pump breakout all-time high profit buy")
        assert bull_res["polarity"] > 0.5
        assert bull_res["label"] == "Bullish"
        assert bull_res["bullish"] > bull_res["bearish"]

        bear_res = analyzer.predict_single("Crash plunge drop dump hack exploit scam bankruptcy lawsuit")
        assert bear_res["polarity"] < -0.5
        assert bear_res["label"] == "Bearish"
        assert bear_res["bearish"] > bear_res["bullish"]

        neutral_res = analyzer.predict_single("Today is Monday and the weather is cloudy")
        assert neutral_res["polarity"] == 0.0
        assert neutral_res["label"] == "Neutral"
        assert neutral_res["neutral"] == 1.0

    def test_negation_inversion(self, analyzer: HeuristicSentimentAnalyzer):
        """Verifies negations ('not', 'no', 'never') invert bullish/bearish keywords."""
        pos = analyzer.predict_single("Market will surge")
        neg_pos = analyzer.predict_single("Market will not surge")
        assert pos["polarity"] > 0.0
        assert neg_pos["polarity"] < 0.0

        bear = analyzer.predict_single("Exchange did crash")
        neg_bear = analyzer.predict_single("Exchange did not crash")
        assert bear["polarity"] < 0.0
        assert neg_bear["polarity"] > 0.0

    def test_empty_and_whitespace_inputs(self, analyzer: HeuristicSentimentAnalyzer):
        """Verifies blank strings return neutral outputs safely."""
        for blank in ["", "   ", "\t\n", None]:
            res = analyzer.predict_single(blank)
            assert res["polarity"] == 0.0
            assert res["label"] == "Neutral"
            assert res["neutral"] == 1.0


# ==============================================================================
# GROUP 3: Mocked RoBERTa Neural Pipeline & OOM Recovery
# ==============================================================================


class TestMockedRoBERTaPipeline:
    """Verifies neural model execution, dynamic batching, and OOM recovery via mocking."""

    def test_mocked_roberta_forward_pass(self):
        """Simulates successful RoBERTa inference with synthetic logits."""
        clf = CryptoBertClassifier(fallback_to_heuristic=False)

        mock_torch = MagicMock()
        mock_torch.device.return_value = "cpu"
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        # Simulate softmax probabilities: Bearish=0.05, Neutral=0.10, Bullish=0.85
        mock_probs = np.array([[0.05, 0.10, 0.85]], dtype=np.float32)
        mock_torch.softmax.return_value.detach.return_value.cpu.return_value.numpy.return_value = mock_probs

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock()}

        mock_model = MagicMock()
        mock_model.return_value.logits = MagicMock()

        clf._tokenizer = mock_tokenizer
        clf._model = mock_model
        clf._device = "cpu"
        clf._is_loaded = True

        with patch.dict(sys.modules, {"torch": mock_torch}):
            res = clf._predict_neural_batch(["Bitcoin rally reaches $100k"])

        assert len(res) == 1
        assert res[0]["label"] == "Bullish"
        assert res[0]["bullish"] == pytest.approx(0.85, abs=1e-3)
        assert res[0]["bearish"] == pytest.approx(0.05, abs=1e-3)
        assert res[0]["polarity"] == pytest.approx(0.80, abs=1e-3)

    def test_cuda_oom_halving_recovery(self):
        """Verifies CUDA OOM automatically halves batch size and succeeds."""
        clf = CryptoBertClassifier(batch_size=4, fallback_to_heuristic=False)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        call_count = [0]

        def fake_predict_batch(batch_texts):
            call_count[0] += 1
            if len(batch_texts) > 1 and call_count[0] == 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.0 GiB")
            return [
                {"bearish": 0.1, "neutral": 0.1, "bullish": 0.8, "polarity": 0.7, "label": "Bullish"}
                for _ in batch_texts
            ]

        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            patch.object(clf, "_predict_neural_batch", side_effect=fake_predict_batch),
        ):
            res = clf._predict_with_oom_protection(["Headline 1", "Headline 2"])

        assert len(res) == 2
        assert mock_torch.cuda.empty_cache.called

    def test_single_item_oom_migrates_to_cpu(self):
        """Verifies single item OOM migrates device to CPU."""
        clf = CryptoBertClassifier(fallback_to_heuristic=False)
        mock_torch = MagicMock()
        mock_torch.device.side_effect = lambda x: x
        mock_torch.cuda.is_available.return_value = True

        mock_model = MagicMock()
        clf._model = mock_model
        clf._device = "cuda"

        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            patch.object(
                clf,
                "_predict_neural_batch",
                return_value=[{"bearish": 0.0, "neutral": 1.0, "bullish": 0.0, "polarity": 0.0, "label": "Neutral"}],
            ),
        ):
            res = clf._recover_oom(["Single massive article"], RuntimeError("out of memory"))

        assert clf._device == "cpu"
        mock_model.to.assert_called_with("cpu")
        assert len(res) == 1


# ==============================================================================
# GROUP 4: Text Cleaning, HTML Sanitization & Date Normalization
# ==============================================================================


class TestTextAndDateHelpers:
    """Verifies HTML sanitization, URL cleaning, date normalization, and sequence matching."""

    def test_clean_html_text_stripping(self):
        """Verifies stripping of tags, scripts, styles, CDATA, and entity decoding."""
        raw = '<p style="color:red;">Breaking: Bitcoin &amp; Ethereum <script>alert(1)</script> surges &gt; $50k!</p>'
        cleaned = clean_html_text(raw)
        assert cleaned == "Breaking: Bitcoin & Ethereum surges > $50k!"
        assert "<script>" not in cleaned
        assert "style=" not in cleaned

    def test_clean_article_url_tracking_removal(self):
        """Verifies tracking parameters (?utm_source=...) and trailing slashes are removed."""
        url = "https://cointelegraph.com/news/btc-breakout?utm_source=rss_feed&utm_medium=rss/"
        clean = clean_article_url(url)
        assert clean == "https://cointelegraph.com/news/btc-breakout"

    def test_normalize_rss_date_formats(self):
        """Verifies RFC 2822, RFC 822 GMT, and ISO-8601 normalization to UTC pd.Timestamp."""
        # RFC 2822
        ts1 = normalize_rss_date("Mon, 02 Nov 2026 01:51:48 +0000")
        assert isinstance(ts1, pd.Timestamp)
        assert ts1.year == 2026 and ts1.month == 11 and ts1.day == 2
        assert str(ts1.tz) in ("UTC", "datetime.timezone.utc")

        # RFC 822 GMT
        ts2 = normalize_rss_date("Sun, 01 Nov 2026 22:15:00 GMT")
        assert ts2.hour == 22 and ts2.minute == 15

        # ISO 8601
        ts3 = normalize_rss_date("2026-11-02T03:30:00Z")
        assert ts3.hour == 3 and ts3.minute == 30

        # Empty / invalid string returns current UTC time without crashing
        ts_invalid = normalize_rss_date("invalid-date-string")
        assert isinstance(ts_invalid, pd.Timestamp)
        assert str(ts_invalid.tz) in ("UTC", "datetime.timezone.utc")

    def test_compute_title_similarity_matrix(self):
        """Verifies SequenceMatcher distinguishes similar from distinct headlines."""
        t1 = "Bitcoin breaks $100k resistance as ETF inflows surge"
        t2 = "Bitcoin Breaks $100k Resistance As ETF Inflows Surge!"
        t3 = "Solana DEX volume hits new record following protocol upgrade"

        assert compute_title_similarity(t1, t2) >= 0.95
        assert compute_title_similarity(t1, t3) < 0.50


# ==============================================================================
# GROUP 5: RSS XML Parsing & Multi-Source Deduplication
# ==============================================================================


MOCK_COINTELEGRAPH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Cointelegraph News</title>
    <item>
      <title>Bitcoin breaks $100k resistance as ETF inflows accelerate</title>
      <pubDate>Mon, 02 Nov 2026 01:51:48 +0000</pubDate>
      <link>https://cointelegraph.com/news/btc-100k?utm_source=rss</link>
      <description>&lt;p style="color:red;"&gt;Inflows exceeded $1.2B today.&lt;/p&gt;</description>
      <dc:creator>Felix Ng</dc:creator>
    </item>
    <item>
      <title>Solana DEX volume hits record post upgrade</title>
      <pubDate>Mon, 02 Nov 2026 00:30:00 +0000</pubDate>
      <link>https://cointelegraph.com/news/solana-dex</link>
      <description>Daily volume hit $5B.</description>
    </item>
  </channel>
</rss>"""

MOCK_COINDESK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>CoinDesk News</title>
    <item>
      <title><![CDATA[Ethereum gas fees collapse to sub-cent levels post Dencun]]></title>
      <pubDate>Sun, 01 Nov 2026 22:15:00 +0000</pubDate>
      <link>https://www.coindesk.com/tech/2026/11/01/ethereum-fees</link>
      <description><![CDATA[Blob transactions reduced fees by 95%.]]></description>
    </item>
  </channel>
</rss>"""

MOCK_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Crypto Atom Feed</title>
  <entry>
    <title>XRP cross-border settlement volume surges</title>
    <updated>2026-11-01T20:00:00Z</updated>
    <link href="https://example.com/xrp-volume" />
    <summary>Payment volume expanded significantly.</summary>
  </entry>
</feed>"""


class TestRSSParsingAndDeduplication:
    """Verifies XML parsing across RSS 2.0, CDATA, Atom, and deduplication logic."""

    def test_parse_cointelegraph_xml(self):
        """Verifies parsing of CoinTelegraph items with HTML sanitization."""
        items = parse_rss_xml(MOCK_COINTELEGRAPH_XML, source_name="CoinTelegraph")
        assert len(items) == 2
        assert items[0]["title"] == "Bitcoin breaks $100k resistance as ETF inflows accelerate"
        assert items[0]["source"] == "CoinTelegraph"
        assert items[0]["description"] == "Inflows exceeded $1.2B today."
        assert items[0]["link"] == "https://cointelegraph.com/news/btc-100k"
        assert isinstance(items[0]["published"], pd.Timestamp)

    def test_parse_coindesk_cdata_xml(self):
        """Verifies unpacking of CDATA elements in CoinDesk feed."""
        items = parse_rss_xml(MOCK_COINDESK_XML, source_name="CoinDesk")
        assert len(items) == 1
        assert "Ethereum gas fees collapse" in items[0]["title"]
        assert "Blob transactions" in items[0]["description"]

    def test_parse_atom_entry_xml(self):
        """Verifies parsing of Atom feed entry structures."""
        items = parse_rss_xml(MOCK_ATOM_XML, source_name="AtomFeed")
        assert len(items) == 1
        assert "XRP cross-border" in items[0]["title"]

    def test_malformed_xml_ampersand_repair(self):
        """Verifies unescaped ampersands in titles do not cause fatal XML parse error."""
        malformed = """<rss version="2.0"><channel><item>
        <title>S&P 500 & Crypto Rally Together</title>
        <pubDate>Mon, 02 Nov 2026 00:00:00 +0000</pubDate>
        </item></channel></rss>"""
        items = parse_rss_xml(malformed, source_name="TestFeed")
        assert len(items) == 1
        assert "S&P 500 & Crypto Rally Together" in items[0]["title"]

    def test_deduplicate_articles_by_url_and_similarity(self):
        """Verifies duplicate articles across feeds are removed cleanly."""
        articles = [
            {
                "title": "Bitcoin surges past $90k on ETF demand",
                "link": "https://source1.com/btc?utm=1",
                "published": pd.Timestamp("2026-11-02 01:00:00", tz="UTC"),
            },
            {
                "title": "Bitcoin Surges Past $90k On ETF Demand!",  # Near-duplicate title
                "link": "https://source2.com/syndicated-btc",
                "published": pd.Timestamp("2026-11-02 00:50:00", tz="UTC"),
            },
            {
                "title": "Different Story: Solana hits record",
                "link": "https://source1.com/btc?utm=1",  # Same canonical URL
                "published": pd.Timestamp("2026-11-02 00:40:00", tz="UTC"),
            },
            {
                "title": "Solana hits new record DEX volume",
                "link": "https://source3.com/solana-dex",
                "published": pd.Timestamp("2026-11-02 00:30:00", tz="UTC"),
            },
        ]
        deduped = deduplicate_articles(articles, similarity_threshold=0.85)
        # First article kept, second dropped due to title similarity, third dropped due to URL match
        assert len(deduped) == 2
        assert deduped[0]["title"] == "Bitcoin surges past $90k on ETF demand"
        assert deduped[1]["title"] == "Solana hits new record DEX volume"


# ==============================================================================
# GROUP 6: Network Ingestion, Retries & Error Recovery
# ==============================================================================


class TestRSSNetworkIngestion:
    """Verifies network fetching, retry backoff, HTTP 429/403 handling, and fail-soft behavior."""

    @patch("requests.Session.get")
    def test_fetch_single_feed_success(self, mock_get):
        """Verifies successful HTTP 200 parsing."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = MOCK_COINTELEGRAPH_XML.encode("utf-8")
        mock_get.return_value = mock_resp

        items = fetch_single_feed("CoinTelegraph", "https://cointelegraph.com/rss")
        assert len(items) == 2
        assert mock_get.called
        headers = mock_get.call_args[1]["headers"]
        assert "Mozilla/5.0" in headers["User-Agent"]

    @patch("requests.Session.get")
    def test_fetch_single_feed_403_returns_empty(self, mock_get):
        """Verifies HTTP 403 Forbidden returns [] gracefully."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp

        items = fetch_single_feed("CoinTelegraph", "https://cointelegraph.com/rss")
        assert items == []

    @patch("requests.Session.get")
    def test_fetch_single_feed_429_retries_and_succeeds(self, mock_get):
        """Verifies HTTP 429 triggers retry backoff and succeeds on subsequent attempt."""
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "0.01"}

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.content = MOCK_COINDESK_XML.encode("utf-8")

        mock_get.side_effect = [resp_429, resp_200]

        items = fetch_single_feed("CoinDesk", "https://coindesk.com/rss", max_retries=3)
        assert len(items) == 1
        assert mock_get.call_count == 2

    @patch("crypto_forecast.models.sentiment.fetch_single_feed")
    def test_fetch_crypto_news_aggregation_and_sorting(self, mock_single):
        """Verifies multi-source aggregation, chronological sorting (newest first), and limit."""
        art1 = {
            "title": "Older article",
            "published": pd.Timestamp("2026-11-02 01:00:00", tz="UTC"),
            "source": "CoinTelegraph",
            "link": "https://ct.com/1",
        }
        art2 = {
            "title": "Newer article",
            "published": pd.Timestamp("2026-11-02 03:00:00", tz="UTC"),
            "source": "CoinDesk",
            "link": "https://cd.com/2",
        }
        mock_single.side_effect = [[art1], [art2]]

        news = fetch_crypto_news(max_articles=10)
        assert len(news) == 2
        # Chronological descending: newer article first
        assert news[0]["title"] == "Newer article"
        assert news[1]["title"] == "Older article"

    @patch("crypto_forecast.models.sentiment.fetch_single_feed")
    def test_fetch_crypto_news_raise_on_error(self, mock_single):
        """Verifies raise_on_error=True raises RSSFeedError if all feeds fail."""
        mock_single.return_value = []
        with pytest.raises(RSSFeedError):
            fetch_crypto_news(raise_on_error=True)

        # raise_on_error=False returns []
        empty = fetch_crypto_news(raise_on_error=False)
        assert empty == []


# ==============================================================================
# GROUP 7: Polarity Scoring & Analytical Exponential Time-Decay
# ==============================================================================


class TestPolarityAndDecayDynamics:
    """Verifies mathematical correctness of polarity calculation and exponential decay."""

    def test_compute_sentiment_polarity_known_bounds(self):
        """Verifies polarity bounds [-1.0, 1.0] and exact test vectors."""
        assert compute_sentiment_polarity(1.0, 0.0) == 1.0
        assert compute_sentiment_polarity(0.0, 1.0) == -1.0
        assert compute_sentiment_polarity(0.5, 0.5) == 0.0
        assert compute_sentiment_polarity(0.75, 0.15) == pytest.approx(0.60, abs=1e-5)
        # Clamping test
        assert compute_sentiment_polarity(1.5, 0.0) == 1.0
        assert compute_sentiment_polarity(0.0, 1.5) == -1.0

    def test_compute_hourly_decay_single_step(self):
        """Verifies single hour decay equals alpha * S_0."""
        assert compute_hourly_decay(1.0, elapsed_hours=1.0, alpha=0.95) == pytest.approx(0.95, abs=1e-5)
        assert compute_hourly_decay(-0.80, elapsed_hours=1.0, alpha=0.95) == pytest.approx(-0.76, abs=1e-5)

    def test_compute_hourly_decay_exact_half_life(self):
        """Verifies analytical half-life: T_half = -ln(2) / ln(0.95) ≈ 13.5134 hours."""
        alpha = 0.95
        half_life = -math.log(2) / math.log(alpha)
        assert half_life == pytest.approx(13.5134, abs=1e-3)
        decayed = compute_hourly_decay(1.0, elapsed_hours=half_life, alpha=alpha)
        assert decayed == pytest.approx(0.50, abs=1e-5)

    def test_compute_hourly_decay_long_horizon_convergence(self):
        """Verifies convergence toward 0.0 and subnormal clamping."""
        decayed_24h = compute_hourly_decay(1.0, elapsed_hours=24.0, alpha=0.95)
        assert decayed_24h == pytest.approx(0.95**24, abs=1e-5)
        assert decayed_24h == pytest.approx(0.291989, abs=1e-4)

        decayed_100h = compute_hourly_decay(1.0, elapsed_hours=100.0, alpha=0.95)
        assert decayed_100h < 0.006

        decayed_350h = compute_hourly_decay(1.0, elapsed_hours=350.0, alpha=0.95)
        assert decayed_350h == 0.0


# ==============================================================================
# GROUP 8: Hourly Continuous Time-Series Reindexing
# ==============================================================================


class TestSentimentSeriesReindexing:
    """Verifies get_sentiment_series alignment, idle decays, and zero NaN invariant."""

    def test_get_sentiment_series_happy_path(self):
        """Verifies multi-hour reindexing with news items and exponential idle decay."""
        dates = pd.date_range("2026-11-01 00:00:00", periods=5, freq="1h", tz="UTC")
        news = [
            {
                "title": "Positive News",
                "published": pd.Timestamp("2026-11-01 00:25:00", tz="UTC"),
                "polarity": 0.80,
            }
        ]
        series = get_sentiment_series(news, dates, decay_factor=0.95)

        assert isinstance(series, pd.Series)
        assert len(series) == 5
        assert series.iloc[0] == pytest.approx(0.80, abs=1e-4)
        assert series.iloc[1] == pytest.approx(0.80 * 0.95, abs=1e-4)
        assert series.iloc[2] == pytest.approx(0.80 * (0.95**2), abs=1e-4)
        assert series.iloc[3] == pytest.approx(0.80 * (0.95**3), abs=1e-4)
        assert series.iloc[4] == pytest.approx(0.80 * (0.95**4), abs=1e-4)
        assert series.isna().sum() == 0

    def test_get_sentiment_series_multi_article_aggregation(self):
        """Verifies multiple articles in the same hour are aggregated via arithmetic mean."""
        dates = pd.date_range("2026-11-01 00:00:00", periods=2, freq="1h", tz="UTC")
        news = [
            {"published": pd.Timestamp("2026-11-01 00:10:00", tz="UTC"), "polarity": 0.90},
            {"published": pd.Timestamp("2026-11-01 00:20:00", tz="UTC"), "polarity": 0.70},
            {"published": pd.Timestamp("2026-11-01 00:40:00", tz="UTC"), "polarity": 0.50},
        ]
        series = get_sentiment_series(news, dates, decay_factor=0.95)
        # Mean of [0.90, 0.70, 0.50] == 0.70
        assert series.iloc[0] == pytest.approx(0.70, abs=1e-4)
        assert series.iloc[1] == pytest.approx(0.70 * 0.95, abs=1e-4)

    def test_get_sentiment_series_empty_news(self):
        """Verifies empty news list returns series of all 0.0 with zero errors."""
        dates = pd.date_range("2026-11-01 00:00:00", periods=24, freq="1h", tz="UTC")
        series = get_sentiment_series([], dates)
        assert len(series) == 24
        assert (series == 0.0).all()
        assert series.isna().sum() == 0

    def test_get_sentiment_series_prior_news_decay_carryover(self):
        """Verifies news occurring before target_timestamps[0] is decayed forward into t0."""
        dates = pd.date_range("2026-11-01 04:00:00", periods=3, freq="1h", tz="UTC")
        news = [
            {
                "published": pd.Timestamp("2026-11-01 02:00:00", tz="UTC"),  # 2 hours before target start
                "polarity": 1.0,
            }
        ]
        series = get_sentiment_series(news, dates, decay_factor=0.95)
        # At 04:00, decayed by 2 hours: 1.0 * 0.95^2 = 0.9025
        assert series.iloc[0] == pytest.approx(1.0 * (0.95**2), abs=1e-4)

    def test_get_sentiment_series_timezone_naive_preservation(self):
        """Verifies timezone-naive target_timestamps are preserved in returned series."""
        dates = pd.date_range("2026-11-01 00:00:00", periods=5, freq="1h")  # naive
        news = [{"published": "2026-11-01 00:00:00", "polarity": 0.5}]
        series = get_sentiment_series(news, dates)
        assert series.index.tz is None
        assert len(series) == 5
        assert series.isna().sum() == 0


# ==============================================================================
# GROUP 9: Multi-Horizon Conviction Drift & Horizon Attenuation
# ==============================================================================


class TestHorizonGammaAndDrift:
    """Verifies gamma horizon attenuation, directional drift, and quantile preservation."""

    def test_compute_horizon_gamma_monotonic_decay(self):
        """Verifies gamma_h decays strictly monotonically across all 11 horizons."""
        gammas = [compute_horizon_gamma(h) for h in DEFAULT_HORIZONS]
        for i in range(len(gammas) - 1):
            assert gammas[i] > gammas[i + 1], (
                f"Gamma did not monotonically decay: {DEFAULT_HORIZONS[i]} ({gammas[i]}) <= {DEFAULT_HORIZONS[i+1]} ({gammas[i+1]})"
            )
        assert gammas[0] == pytest.approx(0.02, abs=2e-3)
        assert gammas[-1] < gammas[0]
        assert compute_horizon_gamma("30d") == pytest.approx(0.002, abs=1e-3)

    def test_apply_sentiment_drift_bullish_and_bearish(self):
        """Verifies directional shift on predictions dictionary."""
        preds = {
            "1h": DummyHorizonPrediction("1h", 1, 100.0, 100.0, 0.0, 0.0, "NEUTRAL", 95.0, 100.0, 105.0),
            "24h": DummyHorizonPrediction("24h", 24, 102.0, 100.0, 2.0, 2.0, "BULLISH", 96.0, 102.0, 108.0),
        }

        # Bullish Shock (+0.80)
        bull_adj = apply_sentiment_drift(preds, sentiment_polarity=0.80)
        assert bull_adj["1h"].predicted_price > 100.0
        assert bull_adj["1h"].direction == "BULLISH"
        assert bull_adj["24h"].predicted_price > 102.0

        # Bearish Shock (-0.80)
        bear_adj = apply_sentiment_drift(preds, sentiment_polarity=-0.80)
        assert bear_adj["1h"].predicted_price < 100.0
        assert bear_adj["1h"].direction == "BEARISH"
        assert bear_adj["24h"].predicted_price < 102.0

    def test_apply_sentiment_drift_zero_polarity_invariance(self):
        """Verifies zero polarity leaves predictions completely unchanged."""
        preds = {
            "1h": DummyHorizonPrediction("1h", 1, 100.0, 100.0, 0.0, 0.0, "NEUTRAL", 95.0, 100.0, 105.0),
        }
        adj = apply_sentiment_drift(preds, sentiment_polarity=0.0)
        assert adj["1h"].predicted_price == 100.0
        assert adj["1h"].direction == "NEUTRAL"
        assert adj["1h"].confidence_p10 == 95.0
        assert adj["1h"].confidence_p90 == 105.0

    def test_sentiment_drift_quantile_inversion_immunity(self):
        """Verifies P10 <= P50 <= P90 invariant is strictly preserved under extreme sentiment shocks."""
        preds = {
            "1h": {
                "step_hours": 1,
                "current_price": 50000.0,
                "predicted_price": 50000.0,
                "confidence_p10": 48000.0,
                "confidence_p50": 50000.0,
                "confidence_p90": 52000.0,
            }
        }
        for extreme_polarity in [-1.0, +1.0]:
            adj = apply_sentiment_drift(preds, sentiment_polarity=extreme_polarity)
            p10 = adj["1h"]["confidence_p10"]
            p50 = adj["1h"]["confidence_p50"]
            p90 = adj["1h"]["confidence_p90"]
            assert p10 < p50 < p90, f"Quantile ordering violated under polarity {extreme_polarity}"
            assert p10 > 0.0

    def test_apply_trajectory_sentiment_drift(self):
        """Verifies 720-step continuous trajectory drift adjustments."""
        points = np.full(720, 100.0)
        quantiles = np.tile([95.0, 100.0, 105.0], (720, 1))

        adj_points, adj_quantiles = apply_trajectory_sentiment_drift(
            points, quantiles, sentiment_polarity=0.80
        )
        assert adj_points.shape == (720,)
        assert adj_quantiles.shape == (720, 3)
        # Drift should be highest at step 1 and decay toward step 720
        drift_step1 = adj_points[0] - 100.0
        drift_step720 = adj_points[-1] - 100.0
        assert drift_step1 > drift_step720 > 0.0


# ==============================================================================
# GROUP 10: Interface Contract Adherence & Singleton Validation
# ==============================================================================


class TestInterfaceContracts:
    """Verifies top-level interface functions match PROJECT.md specifications."""

    def test_analyze_sentiment_contract(self):
        """Verifies analyze_sentiment() returns list of dicts with required keys."""
        res = analyze_sentiment(["Bitcoin breaks records", "Market crashes"])
        assert isinstance(res, list)
        assert len(res) == 2
        for item in res:
            assert "bearish" in item
            assert "neutral" in item
            assert "bullish" in item
            assert "polarity" in item
            assert "label" in item

    def test_get_default_classifier_singleton(self):
        """Verifies get_default_classifier returns the same singleton instance."""
        clf1 = get_default_classifier()
        clf2 = get_default_classifier()
        assert clf1 is clf2

    def test_fetch_latest_crypto_news_alias(self):
        """Verifies fetch_latest_crypto_news is an alias to fetch_crypto_news."""
        assert fetch_latest_crypto_news is fetch_crypto_news
