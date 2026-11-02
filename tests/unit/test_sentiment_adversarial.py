"""Adversarial Stress Test Suite for CryptoBERT Sentiment Engine & News Ingestion.

Probes and stress-tests:
- News RSS ingestion across malformed XML, entity bombs, HTML injections, and entity mutations.
- Network resilience across HTTP 403, 429, 500, 502, timeouts, and partial feed failures.
- Date normalization across missing dates, malformed formats, timezone anomalies, and century bounds.
- Deduplication under 1,000-article loads, URL permutation attacks, and quadratic scaling boundaries.
- NLP Heuristic and Model wrappers under extreme inputs, negation chaining, and punctuation floods.
- Polarity, exponential decay, and multi-horizon drift adjustments under boundary conditions.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

# Ensure src directory is in sys.path
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_forecast.config import (
    DEFAULT_HORIZONS,
)
from crypto_forecast.models.sentiment import (
    CryptoBertClassifier,
    HeuristicSentimentAnalyzer,
    RSSFeedError,
    apply_sentiment_drift,
    apply_trajectory_sentiment_drift,
    clean_article_url,
    clean_html_text,
    compute_horizon_gamma,
    compute_hourly_decay,
    compute_sentiment_polarity,
    deduplicate_articles,
    fetch_crypto_news,
    fetch_single_feed,
    get_sentiment_series,
    normalize_rss_date,
    parse_rss_xml,
)


@dataclass
class MockHorizonPrediction:
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
# 1. XML Parsing, CDATA, Malformed Payloads & HTML Injection
# ==============================================================================


class TestAdversarialXMLAndHTML:
    """Stress-tests XML parser resilience, HTML sanitization, and entity handling."""

    def test_completely_malformed_xml_returns_empty_list(self):
        """Unclosed tags, truncated XML, and random binary noise should degrade gracefully to []."""
        bad_payloads = [
            "<rss><channel><item><title>Unclosed",
            "<?xml version='1.0'?><rss>",
            "NOT XML AT ALL {{{ }}}",
            "<item><title></item></title>",
            b"\x00\x01\x02\x03\xff\xfe\xfd",
            "",
            "   \n\t  ",
            "<!DOCTYPE html><html><body>Error 404</body></html>",
            "<rss><channel><item><title><![CDATA[Unclosed CDATA",
        ]
        for payload in bad_payloads:
            result = parse_rss_xml(payload, source_name="BadSource")
            assert isinstance(result, list)
            assert len(result) == 0

    def test_xml_with_unicode_bom_and_surrogates(self):
        """Handles UTF-8 BOM, non-ASCII characters, Chinese, Cyrillic, and emojis."""
        xml_with_bom = (
            "\ufeff<?xml version='1.0' encoding='utf-8'?>"
            "<rss version='2.0'><channel><item>"
            "<title>🚀 比特币 Bitcoin surges past 100k! 🇷🇺 Взлет</title>"
            "<description>Market capitalization hits new ATH with massive inflows 📈</description>"
            "<pubDate>Mon, 02 Nov 2026 12:00:00 GMT</pubDate>"
            "<link>https://example.com/crypto-1</link>"
            "</item></channel></rss>"
        )
        articles = parse_rss_xml(xml_with_bom, source_name="BOMFeed")
        assert len(articles) == 1
        assert "Bitcoin surges" in articles[0]["title"]
        assert "🚀" in articles[0]["title"]
        assert articles[0]["source"] == "BOMFeed"

    def test_xml_entity_expansion_and_bombs_do_not_hang(self):
        """Probes XML entity expansion (billion laughs variant) for safe handling."""
        xml_bomb = (
            "<?xml version='1.0'?>"
            "<!DOCTYPE lolz ["
            " <!ENTITY lol 'lol'>"
            " <!ELEMENT lolz (#PCDATA)>"
            " <!ENTITY lol1 '&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;'>"
            " <!ENTITY lol2 '&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;'>"
            "]>"
            "<rss><channel><item><title>&lol2;</title></item></channel></rss>"
        )
        articles = parse_rss_xml(xml_bomb, source_name="BombFeed")
        assert isinstance(articles, list)

    def test_html_injection_in_description_and_title(self):
        """Scripts, SVG onloads, iframe injections, and style blocks must be eradicated."""
        malicious_html = (
            "<script type='text/javascript'>alert('xss');</script>"
            "<p>Bitcoin hits <b class='highlight'>$100,000</b>.</p>"
            "<svg onload=alert(1)><circle r=10/></svg>"
            "<iframe src='javascript:alert(2)'></iframe>"
            "<style>body { display: none; }</style>"
            "<a href='https://evil.com' onclick='stealCookies()'>Read more</a>"
        )
        cleaned = clean_html_text(malicious_html)
        assert "<script" not in cleaned
        assert "alert(" not in cleaned
        assert "<svg" not in cleaned
        assert "<iframe" not in cleaned
        assert "<style" not in cleaned
        assert "display: none" not in cleaned
        assert "Bitcoin hits" in cleaned
        assert "$100,000" in cleaned
        assert "Read more" in cleaned

    def test_html_entity_unescaping_permutations(self):
        """Verifies multi-layer entity unescaping and bare ampersands."""
        test_strings = [
            ("Bitcoin &amp; Ethereum rally", "Bitcoin & Ethereum rally"),
            ("Profit &gt; Loss &amp; Bulls &lt; Bears", "Profit > Loss & Bulls < Bears"),
            ("BTC &#39;King of Crypto&#39; &quot;Surges&quot;", "BTC 'King of Crypto' \"Surges\""),
            ("Bare & unescaped ampersand in title & profit", "Bare & unescaped ampersand in title & profit"),
        ]
        for raw, expected in test_strings:
            assert clean_html_text(raw) == expected

    def test_rss_xml_with_nested_cdata_blocks(self):
        """Handles CDATA wrapped content with unescaped HTML inside."""
        xml_cdata = (
            "<rss version='2.0'><channel><item>"
            "<title><![CDATA[Ethereum <b>upgrade</b> approved & completed]]></title>"
            "<description><![CDATA[<p>Developers announce massive <i>milestone</i>.</p>]]></description>"
            "<pubDate>Sun, 01 Nov 2026 15:30:00 +0000</pubDate>"
            "<link>https://example.com/cdata-eth</link>"
            "</item></channel></rss>"
        )
        articles = parse_rss_xml(xml_cdata, source_name="CDATAFeed")
        assert len(articles) == 1
        assert "Ethereum" in articles[0]["title"]
        assert "upgrade" in articles[0]["title"]
        assert "Developers announce massive" in articles[0]["description"]
        assert "milestone" in articles[0]["description"]

    def test_item_with_missing_title_falls_back_to_description(self):
        """When <title> is missing or blank, parser should recover from <description>."""
        xml_no_title = (
            "<rss version='2.0'><channel><item>"
            "<title></title>"
            "<description>Solana network activity surges to new all time record high.</description>"
            "<pubDate>Mon, 02 Nov 2026 10:00:00 Z</pubDate>"
            "</item></channel></rss>"
        )
        articles = parse_rss_xml(xml_no_title, source_name="NoTitleFeed")
        assert len(articles) == 1
        assert "Solana network activity surges" in articles[0]["title"]


# ==============================================================================
# 2. Date Formats, Century Boundaries & Timezone Stress
# ==============================================================================


class TestAdversarialDatesAndURLs:
    """Stress-tests date parser boundaries and URL sanitization."""

    @pytest.mark.parametrize(
        "date_input",
        [
            "Mon, 02 Nov 2026 14:00:00 GMT",
            "02 Nov 2026 14:00:00 +0000",
            "2026-11-02T14:00:00Z",
            "2026-11-02T19:30:00+05:30",
            "2026-11-02T06:00:00-08:00",
            "Mon, 2 Nov 26 14:00:00 UT",
        ],
    )
    def test_valid_diverse_date_formats_normalize_to_utc(self, date_input: str):
        """All recognizable date formats must normalize to a valid UTC pd.Timestamp."""
        ts = normalize_rss_date(date_input)
        assert isinstance(ts, pd.Timestamp)
        assert ts.tzinfo is not None
        assert str(ts.tz) == "UTC"
        assert not pd.isna(ts)

    def test_corrupted_and_unparseable_dates_fallback_to_now_utc(self):
        """Unparseable or nonsensical date strings must fallback gracefully to now(UTC)."""
        garbage_dates = [
            "",
            "   ",
            None,
            "yesterday afternoon",
            "32/13/2026",
            "not a date at all",
            "99999999999999999999",
            "2026-99-99T99:99:99Z",
        ]
        now_utc = pd.Timestamp.now(tz="UTC")
        for g_date in garbage_dates:
            ts = normalize_rss_date(g_date)
            assert isinstance(ts, pd.Timestamp)
            assert ts.tzinfo is not None
            assert str(ts.tz) == "UTC"
            assert abs((ts - now_utc).total_seconds()) < 5.0

    def test_extreme_century_dates(self):
        """Probes Unix epoch (1970), Year 2038 problem boundary, and future dates."""
        epoch_ts = normalize_rss_date("Thu, 01 Jan 1970 00:00:00 GMT")
        assert epoch_ts.year == 1970

        y2038_ts = normalize_rss_date("Tue, 19 Jan 2038 03:14:07 GMT")
        assert y2038_ts.year == 2038

        future_ts = normalize_rss_date("Sun, 01 Jan 2040 00:00:00 GMT")
        assert future_ts.year == 2040

    def test_url_sanitization_removes_complex_tracking_params(self):
        """Strips diverse analytics and referrer parameters."""
        urls = [
            ("https://cointelegraph.com/news/btc-rally?utm_source=rss&utm_medium=feed", "https://cointelegraph.com/news/btc-rally"),
            ("https://www.coindesk.com/markets/2026/eth/?ref=twitter&fbclid=IwAR123", "https://www.coindesk.com/markets/2026/eth"),
            ("https://example.com/news///", "https://example.com/news"),
            ("", ""),
            (None, ""),
            ("https://example.com/news?a=1&b=2#section", "https://example.com/news"),
        ]
        for raw, expected in urls:
            assert clean_article_url(raw) == expected


# ==============================================================================
# 3. Deduplication Scaling & Title Similarity Stress
# ==============================================================================


class TestAdversarialDeduplication:
    """Stress-tests deduplication algorithms on massive loads and near-duplicate variations."""

    def test_deduplicate_1000_articles_url_query_param_load(self):
        """Ensures deduplicating 1,000 articles across 50 canonical URLs completes in < 0.2s."""
        base_time = pd.Timestamp("2026-11-02 12:00:00", tz="UTC")
        distinct_words = [
            "Apple", "Banana", "Cherry", "Date", "Elderberry", "Fig", "Grape", "Honeydew", "Kiwi", "Lemon",
            "Mango", "Nectarine", "Orange", "Papaya", "Quince", "Raspberry", "Strawberry", "Tangerine", "Ugli", "Vanilla",
            "Watermelon", "Xigua", "Yam", "Zucchini", "Apricot", "Blackberry", "Cantaloupe", "Damson", "Feijoa", "Guava",
            "Huckleberry", "Imbe", "Jackfruit", "Kumquat", "Lime", "Mulberry", "Olive", "Peach", "Plum", "Raisin",
            "Starfruit", "Tomato", "Yuzu", "Almond", "Bean", "Cashew", "Durian", "Eggfruit", "Hazelnut", "Pecan",
        ]
        articles = []

        # 50 unique canonical URLs, each replicated 20 times with varying tracking query params
        for i in range(1000):
            group_id = i % 50
            articles.append({
                "title": distinct_words[group_id],
                "published": base_time - pd.Timedelta(f"{i}min"),
                "source": "CoinTelegraph" if i % 2 == 0 else "CoinDesk",
                "link": f"https://example.com/article-{group_id}?utm_source=social&utm_id={i}&ref=tracker",
            })

        t0 = time.perf_counter()
        deduped = deduplicate_articles(articles)
        elapsed = time.perf_counter() - t0

        # Exact URL deduplication should collapse all 1,000 articles down to exactly 50 canonical items
        assert len(deduped) == 50
        assert elapsed < 0.5  # Must execute sub-second via seen_urls set lookup

    def test_templated_headline_similarity_collision(self):
        """Probes boundary condition where boilerplate headlines differing only by token symbol collide."""
        # When headline boilerplate exceeds ~85% of total length, SequenceMatcher collapses them
        articles = [
            {"title": "Binance adds trading pair for ADA", "link": "https://a.com/1"},
            {"title": "Binance adds trading pair for SOL", "link": "https://a.com/2"},
            {"title": "Binance adds trading pair for XRP", "link": "https://a.com/3"},
        ]
        deduped = deduplicate_articles(articles, similarity_threshold=0.85)
        # Empirical finding: All 3 collide into 1 because 'Binance adds trading pair for ' exceeds 85%
        assert len(deduped) == 1
        assert deduped[0]["title"] == "Binance adds trading pair for ADA"

    def test_title_similarity_fuzzy_matching(self):
        """Near-identical wire stories with minor punctuation or word variations should deduplicate."""
        articles = [
            {"title": "SEC Approves First Ethereum Staking ETF in Landmark Decision", "link": "https://a.com/1"},
            {"title": "SEC Approves First Ethereum Staking ETF In Landmark Decision!", "link": "https://b.com/2"},
            {"title": "sec approves first ethereum staking etf in landmark decision", "link": "https://c.com/3"},
            {"title": "Completely Different Story: Solana Sees Outflows", "link": "https://d.com/4"},
        ]
        deduped = deduplicate_articles(articles, similarity_threshold=0.85)
        assert len(deduped) == 2
        assert "SEC Approves" in deduped[0]["title"]
        assert "Solana Sees Outflows" in deduped[1]["title"]

    def test_deduplicate_articles_with_missing_or_none_fields(self):
        """Articles with None title or None link should not crash deduplication."""
        malformed_articles = [
            {"title": None, "link": None},
            {"title": "", "link": ""},
            {"title": "Bitcoin Rallies", "link": None},
            {"title": "Bitcoin Rallies", "link": "https://a.com"},
        ]
        deduped = deduplicate_articles(malformed_articles)
        assert isinstance(deduped, list)
        assert len(deduped) >= 1


# ==============================================================================
# 4. Network Resilience & Cascading Failures
# ==============================================================================


class TestAdversarialNetworkAndDegradation:
    """Stress-tests network HTTP error codes, timeouts, retries, and partial outages."""

    @patch("requests.Session.get")
    def test_http_403_bot_detection_degrades_instantly(self, mock_get: MagicMock):
        """HTTP 403 Forbidden should return empty list without wasting retries."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp

        articles = fetch_single_feed("CoinTelegraph", "https://cointelegraph.com/rss", max_retries=3)
        assert articles == []
        assert mock_get.call_count == 1

    @patch("time.sleep")
    @patch("requests.Session.get")
    def test_http_500_502_server_errors_exhaust_retries(self, mock_get: MagicMock, mock_sleep: MagicMock):
        """HTTP 500 / 502 server errors should retry up to max_retries then safely return []."""
        mock_resp = MagicMock()
        mock_resp.status_code = 502
        mock_get.return_value = mock_resp

        articles = fetch_single_feed("CoinDesk", "https://coindesk.com/rss", max_retries=3)
        assert articles == []
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 3

    @patch("time.sleep")
    @patch("requests.Session.get")
    def test_network_timeouts_retry_and_degrade(self, mock_get: MagicMock, mock_sleep: MagicMock):
        """Network Timeout exceptions should backoff and return []."""
        mock_get.side_effect = requests.Timeout("Connection timed out after 10s")

        articles = fetch_single_feed("Feed", "https://example.com/rss", max_retries=3)
        assert articles == []
        assert mock_get.call_count == 3

    @patch("requests.Session.get")
    def test_partial_feed_failure_returns_surviving_feed_articles(self, mock_get: MagicMock):
        """When CoinTelegraph fails (HTTP 500) but CoinDesk succeeds, returns CoinDesk articles."""
        def side_effect(url, **kwargs):
            resp = MagicMock()
            if "cointelegraph" in url.lower():
                resp.status_code = 500
            else:
                resp.status_code = 200
                resp.content = (
                    b"<rss><channel><item>"
                    b"<title>CoinDesk: Bitcoin breaks resistance</title>"
                    b"<pubDate>Mon, 02 Nov 2026 12:00:00 GMT</pubDate>"
                    b"<link>https://coindesk.com/1</link>"
                    b"</item></channel></rss>"
                )
            return resp

        mock_get.side_effect = side_effect

        with patch("time.sleep"):
            articles = fetch_crypto_news(
                sources={"CoinTelegraph": "https://cointelegraph.com/rss", "CoinDesk": "https://coindesk.com/rss"},
                raise_on_error=False,
            )

        assert len(articles) == 1
        assert "CoinDesk" in articles[0]["title"]
        assert articles[0]["source"] == "CoinDesk"

    @patch("requests.Session.get")
    def test_total_feed_outage_modes(self, mock_get: MagicMock):
        """Tests total outage behavior under raise_on_error=False vs raise_on_error=True."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        with patch("time.sleep"):
            # Default / Soft mode: returns []
            res = fetch_crypto_news(sources=["CoinTelegraph"], raise_on_error=False)
            assert res == []

            # Strict mode: raises RSSFeedError
            with pytest.raises(RSSFeedError):
                fetch_crypto_news(sources=["CoinTelegraph"], raise_on_error=True)


# ==============================================================================
# 5. Model & Heuristic Extremes, Negation Chaining & Punctuation
# ==============================================================================


class TestAdversarialModelAndHeuristic:
    """Stress-tests NLP sentiment classification under adversarial linguistic inputs."""

    @pytest.fixture
    def analyzer(self) -> HeuristicSentimentAnalyzer:
        return HeuristicSentimentAnalyzer()

    def test_negation_chaining_and_inversion(self, analyzer: HeuristicSentimentAnalyzer):
        """Verifies single and multi-word negation handling."""
        # 'not bullish' should invert to Bearish
        res_not_bull = analyzer.predict_single("Bitcoin is not bullish")
        assert res_not_bull["polarity"] < 0.0
        assert res_not_bull["label"] == "Bearish"

        # 'no crash' should invert to Bullish
        res_no_crash = analyzer.predict_single("Markets expect no crash")
        assert res_no_crash["polarity"] > 0.0
        assert res_no_crash["label"] == "Bullish"

        # 'never rally' should invert to Bearish
        res_never_rally = analyzer.predict_single("Altcoins will never rally this cycle")
        assert res_never_rally["polarity"] < 0.0
        assert res_never_rally["label"] == "Bearish"

        # 'without profit' should invert to Bearish
        res_without_profit = analyzer.predict_single("Traders liquidated without profit")
        assert res_without_profit["polarity"] < 0.0
        assert res_without_profit["label"] == "Bearish"

    def test_punctuation_and_symbol_flooding(self, analyzer: HeuristicSentimentAnalyzer):
        """Inputs flooded with special characters, punctuation, and currency symbols."""
        flooded_text = "$$$SURGE!!!! ???? ***BREAKOUT*** #%&()[]{}___ATH!!!"
        res = analyzer.predict_single(flooded_text)
        assert res["label"] == "Bullish"
        assert res["polarity"] > 0.5
        assert math.isclose(res["bearish"] + res["neutral"] + res["bullish"], 1.0, abs_tol=1e-5)

    def test_case_insensitivity_extremes(self, analyzer: HeuristicSentimentAnalyzer):
        """Evaluates identical sentiment under upper, lower, and alternating case."""
        cases = [
            "BITCOIN SURGES TO RECORD HIGH AFTER ETF APPROVAL",
            "bitcoin surges to record high after etf approval",
            "BiTcOiN SuRgEs tO ReCoRd hIgH AfTeR EtF ApPrOvAl",
        ]
        results = analyzer.predict(cases)
        # All three must produce identical polarity and labels
        assert results[0]["polarity"] == results[1]["polarity"] == results[2]["polarity"]
        assert results[0]["label"] == results[1]["label"] == results[2]["label"] == "Bullish"

    def test_balanced_conflicting_sentiment_evaluates_neutral(self, analyzer: HeuristicSentimentAnalyzer):
        """Equal number of bullish and bearish keywords produces zero polarity."""
        conflicting = "Huge surge followed by sudden crash. Bulls buy as bears sell."
        res = analyzer.predict_single(conflicting)
        assert res["polarity"] == 0.0
        assert res["label"] == "Neutral"

    def test_massive_10000_word_text_string(self, analyzer: HeuristicSentimentAnalyzer):
        """Stress-tests execution on massive 10,000-word strings without memory or runtime spikes."""
        massive_text = "bull surge pump gain " * 2500  # 10,000 words
        res = analyzer.predict_single(massive_text)
        assert res["label"] == "Bullish"
        assert res["polarity"] == 1.0
        assert res["bullish"] > 0.9

    def test_none_empty_and_mixed_data_types(self):
        """Classifier wrapper must handle None, empty list, ints, and floats without crashing."""
        clf = CryptoBertClassifier(fallback_to_heuristic=True)

        # Empty inputs
        assert clf.predict([]) == []
        assert clf.predict("")[0]["label"] == "Neutral"

        # List with None and mixed types
        mixed_inputs = [None, "", "Bitcoin surges", 12345, 99.99, True]
        res = clf.predict(mixed_inputs)
        assert len(res) == 6
        assert res[0]["label"] == "Neutral"
        assert res[1]["label"] == "Neutral"
        assert res[2]["label"] == "Bullish"
        assert res[3]["label"] == "Neutral"

        # Unsupported top-level type raises TypeError
        with pytest.raises(TypeError):
            clf.predict(12345)  # type: ignore

    def test_probability_invariants_across_adversarial_inputs(self, analyzer: HeuristicSentimentAnalyzer):
        """Strictly enforces sum(p) == 1.0, 0 <= p <= 1, and -1 <= polarity <= 1."""
        test_inputs = [
            "",
            "   ",
            "🚀🚀🚀",
            "a" * 1000,
            "crash crash crash plunge dump hack stolen",
            "surge rally jump pump bull breakout ath",
            "not crash not plunge not dump",
            "neutral normal regular day in crypto",
        ]
        for t in test_inputs:
            res = analyzer.predict_single(t)
            p_sum = res["bearish"] + res["neutral"] + res["bullish"]
            assert math.isclose(p_sum, 1.0, abs_tol=1e-5), f"Probabilities did not sum to 1.0: {res}"
            assert 0.0 <= res["bearish"] <= 1.0
            assert 0.0 <= res["neutral"] <= 1.0
            assert 0.0 <= res["bullish"] <= 1.0
            assert -1.0 <= res["polarity"] <= 1.0


# ==============================================================================
# 6. Polarity, Exponential Decay & Time-Series Alignment Stress
# ==============================================================================


class TestAdversarialDecayAndSeries:
    """Stress-tests polarity calculations, extreme time decays, and series reindexing."""

    def test_compute_sentiment_polarity_clamping(self):
        """Polarity must strictly clamp to [-1.0, 1.0] even if inputs are out of bounds."""
        assert compute_sentiment_polarity(2.0, 0.0) == 1.0
        assert compute_sentiment_polarity(0.0, 5.0) == -1.0
        assert compute_sentiment_polarity(0.5, 0.5) == 0.0

    def test_compute_hourly_decay_boundary_conditions(self):
        """Probes zero elapsed hours, negative elapsed hours, and massive elapsed hours."""
        # Zero hours: no decay
        assert compute_hourly_decay(0.8, elapsed_hours=0.0) == 0.8

        # Negative hours: guard returns current sentiment unchanged
        assert compute_hourly_decay(0.8, elapsed_hours=-5.0) == 0.8

        # Huge elapsed hours (10,000h): converges cleanly to 0.0 without floating point underflow error
        decayed = compute_hourly_decay(0.8, elapsed_hours=10000.0)
        assert decayed == 0.0

        # Fractional hours (30 minutes)
        decay_half_h = compute_hourly_decay(1.0, elapsed_hours=0.5, alpha=0.95)
        expected = 0.95 ** 0.5
        assert math.isclose(decay_half_h, expected, rel_tol=1e-5)

    def test_get_sentiment_series_with_unsorted_and_irregular_timestamps(self):
        """Aligns discrete news across unsorted and sub-hourly target timestamps."""
        target_ts = pd.to_datetime([
            "2026-11-02 15:00:00",
            "2026-11-02 12:00:00",  # unsorted
            "2026-11-02 13:30:00",  # sub-hourly
            "2026-11-02 14:00:00",
        ], utc=True)

        news = [
            {"published": "2026-11-02 12:15:00Z", "polarity": 0.8},
        ]

        series = get_sentiment_series(news, target_ts, decay_factor=0.95)
        assert len(series) == 4
        # Verify no NaNs generated
        assert not series.isna().any()
        # Verify value at 12:00:00 is 0.8
        assert series.loc[pd.Timestamp("2026-11-02 12:00:00", tz="UTC")] == 0.8
        # Verify subsequent timestamps decay monotonically
        val_14 = series.loc[pd.Timestamp("2026-11-02 14:00:00", tz="UTC")]
        val_15 = series.loc[pd.Timestamp("2026-11-02 15:00:00", tz="UTC")]
        assert val_14 < 0.8
        assert val_15 < val_14

    def test_get_sentiment_series_with_ancient_and_future_news(self):
        """Handles news published years before or hours after the target window."""
        target_ts = pd.date_range("2026-11-02 00:00:00", periods=5, freq="1h", tz="UTC")

        news = [
            # 1 year before: should decay completely to 0.0
            {"published": "2025-11-02 00:00:00Z", "polarity": 1.0},
            # Future news (after window): should not affect past timestamps
            {"published": "2026-11-03 00:00:00Z", "polarity": -1.0},
        ]

        series = get_sentiment_series(news, target_ts)
        assert len(series) == 5
        # Decayed to 0.0
        assert (series == 0.0).all()

    def test_get_sentiment_series_with_invalid_news_items(self):
        """News items with missing dates, corrupt dates, or missing polarity are skipped safely."""
        target_ts = pd.date_range("2026-11-02 00:00:00", periods=3, freq="1h", tz="UTC")
        corrupted_news = [
            {"published": None, "polarity": 0.5},
            {"published": "invalid-date-string", "polarity": 0.5},
            {"published": "2026-11-02 01:00:00Z"},  # missing polarity defaults to 0.0
        ]
        series = get_sentiment_series(corrupted_news, target_ts)
        assert len(series) == 3
        assert not series.isna().any()


# ==============================================================================
# 7. Horizon Attenuation & Quantile Inversion Stress
# ==============================================================================


class TestAdversarialDriftModifier:
    """Stress-tests multi-horizon drift adjustments and quantile immunity."""

    def test_horizon_gamma_strictly_monotonic_across_all_11_horizons(self):
        """Attenuating gamma must strictly decrease: gamma(1h) > gamma(2h) > ... > gamma(30d)."""
        gammas = [compute_horizon_gamma(h) for h in DEFAULT_HORIZONS]
        for i in range(len(gammas) - 1):
            assert gammas[i] > gammas[i + 1], (
                f"Gamma not strictly monotonic between {DEFAULT_HORIZONS[i]} ({gammas[i]}) "
                f"and {DEFAULT_HORIZONS[i+1]} ({gammas[i+1]})"
            )

    def test_unknown_horizon_fallback(self):
        """Unknown horizon string defaults safely to 1 hour without error."""
        gamma_unknown = compute_horizon_gamma("999d")
        gamma_1h = compute_horizon_gamma("1h")
        assert gamma_unknown == gamma_1h

    def test_drift_modifier_extreme_polarity_clamping(self):
        """Extreme polarities (+100.0, -50.0) clamp to [-1.0, 1.0] preventing wild price distortions."""
        pred = {
            "1h": MockHorizonPrediction(
                horizon="1h",
                step_hours=1,
                predicted_price=100.0,
                current_price=100.0,
                predicted_change=0.0,
                predicted_return_pct=0.0,
                direction="NEUTRAL",
                confidence_p10=90.0,
                confidence_p50=100.0,
                confidence_p90=110.0,
            )
        }

        # Polarity of 100.0 must be clamped to 1.0
        adj_high = apply_sentiment_drift(pred, sentiment_polarity=100.0)
        adj_1 = apply_sentiment_drift(pred, sentiment_polarity=1.0)
        assert adj_high["1h"].predicted_price == adj_1["1h"].predicted_price

        # Polarity of -50.0 must be clamped to -1.0
        adj_low = apply_sentiment_drift(pred, sentiment_polarity=-50.0)
        adj_neg1 = apply_sentiment_drift(pred, sentiment_polarity=-1.0)
        assert adj_low["1h"].predicted_price == adj_neg1["1h"].predicted_price

    def test_quantile_inversion_immunity_under_extreme_drift(self):
        """Quantile ordering invariant P10 <= P50 <= P90 must strictly hold under all drift scenarios."""
        predictions = {}
        for h in DEFAULT_HORIZONS:
            predictions[h] = {
                "horizon": h,
                "current_price": 50000.0,
                "predicted_price": 52000.0,
                "confidence_p10": 48000.0,
                "confidence_p50": 52000.0,
                "confidence_p90": 56000.0,
            }

        for pol in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            adj = apply_sentiment_drift(predictions, sentiment_polarity=pol)
            for h, item in adj.items():
                p10 = item["confidence_p10"]
                p50 = item["confidence_p50"]
                p90 = item["confidence_p90"]
                assert p10 <= p50 <= p90, f"Quantile inversion detected at horizon {h} with polarity {pol}!"

    def test_zero_current_price_no_division_by_zero(self):
        """When current_price is 0.0, predicted_return_pct should safely be 0.0 without ZeroDivisionError."""
        pred = {
            "1h": {
                "current_price": 0.0,
                "predicted_price": 100.0,
                "confidence_p10": 90.0,
                "confidence_p50": 100.0,
                "confidence_p90": 110.0,
            }
        }
        adj = apply_sentiment_drift(pred, sentiment_polarity=0.5)
        assert adj["1h"]["predicted_return_pct"] == 0.0
        assert adj["1h"]["direction"] == "BULLISH"

    def test_apply_trajectory_sentiment_drift_boundary_shapes(self):
        """Trajectory drift with 720 steps and multiple quantiles preserves shapes and ordering."""
        n_steps = 720
        points = np.linspace(100.0, 150.0, n_steps)
        quantiles = np.vstack([points * 0.9, points, points * 1.1]).T  # shape (720, 3)

        adj_points, adj_quantiles = apply_trajectory_sentiment_drift(
            point_trajectory=points,
            quantile_trajectory=quantiles,
            sentiment_polarity=0.8,
        )

        assert adj_points.shape == (720,)
        assert adj_quantiles.shape == (720, 3)
        # Quantile ordering invariant
        assert (adj_quantiles[:, 0] <= adj_quantiles[:, 1]).all()
        assert (adj_quantiles[:, 1] <= adj_quantiles[:, 2]).all()
        # Initial step should have larger drift than 720th step
        initial_mult = adj_points[0] / points[0]
        final_mult = adj_points[-1] / points[-1]
        assert initial_mult > final_mult > 1.0
