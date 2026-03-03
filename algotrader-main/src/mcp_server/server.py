"""
Stock Research MCP Server
═════════════════════════
A Model Context Protocol (MCP) server that exposes tools for:

  1. fetch_latest_news      — Aggregate latest news for any NSE stock
  2. fetch_annual_reports   — Fetch annual report links & financial highlights
  3. get_best_stocks        — Rank top analytical picks (enriched with news/reports)
  4. get_stock_research     — Full research report for a single stock

Run standalone:
    python -m src.mcp_server.server

Or integrate as a subprocess with any MCP-compatible client.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

# ── Import fetchers ──────────────────────────────────────────
from .news_fetcher import fetch_latest_news
from .annual_report_fetcher import fetch_annual_reports
from .best_stocks import get_best_analytical_stocks, rank_from_scanner_cache


# ── Create MCP server ────────────────────────────────────────
mcp = FastMCP(
    "StockResearch",
    instructions=(
        "MCP server for Indian equity research — fetches latest news, "
        "annual reports, and identifies best analytical stock picks from "
        "NSE/BSE with data from Google News, MoneyControl, Economic Times, "
        "Screener.in, TrendLyne, BSE India, and NSE India."
    ),
)

# ── Shared scanner cache (populated via integration) ─────────
_scanner_cache: dict[str, Any] = {}


def set_scanner_cache(cache: dict[str, Any]) -> None:
    """Inject scanner results from the algotrader for enriched rankings."""
    global _scanner_cache
    _scanner_cache = cache


# ══════════════════════════════════════════════════════════════
# MCP Tools
# ══════════════════════════════════════════════════════════════


@mcp.tool()
async def fetch_stock_news(symbol: str, limit: int = 15) -> str:
    """
    Fetch the latest news for an NSE stock symbol from multiple sources.

    Aggregates articles from Google News RSS, MoneyControl, Economic Times,
    and NSE corporate announcements.

    Args:
        symbol: NSE trading symbol (e.g., RELIANCE, TCS, INFY, HDFCBANK)
        limit: Maximum number of articles to return (default: 15)

    Returns:
        JSON with articles containing title, source, published date, and URL.
    """
    result = await fetch_latest_news(symbol, limit=limit)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def fetch_stock_annual_reports(symbol: str, limit: int = 10) -> str:
    """
    Fetch annual report links and financial highlights for an NSE stock.

    Searches BSE India, NSE India, Google, TrendLyne, and Screener.in for
    annual report PDFs and extracts key financial metrics.

    Args:
        symbol: NSE trading symbol (e.g., RELIANCE, TCS, INFY)
        limit: Maximum number of report links to return (default: 10)

    Returns:
        JSON with report links, financial highlights (PE, market cap, ROE, etc),
        and pros/cons from Screener.in.
    """
    result = await fetch_annual_reports(symbol, limit=limit)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def get_best_stocks(top_n: int = 5, enrich: bool = True) -> str:
    """
    Get the top analytical stock picks ranked by composite score.

    Uses the algotrader's internal scanner (Minervini super-performance,
    trend scoring, VCP detection, institutional accumulation) when available,
    or falls back to public screener data from Screener.in.

    Each pick is enriched with latest news and annual report links.

    Args:
        top_n: Number of top stocks to return (default: 5)
        enrich: Whether to enrich picks with news and reports (default: True)

    Returns:
        JSON with ranked picks, each containing composite score, triggers,
        latest news, and annual report links.
    """
    result = await get_best_analytical_stocks(
        scanner_cache=_scanner_cache if _scanner_cache else None,
        top_n=top_n,
        enrich=enrich,
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
async def get_stock_research(symbol: str) -> str:
    """
    Get a comprehensive research report for a single NSE stock.

    Combines latest news, annual reports, financial highlights, and
    scanner analysis (if available) into one unified research package.

    Args:
        symbol: NSE trading symbol (e.g., RELIANCE, TCS, INFY)

    Returns:
        JSON with full research data: news, annual reports, financial
        highlights, and scanner analysis if available.
    """
    symbol = symbol.upper().strip()

    # Fetch news and reports concurrently
    news_task = fetch_latest_news(symbol, limit=10)
    reports_task = fetch_annual_reports(symbol, limit=5)

    news_result, reports_result = await asyncio.gather(
        news_task, reports_task,
        return_exceptions=True,
    )

    # Scanner data if available
    scanner_data = None
    if _scanner_cache and symbol in _scanner_cache:
        raw = _scanner_cache[symbol]
        scanner_data = {
            "trend_score": raw.get("trend_score"),
            "sector": raw.get("sector"),
            "super_performance": raw.get("super_performance"),
            "triggers": raw.get("triggers"),
            "accumulation": raw.get("accumulation"),
        }

    research = {
        "symbol": symbol,
        "latest_news": news_result if isinstance(news_result, dict) else {"error": str(news_result)},
        "annual_reports": reports_result if isinstance(reports_result, dict) else {"error": str(reports_result)},
        "scanner_analysis": scanner_data,
    }

    return json.dumps(research, indent=2, default=str)


@mcp.tool()
async def analyze_intraday_oi_flow(underlying: str = "NIFTY", analysis_date: str = "") -> str:
    """
    Analyze intraday option chain OI flow to determine market direction.

    Reads 15-minute option chain snapshots captured throughout the trading
    day and analyses how OI, PCR, straddle premium, IV skew, and OI walls
    change over time for ATM and near-ATM strikes.

    Provides:
      • Per-interval flow changes and signals
      • PCR trend, straddle trend, IV skew evolution
      • OI wall (support/resistance) movement
      • Smart money detection signals
      • Overall market direction with confidence score
      • Trend reinforcement (consecutive confirmations)

    Args:
        underlying: Index to analyze — "NIFTY" or "SENSEX" (default: NIFTY)
        analysis_date: Date in YYYY-MM-DD format (default: today)

    Returns:
        JSON with comprehensive intraday OI flow analysis including
        market direction, confidence, trend strength, and actionable summary.
    """
    from src.options.oi_intraday_analyzer import get_intraday_oi_analyzer

    analyzer = get_intraday_oi_analyzer()
    report = analyzer.analyze(
        underlying=underlying,
        analysis_date=analysis_date or None,
    )
    return json.dumps(report.model_dump(), indent=2, default=str)


@mcp.tool()
async def search_stock_news_by_topic(topic: str, limit: int = 10) -> str:
    """
    Search for stock market news on a specific topic or sector.

    Useful for broad market research like "banking sector", "IT stocks rally",
    "quarterly results", "FII buying", etc.

    Args:
        topic: Search topic (e.g., "Nifty 50 rally", "pharma sector Q3 results")
        limit: Maximum number of articles to return (default: 10)

    Returns:
        JSON with relevant articles from Google News.
    """
    import html as html_mod
    import feedparser
    import httpx as _httpx

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    url = f"https://news.google.com/rss/search?q={topic}+India+stock+market&hl=en-IN&gl=IN&ceid=IN:en"
    articles: list[dict[str, str]] = []

    async with _httpx.AsyncClient(headers=headers, timeout=15, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                feed = feedparser.parse(resp.text)
                for entry in feed.entries[:limit]:
                    articles.append({
                        "title": html_mod.unescape(entry.get("title", "")).strip(),
                        "source": entry.get("source", {}).get("title", "Google News"),
                        "published": entry.get("published", ""),
                        "url": entry.get("link", ""),
                    })
        except Exception:
            pass

    from datetime import datetime as _dt
    result = {
        "topic": topic,
        "fetched_at": _dt.now().isoformat(),
        "total_articles": len(articles),
        "articles": articles,
    }
    return json.dumps(result, indent=2, default=str)


# ══════════════════════════════════════════════════════════════
# MCP Resources (context data the client can read)
# ══════════════════════════════════════════════════════════════


@mcp.resource("stock://scanner/top-picks")
async def scanner_top_picks_resource() -> str:
    """Current top picks from the algotrader scanner (if a scan has been run)."""
    if not _scanner_cache:
        return json.dumps({"message": "No scanner data available. Run a scan first."})
    picks = rank_from_scanner_cache(_scanner_cache, top_n=10)
    return json.dumps(picks, indent=2, default=str)


@mcp.resource("stock://scanner/summary")
async def scanner_summary_resource() -> str:
    """Summary of the last scanner run."""
    if not _scanner_cache:
        return json.dumps({"message": "No scanner data available."})
    return json.dumps({
        "total_stocks_scanned": len(_scanner_cache),
        "symbols": list(_scanner_cache.keys())[:50],
    }, indent=2)


# ══════════════════════════════════════════════════════════════
# MCP Prompts (reusable prompt templates)
# ══════════════════════════════════════════════════════════════


@mcp.prompt()
def stock_analysis_prompt(symbol: str) -> str:
    """Generate a prompt for comprehensive stock analysis."""
    return f"""Analyze the Indian equity {symbol} listed on NSE.

Please use the following tools to gather data:
1. Call `fetch_stock_news` for latest news on {symbol}
2. Call `fetch_stock_annual_reports` for annual reports of {symbol}
3. Call `get_stock_research` for the full research report on {symbol}

Then provide:
- Current market sentiment based on recent news
- Key financial highlights from the annual report
- Technical analysis insights (if scanner data is available)
- Overall investment thesis (bullish/bearish/neutral) with reasoning
- Key risks to monitor
"""


@mcp.prompt()
def market_overview_prompt() -> str:
    """Generate a prompt for market overview analysis."""
    return """Provide a comprehensive Indian stock market overview.

Please use the following tools:
1. Call `get_best_stocks` with top_n=10 to get the best analytical picks
2. Call `search_stock_news_by_topic` with topic "Nifty 50 market today"
3. Call `search_stock_news_by_topic` with topic "FII DII activity India"

Then summarize:
- Top picks and why they stand out
- Market sentiment and key themes
- Sector rotation insights
- Key risks and upcoming events
"""


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════


def main() -> None:
    """Run the MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
