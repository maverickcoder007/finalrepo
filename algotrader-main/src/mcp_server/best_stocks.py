"""
Best Analytical Stock Picker — Uses the existing scanner's trend scores,
super-performance flags, and technical triggers to rank the top analytical
picks from the NIFTY 500 universe, then enriches them with news + report
links.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup

from .news_fetcher import fetch_latest_news
from .annual_report_fetcher import fetch_annual_reports


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}


# ──────────────────────────────────────────────────────────────
# Screener.in — Top Trending / High-Quality Stocks
# ──────────────────────────────────────────────────────────────

async def _fetch_screener_top_stocks(screen_name: str = "high-growth", limit: int = 10) -> list[dict[str, Any]]:
    """
    Scrape top picks from popular Screener.in public screens.
    Built-in screens: high-growth, value-picks, fundamentally-strong
    """
    screens = {
        "high-growth": "https://www.screener.in/screens/71064/high-growth-stocks/",
        "value-picks": "https://www.screener.in/screens/218/value-stocks/",
        "fundamentally-strong": "https://www.screener.in/screens/315/good-stocks/",
    }
    url = screens.get(screen_name, screens["high-growth"])
    stocks: list[dict[str, Any]] = []

    async with httpx.AsyncClient(headers=_HEADERS, timeout=15, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return stocks
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("table tbody tr")
            for row in rows[:limit]:
                cells = row.select("td")
                if len(cells) >= 2:
                    link = cells[0].select_one("a")
                    if link:
                        name = link.get_text(strip=True)
                        href = link.get("href", "")
                        # Extract symbol from URL like /company/RELIANCE/
                        symbol = href.strip("/").split("/")[-1] if href else name
                        stocks.append({
                            "symbol": symbol.upper(),
                            "name": name,
                            "screen": screen_name,
                            "url": f"https://www.screener.in{href}" if href else "",
                        })
        except Exception:
            pass

    return stocks


# ──────────────────────────────────────────────────────────────
# MoneyControl Top Gainers / Most Active
# ──────────────────────────────────────────────────────────────

async def _fetch_moneycontrol_top_movers(limit: int = 10) -> list[dict[str, Any]]:
    """Fetch top gainers/most active from MoneyControl."""
    url = "https://www.moneycontrol.com/stocks/marketstats/nsegainer/index.php"
    stocks: list[dict[str, Any]] = []

    async with httpx.AsyncClient(headers=_HEADERS, timeout=15, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return stocks
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.select("table.tbldata tbody tr, table.bsr_table tbody tr")
            for row in rows[:limit]:
                cells = row.select("td")
                if len(cells) >= 3:
                    link = cells[0].select_one("a")
                    if link:
                        name = link.get_text(strip=True)
                        stocks.append({
                            "name": name,
                            "source": "MoneyControl Top Movers",
                        })
        except Exception:
            pass

    return stocks


# ──────────────────────────────────────────────────────────────
# TrendLyne — Best Stocks to Buy Today
# ──────────────────────────────────────────────────────────────

async def _fetch_trendlyne_best_picks(limit: int = 10) -> list[dict[str, Any]]:
    """Fetch top-rated stocks from TrendLyne."""
    url = "https://trendlyne.com/stock-screeners/best-to-buy-stocks/"
    stocks: list[dict[str, Any]] = []

    async with httpx.AsyncClient(headers=_HEADERS, timeout=15, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return stocks
            soup = BeautifulSoup(resp.text, "html.parser")
            items = soup.select("a.stock-link, td a[href*='/equity/']")
            seen: set[str] = set()
            for a in items[:limit * 2]:
                name = a.get_text(strip=True)
                href = a.get("href", "")
                if name and name not in seen:
                    seen.add(name)
                    stocks.append({
                        "name": name,
                        "url": href if href.startswith("http") else f"https://trendlyne.com{href}",
                        "source": "TrendLyne",
                    })
                    if len(stocks) >= limit:
                        break
        except Exception:
            pass

    return stocks


# ──────────────────────────────────────────────────────────────
# Internal Scanner Integration Helper
# ──────────────────────────────────────────────────────────────

def rank_from_scanner_cache(scanner_results: dict[str, Any], top_n: int = 10) -> list[dict[str, Any]]:
    """
    Given cached scanner results from the algotrader scanner module,
    rank the top N stocks by trend_score, super-performance flags, and
    trigger counts.
    """
    if not scanner_results:
        return []

    scored: list[dict[str, Any]] = []
    for symbol, data in scanner_results.items():
        if not isinstance(data, dict):
            continue
        trend_score = data.get("trend_score", 0)
        super_perf = data.get("super_performance", {})
        triggers = data.get("triggers", [])
        accumulation = data.get("accumulation", {})

        # Composite score
        sp_passed = sum(1 for v in super_perf.values() if v is True) if isinstance(super_perf, dict) else 0
        trigger_count = len(triggers) if isinstance(triggers, list) else 0
        accum_score = 1 if accumulation.get("institutional_accumulation") else 0

        composite = (
            trend_score * 0.5
            + sp_passed * 3.0
            + trigger_count * 2.0
            + accum_score * 5.0
        )

        scored.append({
            "symbol": symbol,
            "sector": data.get("sector", ""),
            "trend_score": round(trend_score, 2),
            "super_performance_passed": sp_passed,
            "trigger_count": trigger_count,
            "institutional_accumulation": accum_score > 0,
            "composite_score": round(composite, 2),
            "triggers": [t.get("type", "") for t in triggers] if isinstance(triggers, list) else [],
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored[:top_n]


# ──────────────────────────────────────────────────────────────
# Public API — Best Analytical Picks (enriched)
# ──────────────────────────────────────────────────────────────

async def get_best_analytical_stocks(
    scanner_cache: dict[str, Any] | None = None,
    top_n: int = 5,
    enrich: bool = True,
) -> dict[str, Any]:
    """
    Get the top analytical stock picks and optionally enrich each with
    latest news and annual report links.

    Args:
        scanner_cache: Optional cached results from the algotrader scanner.
                      If None, falls back to public screener data.
        top_n: Number of top stocks to return.
        enrich: Whether to fetch news and annual reports for each pick.

    Returns:
        Structured dict with ranked picks and their enrichments.
    """
    picks: list[dict[str, Any]] = []

    # Primary: use internal scanner rankings if available
    if scanner_cache:
        picks = rank_from_scanner_cache(scanner_cache, top_n=top_n)

    # Fallback: use public screener data
    if not picks:
        screener_stocks = await _fetch_screener_top_stocks("high-growth", limit=top_n)
        for s in screener_stocks:
            picks.append({
                "symbol": s["symbol"],
                "name": s.get("name", s["symbol"]),
                "source": "Screener.in (High Growth)",
                "composite_score": None,
                "trend_score": None,
            })

    # Enrich each pick with news + annual reports
    enriched_picks: list[dict[str, Any]] = []
    if enrich and picks:
        for pick in picks:
            symbol = pick["symbol"]
            try:
                news_task = fetch_latest_news(symbol, limit=5)
                report_task = fetch_annual_reports(symbol, limit=3)
                news_result, report_result = await asyncio.gather(
                    news_task, report_task,
                    return_exceptions=True,
                )
                pick["latest_news"] = news_result if isinstance(news_result, dict) else {"error": str(news_result)}
                pick["annual_reports"] = report_result if isinstance(report_result, dict) else {"error": str(report_result)}
            except Exception as e:
                pick["latest_news"] = {"error": str(e)}
                pick["annual_reports"] = {"error": str(e)}
            enriched_picks.append(pick)
    else:
        enriched_picks = picks

    return {
        "fetched_at": datetime.now().isoformat(),
        "total_picks": len(enriched_picks),
        "source": "Internal Scanner" if scanner_cache else "Public Screeners",
        "picks": enriched_picks,
    }
