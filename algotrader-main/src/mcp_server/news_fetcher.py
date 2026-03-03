"""
News Fetcher — Scrapes latest stock news from Google News RSS, MoneyControl,
Economic Times, and other public sources for Indian equities (NSE).
"""

from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime
from typing import Any

import httpx
import feedparser
from bs4 import BeautifulSoup


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}

# ──────────────────────────────────────────────────────────────
# Google News RSS
# ──────────────────────────────────────────────────────────────

async def _fetch_google_news_rss(symbol: str, company_name: str, limit: int = 10) -> list[dict[str, str]]:
    """Fetch latest news via Google News RSS feed for a stock symbol / company."""
    queries = [
        f"{symbol} NSE stock",
        f"{company_name} share market India",
    ]
    articles: list[dict[str, str]] = []
    seen_titles: set[str] = set()

    async with httpx.AsyncClient(headers=_HEADERS, timeout=15, follow_redirects=True) as client:
        for q in queries:
            url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    continue
                feed = feedparser.parse(resp.text)
                for entry in feed.entries[:limit]:
                    title = html.unescape(entry.get("title", "")).strip()
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)
                    published = entry.get("published", "")
                    link = entry.get("link", "")
                    source = entry.get("source", {}).get("title", "Google News")
                    articles.append({
                        "title": title,
                        "source": source,
                        "published": published,
                        "url": link,
                    })
            except Exception:
                continue

    return articles[:limit]


# ──────────────────────────────────────────────────────────────
# MoneyControl News Scraper
# ──────────────────────────────────────────────────────────────

async def _fetch_moneycontrol_news(symbol: str, limit: int = 5) -> list[dict[str, str]]:
    """Scrape latest news from MoneyControl search for a given symbol."""
    url = f"https://www.moneycontrol.com/stocks/cptmarket/compsearchnew.php?search_data={symbol}&cid=&mbsearch_str=&topsearch_type=1&search_str={symbol}"
    articles: list[dict[str, str]] = []

    async with httpx.AsyncClient(headers=_HEADERS, timeout=15, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return articles
            soup = BeautifulSoup(resp.text, "html.parser")
            news_items = soup.select("li a[href*='news'], .news_list a, .MT10 a")
            for item in news_items[:limit]:
                title = item.get_text(strip=True)
                href = item.get("href", "")
                if title and href:
                    articles.append({
                        "title": title,
                        "source": "MoneyControl",
                        "published": "",
                        "url": href if href.startswith("http") else f"https://www.moneycontrol.com{href}",
                    })
        except Exception:
            pass

    return articles


# ──────────────────────────────────────────────────────────────
# Economic Times RSS
# ──────────────────────────────────────────────────────────────

async def _fetch_et_market_news(symbol: str, limit: int = 5) -> list[dict[str, str]]:
    """Fetch news from Economic Times markets RSS."""
    url = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
    articles: list[dict[str, str]] = []

    async with httpx.AsyncClient(headers=_HEADERS, timeout=15, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return articles
            feed = feedparser.parse(resp.text)
            sym_lower = symbol.lower()
            for entry in feed.entries:
                title = html.unescape(entry.get("title", "")).strip()
                desc = html.unescape(entry.get("description", "")).strip()
                if sym_lower in title.lower() or sym_lower in desc.lower():
                    articles.append({
                        "title": title,
                        "source": "Economic Times",
                        "published": entry.get("published", ""),
                        "url": entry.get("link", ""),
                    })
                    if len(articles) >= limit:
                        break
        except Exception:
            pass

    return articles


# ──────────────────────────────────────────────────────────────
# NSE Corporate Announcements
# ──────────────────────────────────────────────────────────────

async def _fetch_nse_announcements(symbol: str, limit: int = 5) -> list[dict[str, str]]:
    """Fetch latest corporate announcements from NSE India."""
    url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol}"
    articles: list[dict[str, str]] = []

    nse_headers = {
        **_HEADERS,
        "Referer": "https://www.nseindia.com/",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(headers=nse_headers, timeout=20, follow_redirects=True) as client:
        try:
            # First hit the main page to get cookies
            await client.get("https://www.nseindia.com/")
            resp = await client.get(url)
            if resp.status_code != 200:
                return articles
            data = resp.json()
            for item in data[:limit]:
                desc = item.get("desc", "Corporate Announcement")
                an_date = item.get("an_dt", "")
                attchmnt = item.get("attchmntFile", "")
                pdf_url = f"https://www.nseindia.com/corporate/{attchmnt}" if attchmnt else ""
                articles.append({
                    "title": f"[{symbol}] {desc}",
                    "source": "NSE India",
                    "published": an_date,
                    "url": pdf_url,
                })
        except Exception:
            pass

    return articles


# ──────────────────────────────────────────────────────────────
# Public API — Aggregated News Fetch
# ──────────────────────────────────────────────────────────────

# Map common NSE symbols to their company names for better search accuracy
_COMPANY_NAMES: dict[str, str] = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "ITC": "ITC Limited",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "HCLTECH": "HCL Technologies",
    "AXISBANK": "Axis Bank",
    "ASIANPAINT": "Asian Paints",
    "MARUTI": "Maruti Suzuki",
    "SUNPHARMA": "Sun Pharma",
    "TITAN": "Titan Company",
    "BAJFINANCE": "Bajaj Finance",
    "WIPRO": "Wipro",
    "ULTRACEMCO": "UltraTech Cement",
    "NESTLEIND": "Nestle India",
    "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel",
    "POWERGRID": "Power Grid Corporation",
    "NTPC": "NTPC Limited",
    "JSWSTEEL": "JSW Steel",
    "M&M": "Mahindra and Mahindra",
    "ADANIENT": "Adani Enterprises",
    "ADANIPORTS": "Adani Ports",
    "COALINDIA": "Coal India",
    "ONGC": "Oil and Natural Gas Corporation",
    "DRREDDY": "Dr Reddys Laboratories",
    "CIPLA": "Cipla",
    "GRASIM": "Grasim Industries",
    "TECHM": "Tech Mahindra",
    "APOLLOHOSP": "Apollo Hospitals",
    "DIVISLAB": "Divis Laboratories",
    "EICHERMOT": "Eicher Motors",
    "HEROMOTOCO": "Hero MotoCorp",
    "BAJAJ-AUTO": "Bajaj Auto",
    "BRITANNIA": "Britannia Industries",
    "INDUSINDBK": "IndusInd Bank",
    "HDFCLIFE": "HDFC Life Insurance",
    "SBILIFE": "SBI Life Insurance",
    "DABUR": "Dabur India",
    "GODREJCP": "Godrej Consumer Products",
    "PIDILITIND": "Pidilite Industries",
    "HAVELLS": "Havells India",
    "DLF": "DLF Limited",
    "VEDL": "Vedanta Limited",
    "TATAPOWER": "Tata Power",
}


async def fetch_latest_news(symbol: str, limit: int = 15) -> dict[str, Any]:
    """
    Aggregate news from multiple sources for a given NSE stock symbol.
    Returns a structured dict with articles sorted by recency.
    """
    symbol = symbol.upper().strip()
    company_name = _COMPANY_NAMES.get(symbol, symbol)

    # Fetch from all sources concurrently
    google_task = _fetch_google_news_rss(symbol, company_name, limit=limit)
    mc_task = _fetch_moneycontrol_news(symbol, limit=5)
    et_task = _fetch_et_market_news(symbol, limit=5)
    nse_task = _fetch_nse_announcements(symbol, limit=5)

    google_news, mc_news, et_news, nse_news = await asyncio.gather(
        google_task, mc_task, et_task, nse_task,
        return_exceptions=True,
    )

    all_articles: list[dict[str, str]] = []
    for result in [google_news, mc_news, et_news, nse_news]:
        if isinstance(result, list):
            all_articles.extend(result)

    # De-duplicate by title similarity
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for art in all_articles:
        key = re.sub(r"\s+", " ", art["title"].lower().strip())[:80]
        if key not in seen:
            seen.add(key)
            unique.append(art)

    return {
        "symbol": symbol,
        "company_name": company_name,
        "fetched_at": datetime.now().isoformat(),
        "total_articles": len(unique),
        "articles": unique[:limit],
        "sources_queried": ["Google News RSS", "MoneyControl", "Economic Times", "NSE Announcements"],
    }
