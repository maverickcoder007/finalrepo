"""
Annual Report Fetcher — Retrieves annual report links and key financial
highlights from BSE India, NSE India, and public company investor-relations
pages for Indian equities.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import Any

import httpx
from bs4 import BeautifulSoup


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
}

# ──────────────────────────────────────────────────────────────
# BSE India — Annual Reports
# ──────────────────────────────────────────────────────────────

# Map of popular NSE symbols to BSE scrip codes for API lookup
_BSE_SCRIP_CODES: dict[str, str] = {
    "RELIANCE": "500325", "TCS": "532540", "INFY": "500209",
    "HDFCBANK": "500180", "ICICIBANK": "532174", "HINDUNILVR": "500696",
    "SBIN": "500112", "BHARTIARTL": "532454", "ITC": "500875",
    "KOTAKBANK": "500247", "LT": "500510", "HCLTECH": "532281",
    "AXISBANK": "532215", "ASIANPAINT": "500820", "MARUTI": "532500",
    "SUNPHARMA": "524715", "TITAN": "500114", "BAJFINANCE": "500034",
    "WIPRO": "507685", "ULTRACEMCO": "532538", "NESTLEIND": "500790",
    "TATAMOTORS": "500570", "TATASTEEL": "500470", "POWERGRID": "532898",
    "NTPC": "532555", "JSWSTEEL": "500228", "M&M": "500520",
    "ADANIENT": "512599", "ADANIPORTS": "532921", "COALINDIA": "533278",
    "ONGC": "500312", "DRREDDY": "500124", "CIPLA": "500087",
    "GRASIM": "500300", "TECHM": "532755", "APOLLOHOSP": "508869",
    "DIVISLAB": "532488", "EICHERMOT": "505200", "HEROMOTOCO": "500182",
    "BAJAJ-AUTO": "532977", "BRITANNIA": "500825", "INDUSINDBK": "532187",
    "HDFCLIFE": "540777", "SBILIFE": "540719", "DABUR": "500096",
    "GODREJCP": "532424", "PIDILITIND": "500331", "HAVELLS": "517354",
    "DLF": "532868", "VEDL": "500295", "TATAPOWER": "500400",
}


async def _fetch_bse_annual_reports(symbol: str, limit: int = 5) -> list[dict[str, str]]:
    """Fetch annual report PDFs from BSE India corporate filings."""
    scrip_code = _BSE_SCRIP_CODES.get(symbol.upper())
    if not scrip_code:
        return []

    url = (
        f"https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"
        f"?scripcode={scrip_code}&strType=C"
    )
    reports: list[dict[str, str]] = []

    async with httpx.AsyncClient(headers=_HEADERS, timeout=20, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return reports
            data = resp.json()
            if isinstance(data, list):
                for item in data[:limit]:
                    reports.append({
                        "title": item.get("SLONGNAME", f"{symbol} Annual Report"),
                        "year": item.get("Year", ""),
                        "type": "Annual Report",
                        "url": item.get("ATTACHMENTNAME", ""),
                        "source": "BSE India",
                    })
        except Exception:
            pass

    return reports


# ──────────────────────────────────────────────────────────────
# NSE India — Corporate Filings (Annual Reports)
# ──────────────────────────────────────────────────────────────

async def _fetch_nse_annual_reports(symbol: str, limit: int = 5) -> list[dict[str, str]]:
    """Fetch annual reports from NSE India corporate filings API."""
    url = (
        f"https://www.nseindia.com/api/annual-reports?index=equities&symbol={symbol}"
    )
    reports: list[dict[str, str]] = []

    nse_headers = {
        **_HEADERS,
        "Referer": "https://www.nseindia.com/",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(headers=nse_headers, timeout=20, follow_redirects=True) as client:
        try:
            # Get session cookies
            await client.get("https://www.nseindia.com/")
            resp = await client.get(url)
            if resp.status_code != 200:
                return reports
            data = resp.json()
            if isinstance(data, list):
                for item in data[:limit]:
                    reports.append({
                        "title": item.get("companyName", symbol),
                        "year": item.get("yearEnding", ""),
                        "type": "Annual Report",
                        "url": item.get("fileName", ""),
                        "source": "NSE India",
                    })
        except Exception:
            pass

    return reports


# ──────────────────────────────────────────────────────────────
# Google Search — Annual Report PDF links
# ──────────────────────────────────────────────────────────────

async def _search_annual_report_links(
    symbol: str, company_name: str, limit: int = 5
) -> list[dict[str, str]]:
    """
    Search Google for annual report PDFs for a company.
    Falls back to scraping search results.
    """
    query = f"{company_name} annual report {datetime.now().year} filetype:pdf site:bseindia.com OR site:nseindia.com OR site:{symbol.lower()}.com"
    url = f"https://www.google.com/search?q={query}&num={limit}"
    reports: list[dict[str, str]] = []

    async with httpx.AsyncClient(headers=_HEADERS, timeout=15, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return reports
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.select("a[href]"):
                href = a.get("href", "")
                # Extract actual URL from Google redirect
                match = re.search(r"/url\?q=(https?://[^&]+)", href)
                if match:
                    real_url = match.group(1)
                    if ".pdf" in real_url.lower() or "annual" in real_url.lower():
                        text = a.get_text(strip=True)[:200]
                        reports.append({
                            "title": text or f"{symbol} Annual Report",
                            "year": str(datetime.now().year),
                            "type": "Annual Report (Search)",
                            "url": real_url,
                            "source": "Google Search",
                        })
                        if len(reports) >= limit:
                            break
        except Exception:
            pass

    return reports


# ──────────────────────────────────────────────────────────────
# Screener.in — Financial Highlights
# ──────────────────────────────────────────────────────────────

async def _fetch_screener_financials(symbol: str) -> dict[str, Any]:
    """Scrape key financial highlights from Screener.in (public page)."""
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    financials: dict[str, Any] = {}

    async with httpx.AsyncClient(headers=_HEADERS, timeout=15, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                # Try standalone
                resp = await client.get(f"https://www.screener.in/company/{symbol}/")
                if resp.status_code != 200:
                    return financials

            soup = BeautifulSoup(resp.text, "html.parser")

            # Extract key ratios from the top section
            ratios_section = soup.select("#top-ratios li")
            for li in ratios_section:
                name_el = li.select_one(".name")
                value_el = li.select_one(".value, .number")
                if name_el and value_el:
                    name = name_el.get_text(strip=True)
                    value = value_el.get_text(strip=True)
                    financials[name] = value

            # Extract pros/cons
            pros = [li.get_text(strip=True) for li in soup.select(".pros li")]
            cons = [li.get_text(strip=True) for li in soup.select(".cons li")]
            if pros:
                financials["pros"] = pros
            if cons:
                financials["cons"] = cons

            # Company name
            title_el = soup.select_one("h1")
            if title_el:
                financials["company_name"] = title_el.get_text(strip=True)

        except Exception:
            pass

    return financials


# ──────────────────────────────────────────────────────────────
# TrendLyne — Annual Report Links
# ──────────────────────────────────────────────────────────────

async def _fetch_trendlyne_reports(symbol: str, limit: int = 5) -> list[dict[str, str]]:
    """Fetch annual report links from Trendlyne."""
    url = f"https://trendlyne.com/equity/{symbol}/latest-annual-reports/"
    reports: list[dict[str, str]] = []

    async with httpx.AsyncClient(headers=_HEADERS, timeout=15, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return reports
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.select("a[href*='.pdf']"):
                text = a.get_text(strip=True)
                href = a.get("href", "")
                if text and href:
                    reports.append({
                        "title": text,
                        "year": "",
                        "type": "Annual Report",
                        "url": href if href.startswith("http") else f"https://trendlyne.com{href}",
                        "source": "TrendLyne",
                    })
                    if len(reports) >= limit:
                        break
        except Exception:
            pass

    return reports


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

_COMPANY_NAMES: dict[str, str] = {
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services",
    "INFY": "Infosys", "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank",
    "HINDUNILVR": "Hindustan Unilever", "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel", "ITC": "ITC Limited",
    "KOTAKBANK": "Kotak Mahindra Bank", "LT": "Larsen & Toubro",
    "HCLTECH": "HCL Technologies", "AXISBANK": "Axis Bank",
    "ASIANPAINT": "Asian Paints", "MARUTI": "Maruti Suzuki",
    "SUNPHARMA": "Sun Pharma", "TITAN": "Titan Company",
    "BAJFINANCE": "Bajaj Finance", "WIPRO": "Wipro",
    "TATAMOTORS": "Tata Motors", "TATASTEEL": "Tata Steel",
    "ADANIENT": "Adani Enterprises", "ADANIPORTS": "Adani Ports",
    "DRREDDY": "Dr Reddy's Laboratories", "CIPLA": "Cipla",
}


async def fetch_annual_reports(symbol: str, limit: int = 10) -> dict[str, Any]:
    """
    Aggregate annual report links and financial highlights from multiple
    public sources for a given NSE stock symbol.
    """
    symbol = symbol.upper().strip()
    company_name = _COMPANY_NAMES.get(symbol, symbol)

    # Run all fetchers concurrently
    bse_task = _fetch_bse_annual_reports(symbol, limit=limit)
    nse_task = _fetch_nse_annual_reports(symbol, limit=limit)
    google_task = _search_annual_report_links(symbol, company_name, limit=5)
    trendlyne_task = _fetch_trendlyne_reports(symbol, limit=5)
    screener_task = _fetch_screener_financials(symbol)

    bse, nse, google, trendlyne, screener = await asyncio.gather(
        bse_task, nse_task, google_task, trendlyne_task, screener_task,
        return_exceptions=True,
    )

    all_reports: list[dict[str, str]] = []
    for result in [bse, nse, google, trendlyne]:
        if isinstance(result, list):
            all_reports.extend(result)

    # De-duplicate by URL
    seen_urls: set[str] = set()
    unique_reports: list[dict[str, str]] = []
    for r in all_reports:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_reports.append(r)
        elif not url:
            unique_reports.append(r)

    financial_highlights = screener if isinstance(screener, dict) else {}

    return {
        "symbol": symbol,
        "company_name": company_name,
        "fetched_at": datetime.now().isoformat(),
        "total_reports": len(unique_reports),
        "reports": unique_reports[:limit],
        "financial_highlights": financial_highlights,
        "sources_queried": ["BSE India", "NSE India", "Google Search", "TrendLyne", "Screener.in"],
    }
