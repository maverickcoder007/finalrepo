"""
Trade Execution Analyzer — AI-Powered Critical Analysis
========================================================

Analyzes order execution quality against underlying price movement.
Uses OpenRouter API to provide critical AI-powered analysis of:
- Slippage and fill quality
- Entry/exit timing vs underlying movement
- Missed opportunities (MAE/MFE analysis)
- Cost efficiency
- Strategic execution issues

Requires OPENROUTER_API_KEY environment variable or config setting.
"""

from __future__ import annotations
import asyncio
import httpx
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional, List, Dict

from src.utils.config import get_settings
from src.journal.journal_store import JournalStore
from src.journal.journal_models import JournalEntry

logger = logging.getLogger("trade_execution_analyzer")


class TradeExecutionAnalyzer:
    """
    AI-powered trade execution analysis engine.
    
    Compares actual execution vs underlying movement and uses LLM
    to provide critical analysis of execution quality.
    """
    
    ANALYSIS_PROMPT_TEMPLATE = """You are an expert algorithmic trading analyst specializing in execution quality analysis for Indian F&O and equity markets.

Your task is to critically analyze whether the CORRECT TRADES were executed based on the ORDER PLAN (analysis signals, triggers, recommendations).

## CRITICAL ANALYSIS FRAMEWORK:

### 1. **Order Plan Compliance** ⭐ MOST IMPORTANT
   - Were the right symbols traded? Compare executed trades vs signals in the analysis
   - Were signals from analysis acted upon correctly?
   - Were any high-quality signals MISSED (should have traded but didn't)?
   - Were any BAD trades taken (traded without valid signal)?
   - Signal-to-execution latency: How long between signal and trade?

### 2. **Execution Quality Assessment**
   - Slippage: Compare order price vs actual fill price
   - Fill timing: Was the entry/exit well-timed relative to underlying movement?
   - Order type appropriateness: Should limit/market orders have been used differently?

### 3. **Underlying Movement Analysis**
   - How did the underlying move during the trade window?
   - Was the entry at a local high/low (good/bad depending on direction)?
   - Did the exit capture available movement or leave money on table?

### 4. **MAE/MFE Critical Review**
   - MAE (Max Adverse Excursion): How much drawdown did the position suffer?
   - MFE (Max Favorable Excursion): How much unrealized profit was available?
   - Edge Ratio (MFE/MAE): Quality of trade management

### 5. **Missed Opportunities from Analysis**
   - List symbols from analysis with strong signals that were NOT traded
   - Estimate potential profit missed on these opportunities
   - Were there legitimate reasons to skip these trades?

### 6. **Incorrect Trades Taken**
   - List trades that don't have corresponding signals in analysis
   - Why were these trades taken? Were they discretionary overrides?
   - Did these unsignaled trades perform well or poorly?

### 7. **Actionable Recommendations**
   - Specific improvements for signal-to-execution workflow
   - Which signal types to prioritize
   - Position sizing relative to signal strength
   - Timing improvements

Be brutally honest and critical. Focus on whether the ORDER PLAN was followed.
Use specific numbers from the data provided.
Format response as structured analysis with clear sections.

---

ORDER PLAN / ANALYSIS SIGNALS:
{analysis_data}

---

EXECUTED TRADES:
{trade_data}

---

UNDERLYING PRICE MOVEMENT:
{underlying_data}

---

Provide your critical analysis focusing on ORDER PLAN COMPLIANCE:"""

    def __init__(self, journal_store: Optional[JournalStore] = None):
        self._settings = get_settings()
        self._journal = journal_store or JournalStore()
        self._api_key = self._settings.openrouter_api_key
        self._model = self._settings.openrouter_model
        self._base_url = self._settings.openrouter_base_url
        # Hugging Face fallback
        self._hf_api_key = self._settings.huggingface_api_key
        self._hf_model = self._settings.huggingface_model
        self._hf_base_url = self._settings.huggingface_base_url
        
    async def analyze_trades(
        self,
        trade_ids: Optional[List[str]] = None,
        strategy: str = "",
        instrument: str = "",
        from_date: str = "",
        to_date: str = "",
        limit: int = 20,
        underlying_prices: Optional[Dict[str, List[Dict]]] = None,
        analysis_context: Optional[Dict[str, Any]] = None,
        filter_by_analysis: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyze trade executions with AI against order plan from analysis.
        
        Args:
            trade_ids: Specific trade IDs to analyze
            strategy: Filter by strategy name
            instrument: Filter by instrument (e.g., "NIFTY", "BANKNIFTY")
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            limit: Max trades to analyze
            underlying_prices: Optional pre-fetched underlying price data
            analysis_context: Scanner results, signals, triggers from Analysis tab
            filter_by_analysis: If True, only analyze symbols present in analysis_context
            
        Returns:
            Dict with analysis results, issues found, and recommendations
        """
        if not self._api_key:
            return {
                "success": False,
                "error": "OpenRouter API key not configured. Set OPENROUTER_API_KEY env var.",
                "analysis": None,
            }
        
        # Fetch trades from journal
        trades = self._fetch_trades(trade_ids, strategy, instrument, from_date, to_date, limit)
        
        # Filter trades to only symbols in analysis context if requested
        if filter_by_analysis and analysis_context:
            analysis_symbols = set(analysis_context.get("results", {}).keys())
            if analysis_symbols:
                trades = [t for t in trades if self._extract_underlying(t.tradingsymbol) in analysis_symbols]
        
        if not trades:
            return {
                "success": False,
                "error": "No trades found matching criteria",
                "analysis": None,
            }
        
        # Prepare analysis context data
        analysis_data = self._prepare_analysis_context(trades, analysis_context)
        
        # Prepare trade data for analysis
        trade_data = self._prepare_trade_data(trades)
        
        # Prepare underlying movement data
        underlying_data = self._prepare_underlying_data(trades, underlying_prices)
        
        # Call OpenRouter API for analysis
        analysis = await self._call_openrouter(trade_data, underlying_data, analysis_data)
        
        # Parse and structure the response
        return {
            "success": True,
            "trades_analyzed": len(trades),
            "trade_ids": [t.trade_id for t in trades],
            "analysis": analysis,
            "summary": self._generate_summary(trades),
            "analysis_symbols": list(analysis_context.get("results", {}).keys()) if analysis_context else [],
            "timestamp": datetime.now().isoformat(),
        }
    
    def _fetch_trades(
        self,
        trade_ids: Optional[List[str]],
        strategy: str,
        instrument: str,
        from_date: str,
        to_date: str,
        limit: int,
    ) -> List[JournalEntry]:
        """Fetch trades from journal based on filters."""
        if trade_ids:
            trades = []
            for tid in trade_ids:
                entry = self._journal.get_entry(tid)
                if entry:
                    trades.append(entry)
            return trades[:limit]
        
        # Query with filters
        entries = self._journal.query_entries(
            strategy=strategy,
            instrument=instrument,
            from_date=from_date,
            to_date=to_date,
            is_closed=True,  # Only analyze closed trades
            limit=limit,
        )
        return entries
    
    def _prepare_trade_data(self, trades: List[JournalEntry]) -> str:
        """Format trade data for LLM analysis."""
        lines = []
        for i, t in enumerate(trades, 1):
            lines.append(f"\n--- Trade #{i} ---")
            lines.append(f"Trade ID: {t.trade_id}")
            lines.append(f"Symbol: {t.tradingsymbol} ({t.exchange})")
            lines.append(f"Strategy: {t.strategy_name}")
            lines.append(f"Direction: {t.direction}")
            lines.append(f"Trade Type: {t.trade_type}")
            
            # Execution details
            if t.execution:
                exec_rec = t.execution if isinstance(t.execution, dict) else t.execution.to_dict()
                lines.append(f"Order Type: {exec_rec.get('order_type', 'N/A')}")
                lines.append(f"Order Price: ₹{exec_rec.get('order_price', 0):.2f}")
                lines.append(f"Fill Price: ₹{exec_rec.get('actual_fill_price', 0):.2f}")
                lines.append(f"Quantity: {exec_rec.get('quantity', 0)}")
                slippage = exec_rec.get('slippage_pct', 0)
                lines.append(f"Slippage: {slippage:.3f}%")
                fill_time = exec_rec.get('fill_time_ms', 0)
                lines.append(f"Fill Time: {fill_time}ms")
            
            # P&L
            lines.append(f"Gross P&L: ₹{t.gross_pnl:.2f}")
            lines.append(f"Net P&L: ₹{t.net_pnl:.2f}")
            lines.append(f"Total Costs: ₹{t.total_costs:.2f}")
            lines.append(f"Return %: {t.return_pct:.2f}%")
            
            # MAE/MFE
            lines.append(f"MAE (Max Drawdown): ₹{t.mae:.2f} ({t.mae_pct:.2f}%)")
            lines.append(f"MFE (Max Profit): ₹{t.mfe:.2f} ({t.mfe_pct:.2f}%)")
            lines.append(f"Edge Ratio (MFE/MAE): {t.edge_ratio:.2f}")
            
            # Timing
            lines.append(f"Entry Time: {t.entry_time}")
            lines.append(f"Exit Time: {t.exit_time}")
            
            # Strategy Context
            if t.strategy_context:
                ctx = t.strategy_context if isinstance(t.strategy_context, dict) else t.strategy_context.to_dict()
                lines.append(f"Signal Type: {ctx.get('signal_type', 'N/A')}")
                lines.append(f"Market Regime: {ctx.get('regime', 'N/A')}")
            
            # Signal Quality
            if t.signal_quality:
                sq = t.signal_quality if isinstance(t.signal_quality, dict) else t.signal_quality.to_dict()
                lines.append(f"Signal Confidence: {sq.get('confidence', 'N/A')}")
        
        return "\n".join(lines)
    
    def _prepare_underlying_data(
        self,
        trades: List[JournalEntry],
        underlying_prices: Optional[Dict[str, List[Dict]]] = None
    ) -> str:
        """Format underlying price movement for LLM analysis."""
        if underlying_prices:
            # Use provided price data
            lines = []
            for symbol, prices in underlying_prices.items():
                lines.append(f"\n{symbol} Price Movement:")
                for p in prices[-50:]:  # Last 50 data points
                    lines.append(f"  {p.get('timestamp', '')}: O={p.get('open', 0):.2f} H={p.get('high', 0):.2f} L={p.get('low', 0):.2f} C={p.get('close', 0):.2f}")
            return "\n".join(lines)
        
        # Generate basic summary from trade entry/exit
        lines = ["(Live underlying price data not available - using trade entry/exit data)"]
        
        for t in trades:
            if t.execution:
                exec_rec = t.execution if isinstance(t.execution, dict) else t.execution.to_dict()
                entry_price = exec_rec.get('actual_fill_price', 0) or exec_rec.get('order_price', 0)
                exit_price = 0
                
                # Try to get exit price from exit execution
                if hasattr(t, 'exit_execution') and t.exit_execution:
                    exit_exec = t.exit_execution if isinstance(t.exit_execution, dict) else t.exit_execution.to_dict()
                    exit_price = exit_exec.get('actual_fill_price', 0)
                elif t.net_pnl and entry_price:
                    # Estimate exit price from P&L
                    qty = exec_rec.get('quantity', 1)
                    if t.direction == "LONG":
                        exit_price = entry_price + (t.gross_pnl / qty) if qty else entry_price
                    else:
                        exit_price = entry_price - (t.gross_pnl / qty) if qty else entry_price
                
                lines.append(f"\n{t.tradingsymbol}:")
                lines.append(f"  Entry: ₹{entry_price:.2f} at {t.entry_time}")
                if exit_price:
                    lines.append(f"  Exit: ₹{exit_price:.2f} at {t.exit_time}")
                    move_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0
                    lines.append(f"  Movement: {move_pct:+.2f}%")
                
                # MAE/MFE gives us intra-trade movement info
                if t.mae > 0:
                    lines.append(f"  Max Adverse: ₹{t.mae:.2f} below entry")
                if t.mfe > 0:
                    lines.append(f"  Max Favorable: ₹{t.mfe:.2f} above entry")
        
        return "\n".join(lines)
    
    def _extract_underlying(self, tradingsymbol: str) -> str:
        """Extract underlying symbol from trading symbol (e.g., NIFTY24MAR23000CE -> NIFTY)."""
        if not tradingsymbol:
            return ""
        # Common F&O underlyings
        for underlying in ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"]:
            if tradingsymbol.startswith(underlying):
                return underlying
        # Equity: symbol is the underlying itself
        return tradingsymbol.split("-")[0].split(":")[0].upper()
    
    def _prepare_analysis_context(
        self,
        trades: List[JournalEntry],
        analysis_context: Optional[Dict[str, Any]]
    ) -> str:
        """Format analysis/scanner results for LLM context."""
        if not analysis_context:
            return "(No Analysis tab data provided - analyzing trades without order plan context)"
        
        lines = []
        results = analysis_context.get("results", {})
        scan_time = analysis_context.get("last_scan_time", "Unknown")
        
        lines.append(f"Scanner Results (Last Scan: {scan_time})")
        lines.append(f"Total Symbols Analyzed: {len(results)}")
        lines.append("")
        
        # Get symbols from trades to focus on relevant analysis
        traded_symbols = set()
        for t in trades:
            underlying = self._extract_underlying(t.tradingsymbol)
            traded_symbols.add(underlying)
            traded_symbols.add(t.tradingsymbol)
        
        # Analyze each result with signals/triggers
        symbols_with_signals = []
        for symbol, data in results.items():
            triggers = data.get("triggers", [])
            super_perf = data.get("super_performance", {})
            trend_score = data.get("trend_score", 0)
            accumulation = data.get("accumulation", {})
            
            # Check if this symbol was traded
            was_traded = symbol in traded_symbols
            
            signal_info = {
                "symbol": symbol,
                "trend_score": trend_score,
                "triggers": triggers,
                "super_performance_passed": sum(1 for v in super_perf.values() if v) if isinstance(super_perf, dict) else 0,
                "super_performance_total": len(super_perf) if isinstance(super_perf, dict) else 0,
                "accumulation_signals": accumulation.get("signals", []) if isinstance(accumulation, dict) else [],
                "was_traded": was_traded,
            }
            
            # Only include symbols with actionable signals OR symbols that were traded
            if triggers or trend_score > 60 or was_traded:
                symbols_with_signals.append(signal_info)
        
        # Sort by trend score descending
        symbols_with_signals.sort(key=lambda x: x["trend_score"], reverse=True)
        
        lines.append("=" * 60)
        lines.append("SYMBOLS WITH SIGNALS (ORDER PLAN / RECOMMENDATIONS):")
        lines.append("=" * 60)
        
        for info in symbols_with_signals:
            traded_marker = "✅ TRADED" if info["was_traded"] else "❌ NOT TRADED"
            lines.append(f"\n{info['symbol']} [{traded_marker}]")
            lines.append(f"  Trend Score: {info['trend_score']}/100")
            lines.append(f"  Super Performance: {info['super_performance_passed']}/{info['super_performance_total']} criteria passed")
            
            if info["triggers"]:
                lines.append(f"  Active Triggers: {', '.join(info['triggers'][:5])}")  # First 5 triggers
            
            if info["accumulation_signals"]:
                lines.append(f"  Accumulation: {', '.join(info['accumulation_signals'][:3])}")
        
        # Summary: Missed opportunities
        missed = [s for s in symbols_with_signals if not s["was_traded"] and s["trend_score"] > 70]
        if missed:
            lines.append("\n" + "=" * 60)
            lines.append("⚠️ POTENTIAL MISSED OPPORTUNITIES (High Score, Not Traded):")
            lines.append("=" * 60)
            for info in missed[:10]:  # Top 10 missed
                lines.append(f"  {info['symbol']}: Score {info['trend_score']}, Triggers: {info['triggers'][:3]}")
        
        # Summary: Traded without strong signal
        traded_without_signal = [s for s in symbols_with_signals if s["was_traded"] and s["trend_score"] < 50 and not s["triggers"]]
        if traded_without_signal:
            lines.append("\n" + "=" * 60)
            lines.append("⚠️ TRADES WITHOUT STRONG ANALYSIS SIGNAL:")
            lines.append("=" * 60)
            for info in traded_without_signal:
                lines.append(f"  {info['symbol']}: Score {info['trend_score']}, No triggers")
        
        return "\n".join(lines)
    
    async def _call_openrouter(self, trade_data: str, underlying_data: str, analysis_data: str = "") -> str:
        """Call OpenRouter API for AI analysis with retry logic for rate limits."""
        prompt = self.ANALYSIS_PROMPT_TEMPLATE.format(
            analysis_data=analysis_data or "(No analysis context provided)",
            trade_data=trade_data,
            underlying_data=underlying_data
        )
        
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://algotrader.local",
            "X-Title": "AlgoTrader Execution Analyzer",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert algorithmic trading analyst. Provide detailed, critical analysis of trade execution quality."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 4000,
        }
        
        max_retries = 3
        base_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    # Handle rate limiting with retry
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", base_delay * (2 ** attempt)))
                        logger.warning(f"OpenRouter rate limited (429). Retry {attempt + 1}/{max_retries} after {retry_after}s")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            # All retries exhausted, break to fallback
                            logger.warning("OpenRouter rate limit retries exhausted, will try fallback")
                            break
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"Unexpected OpenRouter response: {result}")
                        break  # Try fallback
                        
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"OpenRouter rate limited. Retry {attempt + 1}/{max_retries} after {delay}s")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"OpenRouter HTTP error: {e.response.status_code} - {e.response.text}")
                break  # Try fallback instead of returning error
            except Exception as e:
                logger.error(f"OpenRouter call failed: {e}")
                break  # Try fallback
        
        # All OpenRouter retries failed, try Hugging Face fallback
        if self._hf_api_key:
            logger.info("OpenRouter failed, attempting Hugging Face fallback...")
            return await self._call_huggingface(trade_data, underlying_data, analysis_data)
        
        return "Error: Max retries exceeded for OpenRouter API (no fallback configured)"
    
    async def _call_huggingface(self, trade_data: str, underlying_data: str, analysis_data: str = "") -> str:
        """Call Hugging Face Inference API as fallback."""
        prompt = self.ANALYSIS_PROMPT_TEMPLATE.format(
            analysis_data=analysis_data or "(No analysis context provided)",
            trade_data=trade_data,
            underlying_data=underlying_data
        )
        
        # Format for instruction-tuned Zephyr model
        formatted_prompt = f"""<|system|>
You are an expert algorithmic trading analyst. Provide detailed, critical analysis of trade execution quality.</s>
<|user|>
{prompt}</s>
<|assistant|>
"""
        
        headers = {
            "Authorization": f"Bearer {self._hf_api_key}",
            "Content-Type": "application/json",
        }
        
        # HF text-generation format
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 2000,
                "temperature": 0.3,
                "do_sample": True,
                "return_full_text": False,
            },
        }
        
        max_retries = 3
        base_delay = 10  # HF models may need loading time
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        f"{self._hf_base_url}/{self._hf_model}",
                        headers=headers,
                        json=payload
                    )
                    
                    # Handle model loading (503)
                    if response.status_code == 503:
                        data = response.json()
                        wait_time = data.get("estimated_time", base_delay * (attempt + 1))
                        logger.warning(f"HuggingFace model loading. Retry {attempt + 1}/{max_retries} after {wait_time:.0f}s")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(min(wait_time, 60))  # Cap at 60s
                            continue
                    
                    # Handle rate limiting
                    if response.status_code == 429:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"HuggingFace rate limited. Retry {attempt + 1}/{max_retries} after {delay}s")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(delay)
                            continue
                    
                    response.raise_for_status()
                    
                    result = response.json()
                    # Text generation format returns list
                    if isinstance(result, list) and result:
                        return result[0].get("generated_text", "Error: Empty response from HuggingFace")
                    elif isinstance(result, dict):
                        # OpenAI-compatible format
                        if "choices" in result and result["choices"]:
                            return result["choices"][0]["message"]["content"]
                        return result.get("generated_text", str(result))
                    else:
                        return str(result)
                        
            except httpx.HTTPStatusError as e:
                logger.error(f"HuggingFace HTTP error: {e.response.status_code} - {e.response.text}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (attempt + 1))
                    continue
                return f"Error: HuggingFace API returned {e.response.status_code}"
            except Exception as e:
                logger.error(f"HuggingFace call failed: {e}")
                return f"Error calling HuggingFace: {str(e)}"
        
        return "Error: Max retries exceeded for HuggingFace API"
    
    def _generate_summary(self, trades: List[JournalEntry]) -> Dict[str, Any]:
        """Generate statistical summary of trades."""
        if not trades:
            return {}
        
        total_gross = sum(t.gross_pnl for t in trades)
        total_net = sum(t.net_pnl for t in trades)
        total_costs = sum(t.total_costs for t in trades)
        
        winners = [t for t in trades if t.net_pnl > 0]
        losers = [t for t in trades if t.net_pnl < 0]
        
        avg_mae = sum(t.mae for t in trades) / len(trades) if trades else 0
        avg_mfe = sum(t.mfe for t in trades) / len(trades) if trades else 0
        
        # Calculate execution quality metrics
        slippages = []
        for t in trades:
            if t.execution:
                exec_rec = t.execution if isinstance(t.execution, dict) else t.execution.to_dict()
                slip = exec_rec.get('slippage_pct', 0)
                if slip:
                    slippages.append(slip)
        
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0
        
        return {
            "total_trades": len(trades),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": len(winners) / len(trades) * 100 if trades else 0,
            "gross_pnl": total_gross,
            "net_pnl": total_net,
            "total_costs": total_costs,
            "cost_ratio": (total_costs / abs(total_gross) * 100) if total_gross else 0,
            "avg_mae": avg_mae,
            "avg_mfe": avg_mfe,
            "avg_edge_ratio": avg_mfe / avg_mae if avg_mae > 0 else 0,
            "avg_slippage_pct": avg_slippage,
            "strategies": list(set(t.strategy_name for t in trades)),
            "instruments": list(set(t.tradingsymbol for t in trades)),
        }
    
    async def analyze_single_order(
        self,
        order_id: str,
        order_data: Dict[str, Any],
        underlying_candles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze a single order's execution quality in real-time.
        
        Used for immediate feedback after order execution.
        """
        if not self._api_key:
            return {
                "success": False,
                "error": "OpenRouter API key not configured",
            }
        
        # Format order data
        order_lines = [
            f"Order ID: {order_id}",
            f"Symbol: {order_data.get('tradingsymbol', 'N/A')}",
            f"Direction: {order_data.get('transaction_type', 'N/A')}",
            f"Order Type: {order_data.get('order_type', 'N/A')}",
            f"Order Price: ₹{order_data.get('price', 0):.2f}",
            f"Trigger Price: ₹{order_data.get('trigger_price', 0):.2f}",
            f"Fill Price: ₹{order_data.get('average_price', 0):.2f}",
            f"Quantity: {order_data.get('quantity', 0)}",
            f"Status: {order_data.get('status', 'N/A')}",
            f"Order Time: {order_data.get('order_timestamp', 'N/A')}",
            f"Fill Time: {order_data.get('exchange_timestamp', 'N/A')}",
        ]
        
        # Calculate slippage
        order_price = order_data.get('price', 0) or order_data.get('trigger_price', 0)
        fill_price = order_data.get('average_price', 0)
        if order_price and fill_price:
            slippage_pct = ((fill_price - order_price) / order_price * 100)
            order_lines.append(f"Slippage: {slippage_pct:+.3f}%")
        
        trade_data = "\n".join(order_lines)
        
        # Format underlying candles
        underlying_lines = ["Underlying 1-minute candles around order:"]
        for c in underlying_candles[-15:]:  # Last 15 candles
            underlying_lines.append(
                f"  {c.get('timestamp', '')}: O={c.get('open', 0):.2f} H={c.get('high', 0):.2f} "
                f"L={c.get('low', 0):.2f} C={c.get('close', 0):.2f} V={c.get('volume', 0)}"
            )
        underlying_data = "\n".join(underlying_lines)
        
        analysis = await self._call_openrouter(trade_data, underlying_data)
        
        return {
            "success": True,
            "order_id": order_id,
            "analysis": analysis,
            "slippage_pct": slippage_pct if order_price and fill_price else None,
            "timestamp": datetime.now().isoformat(),
        }


# Singleton for easy access
_analyzer: Optional[TradeExecutionAnalyzer] = None


def get_analyzer() -> TradeExecutionAnalyzer:
    """Get or create the trade execution analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = TradeExecutionAnalyzer()
    return _analyzer
