"""Position validation for multi-leg strategies (Fix #3)."""

import asyncio
from typing import Any, Optional

from src.data.models import TransactionType
from src.data.position_tracker import PositionTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PositionValidator:
    """Validates that multi-leg positions are properly hedged."""
    
    def __init__(self, tracker: PositionTracker):
        self.tracker = tracker
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def validate_multi_leg_hedge(self, strategy_name: str) -> dict[str, Any]:
        """
        Validate that multi-leg positions are properly hedged.
        
        Checks that for each symbol in a strategy:
        - Buyers and sellers are balanced (same quantity)
        - No single unhedged leg
        
        Args:
            strategy_name: Name of the strategy to validate
            
        Returns:
            Dict with validation results and issues found
        """
        positions = self.tracker.get_strategy_positions(strategy_name)
        
        if not positions:
            return {
                "strategy": strategy_name,
                "validated": True,
                "issues": [],
            }
        
        # Group by underlying symbol
        by_symbol: dict[str, list] = {}
        for pos in positions:
            symbol = pos.tradingsymbol
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(pos)
        
        issues = []
        
        # Check each symbol for hedge ratio
        if strategy_name.endswith("_credit_spread") or "condor" in strategy_name:
            for symbol, symbol_positions in by_symbol.items():
                
                # For spreads: should have buy + sell at different strikes
                buyers = [p for p in symbol_positions if p.transaction_type == TransactionType.BUY]
                sellers = [p for p in symbol_positions if p.transaction_type == TransactionType.SELL]
                
                if not buyers or not sellers:
                    issues.append({
                        "symbol": symbol,
                        "problem": "Missing hedge leg (buyer or seller)",
                        "severity": "CRITICAL",
                        "buyers": len(buyers),
                        "sellers": len(sellers),
                    })
                    continue
                
                buyer_qty = sum(p.quantity for p in buyers)
                seller_qty = sum(p.quantity for p in sellers)
                
                if buyer_qty != seller_qty:
                    variance = seller_qty - buyer_qty
                    variance_pct = abs(variance) / seller_qty * 100 if seller_qty > 0 else 0
                    
                    severity = "CRITICAL" if variance_pct > 10 else "WARNING"
                    
                    issues.append({
                        "symbol": symbol,
                        "problem": f"Mismatched hedge: {seller_qty} short, {buyer_qty} long",
                        "variance_qty": variance,
                        "variance_pct": round(variance_pct, 2),
                        "severity": severity,
                    })
        
        return {
            "strategy": strategy_name,
            "validated": len(issues) == 0,
            "issues": issues,
        }
    
    async def monitor_continuous(self, check_interval: int = 30):
        """
        Run continuous validation check every N seconds.
        
        Args:
            check_interval: Seconds between validation checks
        """
        self._monitoring = True
        logger.info("position_monitor_started", interval=check_interval)
        
        try:
            while self._monitoring:
                # Check common multi-leg strategies
                strategies = [
                    "iron_condor",
                    "bull_call_credit_spread",
                    "bear_put_credit_spread",
                ]
                
                for strategy in strategies:
                    result = self.validate_multi_leg_hedge(strategy)
                    
                    if result["issues"]:
                        logger.warning(
                            "position_validation_warning",
                            strategy=strategy,
                            issues=result["issues"],
                        )
                        
                        # Alert risk team (would integrate with notification system)
                        await self._notify_risk_team(result)
                
                await asyncio.sleep(check_interval)
        
        except asyncio.CancelledError:
            logger.info("position_monitor_stopped")
        except Exception as e:
            logger.error("position_monitor_error", error=str(e))
            # Restart after delay
            await asyncio.sleep(60)
            self._monitor_task = asyncio.create_task(self.monitor_continuous(check_interval))
    
    async def _notify_risk_team(self, result: dict[str, Any]) -> None:
        """
        Notify risk team of validation issues.
        
        This is a placeholder for integration with:
        - Email alerts
        - Slack notifications
        - Dashboard alerts
        - Log files
        """
        if not result["issues"]:
            return
        
        severity_map = {"CRITICAL": 0, "WARNING": 1}
        issues_by_severity = {}
        
        for issue in result["issues"]:
            severity = issue.get("severity", "WARNING")
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)
        
        for severity in sorted(issues_by_severity.keys(), key=lambda x: severity_map.get(x, 999)):
            logger.error(
                "position_validation_failed",
                strategy=result["strategy"],
                severity=severity,
                issue_count=len(issues_by_severity[severity]),
                issues=issues_by_severity[severity],
            )
    
    def start_monitoring(self, check_interval: int = 30) -> None:
        """Start the continuous monitoring task."""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(
                self.monitor_continuous(check_interval)
            )
    
    def stop_monitoring(self) -> None:
        """Stop the continuous monitoring task."""
        self._monitoring = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()


def create_position_validator(tracker: PositionTracker) -> PositionValidator:
    """Factory function to create a position validator."""
    return PositionValidator(tracker)
