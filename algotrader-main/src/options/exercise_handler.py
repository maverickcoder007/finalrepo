"""Exercise and assignment handling for options (Fix #4)."""

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional
from uuid import uuid4

from src.api.client import KiteClient
from src.data.models import TransactionType
from src.data.position_tracker import PositionTracker
from src.data.journal import TradeJournal
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExerciseEvent:
    """Notification that an option was exercised or assigned."""
    tradingsymbol: str
    option_type: str  # "CE" or "PE"
    strike: float
    quantity: int
    exercise_type: str  # "EXERCISE" or "ASSIGNMENT"
    underlying_created: bool  # If true, creates equity position
    settlement_date: date
    current_price: float


class ExerciseHandler:
    """Monitors and handles options exercises and assignments at expiry."""
    
    def __init__(
        self,
        client: KiteClient,
        tracker: PositionTracker,
        journal: TradeJournal,
    ):
        self.client = client
        self.tracker = tracker
        self.journal = journal
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def _is_option(self, tradingsymbol: str) -> bool:
        """Check if symbol is an option contract."""
        return tradingsymbol.endswith("CE") or tradingsymbol.endswith("PE")
    
    def _is_expiry_today(self, tradingsymbol: str) -> bool:
        """Check if option expires today."""
        # Extract expiry date from tradingsymbol (typically YYYYMMDD format before CE/PE)
        try:
            # Example: "20000CE21FEB"
            parts = tradingsymbol.split()
            if len(parts) >= 1:
                symbol_part = parts[0]
                # Find the CE or PE
                if "CE" in symbol_part:
                    expiry_part = symbol_part.split("CE")[-1]
                elif "PE" in symbol_part:
                    expiry_part = symbol_part.split("PE")[-1]
                else:
                    return False
                
                # Check if it matches today's date
                # This is simplified; real implementation would parse actual expiry
                today = datetime.now().date()
                # For now, return False as we need to implement proper date parsing
                return False
        except Exception as e:
            logger.error("expiry_check_error", symbol=tradingsymbol, error=str(e))
            return False
    
    def _is_itm(self, position_data: dict[str, Any], current_price: float, strike: float, option_type: str) -> bool:
        """Check if option is in-the-money."""
        if option_type == "CE":  # Call option
            return current_price > strike
        else:  # Put option (PE)
            return current_price < strike
    
    def _get_underlying(self, option_symbol: str) -> str:
        """Extract underlying symbol from option symbol."""
        # Example: "NIFTY" from "20000CE" or "BANKNIFTY" from "45000PE"
        # For simplicity, remove CE/PE suffixes
        return option_symbol.replace("CE", "").replace("PE", "").split()[0]
    
    async def monitor_exercises(self, check_interval: int = 60):
        """
        Check for ITM positions at expiry.
        
        Args:
            check_interval: Seconds between checks
        """
        self._monitoring = True
        logger.info("exercise_monitor_started", interval=check_interval)
        
        try:
            while self._monitoring:
                # Check positions for options expiring today
                positions = self.tracker.get_positions()
                
                for pos in positions:
                    if self._is_option(pos.tradingsymbol) and self._is_expiry_today(pos.tradingsymbol):
                        try:
                            # Get current quote
                            quote = await self.client.get_ltp(pos.tradingsymbol)
                            current_price = quote.last_price if hasattr(quote, 'last_price') else pos.current_price
                            
                            # Check if ITM
                            option_type = "CE" if pos.tradingsymbol.endswith("CE") else "PE"
                            
                            # Parse strike from tradingsymbol
                            strike = self._parse_strike(pos.tradingsymbol)
                            if strike is None:
                                continue
                            
                            if self._is_itm({"price": current_price}, current_price, strike, option_type):
                                # Option will be exercised
                                exercise = ExerciseEvent(
                                    tradingsymbol=pos.tradingsymbol,
                                    option_type=option_type,
                                    strike=strike,
                                    quantity=pos.quantity,
                                    exercise_type="EXERCISE" if pos.transaction_type == TransactionType.BUY else "ASSIGNMENT",
                                    underlying_created=True,
                                    settlement_date=datetime.now().date(),
                                    current_price=current_price,
                                )
                                
                                await self._handle_exercise(exercise)
                        except Exception as e:
                            logger.error("exercise_check_error", symbol=pos.tradingsymbol, error=str(e))
                
                await asyncio.sleep(check_interval)
        
        except asyncio.CancelledError:
            logger.info("exercise_monitor_stopped")
        except Exception as e:
            logger.error("exercise_monitor_error", error=str(e))
            await asyncio.sleep(60)
            self._monitor_task = asyncio.create_task(self.monitor_exercises(check_interval))
    
    def _parse_strike(self, tradingsymbol: str) -> Optional[float]:
        """Parse strike price from option symbol."""
        try:
            # Example: "20000CE21FEB" -> 20000
            if "CE" in tradingsymbol:
                parts = tradingsymbol.split("CE")
                strike_str = parts[0].strip()
            elif "PE" in tradingsymbol:
                parts = tradingsymbol.split("PE")
                strike_str = parts[0].strip()
            else:
                return None
            
            return float(strike_str)
        except Exception as e:
            logger.error("strike_parse_error", symbol=tradingsymbol, error=str(e))
            return None
    
    async def _handle_exercise(self, exercise: ExerciseEvent) -> None:
        """
        Process an exercise/assignment event.
        
        For EXERCISE (long calls/puts exercised):
        - Buy call exercised -> Creates LONG underlying position
        - Buy put exercised -> Creates SHORT underlying position
        
        For ASSIGNMENT (short calls/puts assigned):
        - Short call assigned -> Creates SHORT underlying position
        - Short put assigned -> Creates LONG underlying position
        """
        
        logger.info("exercise_event_processing",
                   symbol=exercise.tradingsymbol,
                   exercise_type=exercise.exercise_type,
                   quantity=exercise.quantity)
        
        # Record the exercise in journal
        self.journal.record_trade(
            strategy="exercise",
            tradingsymbol=exercise.tradingsymbol,
            transaction_type="EXERCISE",
            quantity=exercise.quantity,
            price=exercise.strike,
            order_id=str(uuid4()),
            status="EXERCISE",
            metadata={
                "exercise_date": exercise.settlement_date.isoformat(),
                "exercise_type": exercise.exercise_type,
                "underlying_created": exercise.underlying_created,
            }
        )
        
        # Create underlying position if exercise creates one
        if exercise.underlying_created:
            underlying_symbol = self._get_underlying(exercise.tradingsymbol)
            
            # Determine transaction type based on option type and exercise type
            if exercise.option_type == "CE":
                # Call option
                if exercise.exercise_type == "EXERCISE":
                    underlying_tx = TransactionType.BUY
                else:  # ASSIGNMENT
                    underlying_tx = TransactionType.SELL
            else:  # PE (Put option)
                # Put option
                if exercise.exercise_type == "EXERCISE":
                    underlying_tx = TransactionType.SELL
                else:  # ASSIGNMENT
                    underlying_tx = TransactionType.BUY
            
            # Add underlying position to tracker
            self.tracker.add_position(
                tradingsymbol=underlying_symbol,
                strategy_name="exercise_settlement",
                transaction_type=underlying_tx,
                entry_price=exercise.strike,
                quantity=exercise.quantity,
                signal_id=str(uuid4()),
                metadata={
                    "exercise": True,
                    "original_option": exercise.tradingsymbol,
                    "exercise_date": exercise.settlement_date.isoformat(),
                    "exercise_type": exercise.exercise_type,
                }
            )
            
            logger.info("underlying_position_created",
                       underlying=underlying_symbol,
                       transaction_type=underlying_tx.value,
                       quantity=exercise.quantity,
                       original_option=exercise.tradingsymbol)
    
    def start_monitoring(self, check_interval: int = 60) -> None:
        """Start the continuous monitoring task."""
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(
                self.monitor_exercises(check_interval)
            )
    
    def stop_monitoring(self) -> None:
        """Stop the continuous monitoring task."""
        self._monitoring = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()


def create_exercise_handler(
    client: KiteClient,
    tracker: PositionTracker,
    journal: TradeJournal,
) -> ExerciseHandler:
    """Factory function to create an exercise handler."""
    return ExerciseHandler(client, tracker, journal)
