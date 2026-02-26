"""Multi-leg order group management with atomic execution and rollback."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from src.data.models import Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OrderGroupStatus(str, Enum):
    """Status of a multi-leg order group."""
    PENDING = "PENDING"       # Waiting for execution
    PARTIAL = "PARTIAL"       # Some legs filled, some pending
    COMPLETE = "COMPLETE"     # All legs filled as specified
    FAILED = "FAILED"         # One or more legs failed/cancelled
    ROLLED = "ROLLED"         # Adjusted/rolled to new strikes


@dataclass
class OrderGroupLeg:
    """Single leg of a multi-leg order group."""
    signal: Signal
    order_id: Optional[str] = None
    filled_quantity: int = 0
    required_quantity: int = 0
    execution_price: float = 0.0
    status: str = "PENDING"
    error: Optional[str] = None


@dataclass
class MultiLegOrderGroup:
    """
    Groups multi-leg orders for atomic execution and validation.
    
    Ensures that either all legs of a strategy are filled or none are,
    preventing orphaned/unhedged positions.
    """
    
    group_id: str = field(default_factory=lambda: str(uuid4()))
    strategy_name: str = ""  # e.g., "iron_condor", "bull_call_credit_spread"
    legs: list[OrderGroupLeg] = field(default_factory=list)
    status: OrderGroupStatus = OrderGroupStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    total_quantity: int = 0  # Qty per leg (e.g., 50 for iron condor)
    
    def add_leg(self, signal: Signal, required_qty: int) -> None:
        """Add a leg to the group."""
        leg = OrderGroupLeg(signal=signal, required_quantity=required_qty)
        self.legs.append(leg)
    
    def validate_fills(self) -> dict[str, bool]:
        """
        Check if all legs are filled as expected.
        
        Returns:
            Dict mapping tradingsymbol to fill status (True = filled, False = not filled)
        """
        validation = {}
        
        for leg in self.legs:
            symbol = leg.signal.tradingsymbol
            
            # Each leg should be filled completely or all fail
            if leg.filled_quantity == 0:
                validation[symbol] = False
                leg.status = "FAILED"
            elif leg.filled_quantity < leg.required_quantity:
                validation[symbol] = False
                leg.status = "PARTIAL"
                leg.error = f"Partial fill: {leg.filled_quantity}/{leg.required_quantity}"
            else:
                validation[symbol] = True
                leg.status = "FILLED"
        
        # All legs must be filled for group to be valid
        all_filled = all(validation.values())
        any_partial = any(leg.filled_quantity > 0 for leg in self.legs)
        
        if all_filled:
            self.status = OrderGroupStatus.COMPLETE
        elif any_partial:
            self.status = OrderGroupStatus.PARTIAL
        else:
            self.status = OrderGroupStatus.FAILED
        
        return validation
    
    def get_mismatch_summary(self) -> dict[str, Any]:
        """Return details of any fills that don't match expected quantities."""
        mismatches = []
        
        for leg in self.legs:
            if leg.filled_quantity != leg.required_quantity:
                variance = leg.required_quantity - leg.filled_quantity
                variance_pct = abs(variance) / leg.required_quantity * 100 if leg.required_quantity > 0 else 0
                
                mismatches.append({
                    "tradingsymbol": leg.signal.tradingsymbol,
                    "expected": leg.required_quantity,
                    "filled": leg.filled_quantity,
                    "variance": variance,
                    "variance_pct": round(variance_pct, 2)
                })
        
        return {
            "group_id": self.group_id,
            "status": self.status.value,
            "mismatches": mismatches,
            "critical": len(mismatches) > 0
        }
    
    async def execute_with_rollback(self, execution_engine) -> bool:
        """
        Execute all legs sequentially but with rollback capability.
        
        If any leg fails or partially fills, cancels all other legs
        to prevent orphaned positions.
        
        Args:
            execution_engine: ExecutionEngine instance
            
        Returns:
            True if all legs filled as expected
            False if any leg failed (all others rolled back)
        """
        placed_orders = []
        
        try:
            logger.info("multi_leg_execution_start",
                       group_id=self.group_id,
                       strategy=self.strategy_name,
                       num_legs=len(self.legs))
            
            # Place all orders
            for i, leg in enumerate(self.legs):
                try:
                    order_id = await execution_engine.execute_signal(leg.signal)
                    leg.order_id = order_id
                    placed_orders.append((leg, order_id))
                    logger.info("leg_order_placed",
                               group_id=self.group_id,
                               leg_index=i,
                               tradingsymbol=leg.signal.tradingsymbol,
                               order_id=order_id)
                except Exception as e:
                    logger.error("leg_order_place_failed",
                                group_id=self.group_id,
                                leg_index=i,
                                tradingsymbol=leg.signal.tradingsymbol,
                                error=str(e))
                    raise
            
            # Query fills after brief delay for matching
            await asyncio.sleep(0.5)
            
            # Check if all legs filled correctly
            for i, leg in enumerate(self.legs):
                try:
                    # Get order status from broker
                    broker_order = await execution_engine._client.get_order(leg.order_id)
                    leg.filled_quantity = broker_order.filled_quantity
                    leg.execution_price = broker_order.average_price
                    
                    logger.info("leg_fill_verified",
                               group_id=self.group_id,
                               leg_index=i,
                               tradingsymbol=leg.signal.tradingsymbol,
                               filled=leg.filled_quantity,
                               required=leg.required_quantity)
                except Exception as e:
                    logger.error("leg_fill_check_failed",
                                group_id=self.group_id,
                                leg_index=i,
                                error=str(e))
                    raise
            
            # Validate all fills
            validation = self.validate_fills()
            
            if not all(validation.values()):
                # ROLLBACK: Cancel all orders
                logger.error("multi_leg_validation_failed",
                           group_id=self.group_id,
                           mismatches=self.get_mismatch_summary())
                
                # Cancel all legs
                for leg in self.legs:
                    if leg.status in ["PENDING", "PARTIAL"]:
                        try:
                            await execution_engine._client.cancel_order(leg.order_id)
                            logger.info("leg_order_cancelled",
                                       group_id=self.group_id,
                                       order_id=leg.order_id)
                        except Exception as e:
                            logger.error("cancel_failed",
                                       group_id=self.group_id,
                                       order_id=leg.order_id,
                                       error=str(e))
                
                return False
            
            logger.info("multi_leg_execution_success",
                       group_id=self.group_id,
                       strategy=self.strategy_name)
            return True
            
        except Exception as e:
            logger.error("multi_leg_execution_error",
                       group_id=self.group_id,
                       strategy=self.strategy_name,
                       error=str(e))
            
            # Emergency rollback: Cancel everything
            for leg, order_id in placed_orders:
                try:
                    await execution_engine._client.cancel_order(order_id)
                except Exception as cancel_error:
                    logger.error("emergency_cancel_failed",
                               order_id=order_id,
                               error=str(cancel_error))
            
            return False
