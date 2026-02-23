"""Test suite for critical execution fixes."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.api.client import KiteClient
from src.data.models import Signal, Exchange, OrderType, ProductType, TransactionType, Order
from src.data.journal import TradeJournal
from src.data.position_tracker import PositionTracker, get_position_tracker
from src.execution.engine import ExecutionEngine
from src.execution.order_group import MultiLegOrderGroup, OrderGroupStatus
from src.risk.manager import RiskManager
from src.risk.position_validator import PositionValidator
from src.options.exercise_handler import ExerciseHandler


class TestMultiLegOrderGroup:
    """Test Fix #1: Multi-Leg Order Group Management."""
    
    def test_add_legs(self):
        """Test adding legs to order group."""
        group = MultiLegOrderGroup(strategy_name="iron_condor")
        
        signal1 = Signal(
            tradingsymbol="20000CE",
            exchange=Exchange.NFO,
            transaction_type=TransactionType.SELL,
            quantity=50,
        )
        
        group.add_leg(signal1, required_qty=50)
        assert len(group.legs) == 1
        assert group.legs[0].signal.tradingsymbol == "20000CE"
        assert group.legs[0].required_quantity == 50
    
    def test_validate_fills_all_filled(self):
        """Test validation when all legs are filled."""
        group = MultiLegOrderGroup(strategy_name="iron_condor")
        
        signal1 = Signal(
            tradingsymbol="20000CE",
            exchange=Exchange.NFO,
            transaction_type=TransactionType.SELL,
            quantity=50,
        )
        
        group.add_leg(signal1, required_qty=50)
        group.legs[0].filled_quantity = 50
        
        validation = group.validate_fills()
        
        assert validation["20000CE"] is True
        assert group.status == OrderGroupStatus.COMPLETE
    
    def test_validate_fills_partial_fill(self):
        """Test validation when partial fill occurs."""
        group = MultiLegOrderGroup(strategy_name="iron_condor")
        
        signal1 = Signal(
            tradingsymbol="20000CE",
            exchange=Exchange.NFO,
            transaction_type=TransactionType.SELL,
            quantity=50,
        )
        
        group.add_leg(signal1, required_qty=50)
        group.legs[0].filled_quantity = 30  # Only 30 filled of 50
        
        validation = group.validate_fills()
        
        assert validation["20000CE"] is False
        assert group.status == OrderGroupStatus.PARTIAL
    
    def test_mismatch_summary(self):
        """Test getting mismatch details."""
        group = MultiLegOrderGroup(strategy_name="iron_condor")
        
        signal1 = Signal(
            tradingsymbol="20000CE",
            exchange=Exchange.NFO,
            transaction_type=TransactionType.SELL,
            quantity=50,
        )
        
        group.add_leg(signal1, required_qty=50)
        group.legs[0].filled_quantity = 30
        
        summary = group.get_mismatch_summary()
        
        assert summary["critical"] is True
        assert len(summary["mismatches"]) == 1
        assert summary["mismatches"][0]["variance"] == 20
        assert summary["mismatches"][0]["variance_pct"] == 40.0
    
    @pytest.mark.asyncio
    async def test_execute_with_rollback_success(self):
        """Test successful multi-leg execution."""
        # Mock objects
        mock_client = AsyncMock(spec=KiteClient)
        mock_risk_manager = MagicMock(spec=RiskManager)
        mock_risk_manager.validate_signal.return_value = (True, "Valid")
        
        engine = ExecutionEngine(mock_client, mock_risk_manager)
        
        # Mock order placement
        mock_client.place_order = AsyncMock(return_value="order_1")
        mock_client.get_order = AsyncMock(
            return_value=MagicMock(filled_quantity=50, average_price=100.5)
        )
        
        group = MultiLegOrderGroup(strategy_name="iron_condor")
        signal1 = Signal(
            tradingsymbol="20000CE",
            exchange=Exchange.NFO,
            transaction_type=TransactionType.SELL,
            quantity=50,
        )
        
        group.add_leg(signal1, required_qty=50)
        
        # Execute
        success = await group.execute_with_rollback(engine)
        
        assert success is True
        assert group.status == OrderGroupStatus.COMPLETE
    
    @pytest.mark.asyncio
    async def test_execute_with_rollback_partial_fill(self):
        """Test rollback on partial fill."""
        # Mock objects
        mock_client = AsyncMock(spec=KiteClient)
        mock_risk_manager = MagicMock(spec=RiskManager)
        mock_risk_manager.validate_signal.return_value = (True, "Valid")
        
        engine = ExecutionEngine(mock_client, mock_risk_manager)
        
        # Mock partial fill
        mock_client.place_order = AsyncMock(return_value="order_1")
        mock_client.get_order = AsyncMock(
            return_value=MagicMock(filled_quantity=30, average_price=100.5)  # Only 30 of 50
        )
        mock_client.cancel_order = AsyncMock()
        
        group = MultiLegOrderGroup(strategy_name="iron_condor")
        signal1 = Signal(
            tradingsymbol="20000CE",
            exchange=Exchange.NFO,
            transaction_type=TransactionType.SELL,
            quantity=50,
        )
        
        group.add_leg(signal1, required_qty=50)
        
        # Execute
        success = await group.execute_with_rollback(engine)
        
        assert success is False
        assert group.status == OrderGroupStatus.PARTIAL
        # Verify cancel was called
        mock_client.cancel_order.assert_called()


class TestStopLossExecution:
    """Test Fix #2: Stop-Loss Execution Price."""
    
    def test_apply_stop_loss_buy_order(self):
        """Test stop-loss for BUY order."""
        mock_client = MagicMock(spec=KiteClient)
        mock_risk_manager = MagicMock(spec=RiskManager)
        
        engine = ExecutionEngine(mock_client, mock_risk_manager)
        
        signal = Signal(
            tradingsymbol="NIFTY",
            exchange=Exchange.NSE,
            transaction_type=TransactionType.BUY,
            quantity=1,
            price=100.0,
            stop_loss=95.0,
        )
        
        order = engine._signal_to_order(signal)
        engine._apply_stop_loss_to_order(order, signal)
        
        # For BUY, execution price should be below trigger
        assert order.trigger_price == 95.0
        assert order.price == 94.525  # 95.0 * (1 - 0.5/100)
        assert order.order_type == OrderType.SL
    
    def test_apply_stop_loss_sell_order(self):
        """Test stop-loss for SELL order."""
        mock_client = MagicMock(spec=KiteClient)
        mock_risk_manager = MagicMock(spec=RiskManager)
        
        engine = ExecutionEngine(mock_client, mock_risk_manager)
        
        signal = Signal(
            tradingsymbol="NIFTY",
            exchange=Exchange.NSE,
            transaction_type=TransactionType.SELL,
            quantity=1,
            price=100.0,
            stop_loss=105.0,
        )
        
        order = engine._signal_to_order(signal)
        engine._apply_stop_loss_to_order(order, signal)
        
        # For SELL, execution price should be above trigger
        assert order.trigger_price == 105.0
        assert order.price == 105.525  # 105.0 * (1 + 0.5/100)
        assert order.order_type == OrderType.SL


class TestPositionValidation:
    """Test Fix #3: Position Validation."""
    
    def test_validate_balanced_hedge(self):
        """Test validation of balanced buy/sell hedge."""
        tracker = PositionTracker()
        validator = PositionValidator(tracker)
        
        # Add balanced positions
        tracker.add_position(
            tradingsymbol="20000CE",
            strategy_name="bull_call_credit_spread",
            transaction_type=TransactionType.SELL,
            entry_price=100.0,
            quantity=50,
            signal_id="sig1",
        )
        
        tracker.add_position(
            tradingsymbol="21000CE",
            strategy_name="bull_call_credit_spread",
            transaction_type=TransactionType.BUY,
            entry_price=50.0,
            quantity=50,
            signal_id="sig2",
        )
        
        result = validator.validate_multi_leg_hedge("bull_call_credit_spread")
        
        assert result["validated"] is True
        assert len(result["issues"]) == 0
    
    def test_validate_mismatched_hedge(self):
        """Test validation of mismatched buy/sell hedge."""
        tracker = PositionTracker()
        validator = PositionValidator(tracker)
        
        # Add mismatched positions
        tracker.add_position(
            tradingsymbol="20000CE",
            strategy_name="bull_call_credit_spread",
            transaction_type=TransactionType.SELL,
            entry_price=100.0,
            quantity=50,
            signal_id="sig1",
        )
        
        tracker.add_position(
            tradingsymbol="21000CE",
            strategy_name="bull_call_credit_spread",
            transaction_type=TransactionType.BUY,
            entry_price=50.0,
            quantity=30,  # Only 30 instead of 50
            signal_id="sig2",
        )
        
        result = validator.validate_multi_leg_hedge("bull_call_credit_spread")
        
        assert result["validated"] is False
        assert len(result["issues"]) > 0
        assert result["issues"][0]["variance"] == 20


class TestMarginPreValidation:
    """Test Fix #5: Margin Pre-Validation."""
    
    @pytest.mark.asyncio
    async def test_margin_validation_sufficient(self):
        """Test margin validation when sufficient."""
        mock_client = AsyncMock(spec=KiteClient)
        
        risk_mgr = RiskManager(client=mock_client)
        
        signals = [
            Signal(
                tradingsymbol="20000CE",
                exchange=Exchange.NFO,
                transaction_type=TransactionType.SELL,
                quantity=50,
                product=ProductType.NRML,
                price=100.0,
            )
        ]
        
        # Sufficient margin available
        margins_data = {"available": 1000000.0}
        
        result = await risk_mgr.validate_margin_for_multi_leg(signals, margins_data)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_margin_validation_insufficient(self):
        """Test margin validation when insufficient."""
        mock_client = AsyncMock(spec=KiteClient)
        
        risk_mgr = RiskManager(client=mock_client)
        
        signals = [
            Signal(
                tradingsymbol="20000CE",
                exchange=Exchange.NFO,
                transaction_type=TransactionType.SELL,
                quantity=50,
                product=ProductType.NRML,
                price=100.0,
            )
        ]
        
        # Insufficient margin
        margins_data = {"available": 1000.0}  # Way too low
        
        with pytest.raises(Exception):  # Should raise RiskLimitError
            await risk_mgr.validate_margin_for_multi_leg(signals, margins_data)


class TestExerciseHandling:
    """Test Fix #4: Exercise Handling."""
    
    def test_parse_strike(self):
        """Test parsing strike from symbol."""
        mock_client = MagicMock(spec=KiteClient)
        tracker = PositionTracker()
        journal = TradeJournal()
        
        handler = ExerciseHandler(mock_client, tracker, journal)
        
        strike = handler._parse_strike("20000CE21FEB")
        assert strike == 20000.0
        
        strike = handler._parse_strike("19000PE21FEB")
        assert strike == 19000.0
    
    def test_is_option(self):
        """Test option symbol detection."""
        mock_client = MagicMock(spec=KiteClient)
        tracker = PositionTracker()
        journal = TradeJournal()
        
        handler = ExerciseHandler(mock_client, tracker, journal)
        
        assert handler._is_option("20000CE") is True
        assert handler._is_option("20000PE") is True
        assert handler._is_option("NIFTY") is False
    
    def test_is_itm_call(self):
        """Test ITM detection for call option."""
        mock_client = MagicMock(spec=KiteClient)
        tracker = PositionTracker()
        journal = TradeJournal()
        
        handler = ExerciseHandler(mock_client, tracker, journal)
        
        # Call option ITM when spot > strike
        assert handler._is_itm({}, 21000, 20000, "CE") is True
        assert handler._is_itm({}, 19000, 20000, "CE") is False
    
    def test_is_itm_put(self):
        """Test ITM detection for put option."""
        mock_client = MagicMock(spec=KiteClient)
        tracker = PositionTracker()
        journal = TradeJournal()
        
        handler = ExerciseHandler(mock_client, tracker, journal)
        
        # Put option ITM when spot < strike
        assert handler._is_itm({}, 19000, 20000, "PE") is True
        assert handler._is_itm({}, 21000, 20000, "PE") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
