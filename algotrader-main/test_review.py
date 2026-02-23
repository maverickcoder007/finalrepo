"""Quick integration test for analysis strategies and FnO builder."""
from src.api.service import TradingService
from src.options.fno_builder import (
    FnOStrategyStore, FNO_TEMPLATES, compute_strategy_payoff,
    resolve_legs, FnOStrategyConfig, FnOLeg
)
from src.strategy.analysis_strategies import ANALYSIS_STRATEGIES, ANALYSIS_STRATEGY_LABELS

# Test 1: Analysis strategy list
labels = [{"value": k, "label": v} for k, v in ANALYSIS_STRATEGY_LABELS.items()]
assert len(labels) == 12, f"Expected 12, got {len(labels)}"
print(f"[PASS] Analysis strategies: {len(labels)} items")

# Test 2: All analysis strategies can be instantiated
for name, cls in ANALYSIS_STRATEGIES.items():
    s = cls()
    assert s.name == name, f"Name mismatch: {s.name} != {name}"
print(f"[PASS] All 12 analysis strategies instantiate correctly")

# Test 3: FnO templates
store = FnOStrategyStore()
templates = store.get_templates()
assert len(templates) == 8, f"Expected 8 templates, got {len(templates)}"
print(f"[PASS] Templates: {len(templates)} items")
for t in templates:
    assert "name" in t and "legs" in t and "description" in t
    print(f"  {t['name']}: {len(t['legs'])} legs")

# Test 4: Payoff computation (credit spread: sell 25500 CE, buy 25750 CE)
config = FnOStrategyConfig(
    name="test_spread",
    underlying="NIFTY",
    strategy_type="credit_spread",
    legs=[
        FnOLeg(action="sell", option_type="CE", strike_mode="absolute", strike_value=25500, lots=1),
        FnOLeg(action="buy", option_type="CE", strike_mode="absolute", strike_value=25750, lots=1),
    ]
)
resolved = resolve_legs(config, 25500, chain=None)
assert len(resolved) == 2
print(f"[PASS] Resolved 2 legs: {resolved[0]['strike']} / {resolved[1]['strike']}")

payoff = compute_strategy_payoff(resolved, 25500, 50)
assert len(payoff["prices"]) == 101
assert len(payoff["payoff"]) == 101
assert "max_profit" in payoff and "max_loss" in payoff and "breakeven" in payoff
print(f"[PASS] Payoff: max_profit={payoff['max_profit']}, max_loss={payoff['max_loss']}")
print(f"  Breakevens: {payoff['breakeven']}")

# Test 5: Offset mode
config2 = FnOStrategyConfig(
    name="test_offset",
    underlying="NIFTY",
    strategy_type="credit_spread",
    legs=[
        FnOLeg(action="sell", option_type="CE", strike_mode="offset", strike_value=0, lots=1),
        FnOLeg(action="buy", option_type="CE", strike_mode="offset", strike_value=250, lots=1),
    ]
)
resolved2 = resolve_legs(config2, 25500, chain=None)
assert resolved2[0]["strike"] == 25500
assert resolved2[1]["strike"] == 25750
print(f"[PASS] Offset mode: ATM={resolved2[0]['strike']}, ATM+250={resolved2[1]['strike']}")

# Test 6: Save/load/delete
store2 = FnOStrategyStore(path="data/test_fno_strategies.json")
result = store2.save(config)
assert result["status"] == "saved"
loaded = store2.get("test_spread")
assert loaded is not None
assert loaded.name == "test_spread"
all_list = store2.list_all()
assert len(all_list) >= 1
del_result = store2.delete("test_spread")
assert del_result["status"] == "deleted"
print("[PASS] Save/load/delete cycle works")

# Cleanup
import os
if os.path.exists("data/test_fno_strategies.json"):
    os.remove("data/test_fno_strategies.json")

print("\n=== ALL TESTS PASSED ===")
