#!/usr/bin/env python3
"""Test all critical imports for recent fixes."""

import sys

def test_imports():
    """Test that all critical modules can be imported."""
    
    results = []
    
    # Test 1: order_group_corrected
    try:
        from src.execution.order_group_corrected import (
            CorrectMultiLegOrderGroup, 
            PersistenceLayer, 
            ExecutionLifecycle,
            EmergencyHedgeExecutor,
            HybridStopLossManager,
            WorstCaseMarginSimulator,
            LegPriorityCalculator,
            InterimExposureHandler,
        )
        results.append(("✓", "order_group_corrected", "All 8 classes imported"))
    except Exception as e:
        results.append(("✗", "order_group_corrected", str(e)))
    
    # Test 2: credit_spreads
    try:
        from src.options.credit_spreads import (
            BullCallCreditSpreadStrategy,
            BearPutCreditSpreadStrategy
        )
        results.append(("✓", "credit_spreads", "Both spread strategies imported"))
    except Exception as e:
        results.append(("✗", "credit_spreads", str(e)))
    
    # Test 3: exercise_handler
    try:
        from src.options.exercise_handler import (
            ExerciseHandler,
            ExerciseEvent
        )
        results.append(("✓", "exercise_handler", "Handler and event classes imported"))
    except Exception as e:
        results.append(("✗", "exercise_handler", str(e)))
    
    # Test 4: position_validator
    try:
        from src.risk.position_validator import PositionValidator
        results.append(("✓", "position_validator", "Validator imported"))
    except Exception as e:
        results.append(("✗", "position_validator", str(e)))
    
    # Test 5: original order_group
    try:
        from src.execution.order_group import (
            MultiLegOrderGroup,
            OrderGroupStatus,
            OrderGroupLeg
        )
        results.append(("✓", "order_group", "Original multi-leg group imported"))
    except Exception as e:
        results.append(("✗", "order_group", str(e)))
    
    # Print results
    print("\n" + "="*70)
    print("IMPORT TEST RESULTS")
    print("="*70)
    
    for status, module, message in results:
        print(f"{status} {module:30s} {message}")
    
    print("="*70)
    
    # Summary
    passed = sum(1 for s, _, _ in results if s == "✓")
    total = len(results)
    
    print(f"\nSummary: {passed}/{total} modules imported successfully")
    
    if passed == total:
        print("✅ ALL IMPORTS SUCCESSFUL - Production code structure is valid")
        return 0
    else:
        print("❌ SOME IMPORTS FAILED - Fix import issues")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
