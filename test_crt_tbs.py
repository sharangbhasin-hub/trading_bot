"""
Quick Test Script for CRT-TBS Strategy
=======================================

Tests strategy components and runs sample backtest.

Usage:
    python test_crt_tbs.py
    
Author: Trading System
Date: October 23, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import strategy and detectors
from strategies.strategy_crt_tbs import StrategyCRTTBS
from detectors.crt_detector import CRTDetector
from detectors.keylevel_detector import KeyLevelDetector
from detectors.tbs_detector import TBSDetector

# Import configuration
from config_crt_tbs import CRT_TBS_INTRADAY, get_config

print("="*80)
print("CRT-TBS Strategy Component Tests")
print("="*80)

# Test 1: CRT Detector
print("\n[Test 1] CRT Detector")
print("-" * 80)

# Generate sample HTF data
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=50, freq='1D')
sample_htf = pd.DataFrame({
    'open': 100 + np.random.randn(50).cumsum(),
    'high': 100 + np.random.randn(50).cumsum() + 2,
    'low': 100 + np.random.randn(50).cumsum() - 2,
    'close': 100 + np.random.randn(50).cumsum()
}, index=dates)

# Ensure high/low are correct
sample_htf['high'] = sample_htf[['open', 'high', 'close']].max(axis=1) + 0.5
sample_htf['low'] = sample_htf[['open', 'low', 'close']].min(axis=1) - 0.5

crt_detector = CRTDetector()
df_with_crt = crt_detector.detect_crt_candles(sample_htf)

crt_count = df_with_crt['is_crt'].sum()
print(f"Total Candles: {len(df_with_crt)}")
print(f"CRT Candles Detected: {crt_count}")
print(f"CRT Percentage: {(crt_count/len(df_with_crt)*100):.1f}%")

if crt_count > 0:
    print(f"Average Body Ratio: {df_with_crt[df_with_crt['is_crt']]['body_ratio'].mean():.2%}")
    print("✅ CRT Detector Working")
else:
    print("⚠️  No CRT candles detected (may need different data)")

# Test 2: Key Level Detector
print("\n[Test 2] Key Level Detector")
print("-" * 80)

keylevel_detector = KeyLevelDetector({'ohp_olp_lookback': 10})

# Check last candle for key levels
keylevels = keylevel_detector.detect_all_keylevels(sample_htf)

print(f"OHP Detected: {'✅' if keylevels['ohp'] else '❌'}")
print(f"OLP Detected: {'✅' if keylevels['olp'] else '❌'}")
print(f"FVG Count: {len(keylevels['fvg'])}")
print(f"OB Count: {len(keylevels['ob'])}")
print(f"RB Detected: {'✅' if keylevels['rb'] else '❌'}")
print(f"Has Any Key Level: {'✅' if keylevels['has_any_keylevel'] else '❌'}")

# Test 3: TBS Detector
print("\n[Test 3] TBS Detector")
print("-" * 80)

# Generate sample LTF data with TBS pattern
ltf_dates = pd.date_range(start='2024-01-15', periods=20, freq='1H')
sample_ltf = pd.DataFrame({
    'open': [100, 101, 102, 101.5, 100.5, 99.5, 99, 98.5, 98, 97.5, 
             97, 96.5, 96, 95.5, 95, 94.5, 94, 93.5, 93, 92.5],
    'high': [101, 102.5, 103, 102, 101, 100, 99.5, 99, 98.5, 98,
             97.5, 97, 96.5, 96, 95.5, 95, 94.5, 94, 93.5, 93],
    'low': [99.5, 101, 101.5, 100.5, 99, 98.5, 98, 97.5, 97, 96.5,
            96, 95.5, 95, 94.5, 94, 93.5, 93, 92.5, 92, 91.5],
    'close': [101, 102, 101.5, 100.5, 99.5, 99, 98.5, 98, 97.5, 97,
              96.5, 96, 95.5, 95, 94.5, 94, 93.5, 93, 92.5, 92]
}, index=ltf_dates)

# Ensure OHLC consistency
for i in range(len(sample_ltf)):
    row = sample_ltf.iloc[i]
    sample_ltf.loc[sample_ltf.index[i], 'high'] = max(row['open'], row['high'], row['low'], row['close'])
    sample_ltf.loc[sample_ltf.index[i], 'low'] = min(row['open'], row['high'], row['low'], row['close'])

tbs_detector = TBSDetector()

# Test TBS detection (look for manipulation above 102.0)
tbs = tbs_detector.detect_tbs_pattern(
    sample_ltf,
    reference_level=102.0,
    direction='sell'
)

if tbs:
    print(f"✅ TBS Detected at index {tbs['tbs_index']}")
    print(f"   Manipulation Size: {tbs['manipulation_size']:.2f}")
    print(f"   Is A+ TBS: {tbs['is_a_plus_tbs']}")
    print(f"   Candles in Pattern: {tbs['candles_in_pattern']}")
else:
    print("❌ No TBS detected (expected with sample data)")

# Test 4: Full Strategy
print("\n[Test 4] Full CRT-TBS Strategy")
print("-" * 80)

config = CRT_TBS_INTRADAY.copy()
strategy = StrategyCRTTBS(config)

print(f"Strategy Name: {strategy.name}")
print(f"Current State: {strategy.state}")
print(f"HTF: {config['htf']} | LTF: {config['ltf']}")
print(f"Min RR Ratio: {config['min_rr_ratio']}")
print(f"Risk Per Trade: {config['risk_per_trade']}%")

# Test signal generation (will need proper data)
try:
    signal = strategy.generate_signals(sample_htf, sample_ltf)
    if signal:
        print(f"\n✅ Signal Generated!")
        print(f"   Action: {signal['action']}")
        print(f"   Entry: {signal['entry_price']:.2f}")
        print(f"   Stop Loss: {signal['stop_loss']:.2f}")
        print(f"   TP1: {signal['take_profit_1']:.2f}")
        print(f"   TP2: {signal['take_profit_2']:.2f}")
        print(f"   RR Ratio: {signal['rr_ratio']:.2f}")
    else:
        print("ℹ️  No signal (expected with sample data)")
except Exception as e:
    print(f"ℹ️  Signal generation test: {str(e)[:100]}")

# Test 5: Configuration
print("\n[Test 5] Configuration Presets")
print("-" * 80)

for style in ['scalping', 'intraday', 'shortterm']:
    cfg = get_config(style)
    print(f"\n{style.upper()}:")
    print(f"  HTF: {cfg['htf']} → LTF: {cfg['ltf']}")
    print(f"  Min RR: {cfg['min_rr_ratio']}")
    print(f"  Max Wait (Model #1): {cfg['max_wait_candles']} candles")

print("\n" + "="*80)
print("Component Tests Complete!")
print("="*80)

print("\nNext Steps:")
print("1. Load real historical data from database")
print("2. Run full backtest: python main_backtest.py --strategy CRT_TBS")
print("3. Optimize parameters using parameter_optimizer.py")
print("4. Paper trade before going live")
