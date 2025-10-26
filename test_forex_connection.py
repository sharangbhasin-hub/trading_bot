"""
Test Script for ForexHandler
Validates MT5 connection, symbol loading, and data fetching
Run this BEFORE integrating into main UI
"""

import sys
from datetime import datetime, timedelta
from forex_handler import get_forex_handler
import pandas as pd

def test_connection():
    """Test 1: MT5 Connection"""
    print("\n" + "="*60)
    print("TEST 1: MT5 Connection & Initialization")
    print("="*60)
    
    try:
        handler = get_forex_handler(mode='backtest')
        
        if handler.connected:
            print("✅ MT5 Connected Successfully!")
            print(f"   Account Balance: ${handler.account_info.balance:,.2f}")
            print(f"   Account Leverage: 1:{handler.account_info.leverage}")
            print(f"   Account Currency: {handler.account_info.currency}")
            return handler
        else:
            print("❌ MT5 Connection Failed")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_symbol_loading(handler):
    """Test 2: Dynamic Symbol Loading"""
    print("\n" + "="*60)
    print("TEST 2: Dynamic Symbol Loading from MT5 API")
    print("="*60)
    
    try:
        # Get all symbols
        all_symbols = handler.get_available_symbols_by_category()
        print(f"✅ Total Symbols Loaded: {len(all_symbols)}")
        
        # Get by category
        categories = ['Forex Majors', 'Metals', 'Indices', 'Commodities']
        
        for category in categories:
            symbols = handler.get_available_symbols_by_category(category)
            if symbols:
                print(f"\n📊 {category}: {len(symbols)} symbols")
                # Print first 5 symbols in each category
                for i, sym in enumerate(symbols[:5]):
                    print(f"   {i+1}. {sym['symbol']:15} - {sym['description']}")
                if len(symbols) > 5:
                    print(f"   ... and {len(symbols)-5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading symbols: {e}")
        return False

def test_symbol_info(handler):
    """Test 3: Symbol Information Retrieval"""
    print("\n" + "="*60)
    print("TEST 3: Get Detailed Symbol Information")
    print("="*60)
    
    test_symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']  # Test with common symbols
    
    for symbol in test_symbols:
        try:
            info = handler.get_symbol_info(symbol)
            if info:
                print(f"\n✅ {symbol} Info:")
                print(f"   Description: {info['description']}")
                print(f"   Digits: {info['digits']}")
                print(f"   Spread: {info['spread']} points")
                print(f"   Contract Size: {info['contract_size']}")
                print(f"   Min Volume: {info['min_volume']}")
                print(f"   Base Currency: {info['currency_base']}")
            else:
                print(f"⚠️  {symbol} not found (might be named differently on your broker)")
        except Exception as e:
            print(f"❌ Error getting info for {symbol}: {e}")

def test_historical_data(handler):
    """Test 4: Historical Data Fetching"""
    print("\n" + "="*60)
    print("TEST 4: Fetch Historical OHLC Data")
    print("="*60)
    
    # Test parameters
    symbol = 'EURUSD'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)  # Last 7 days
    
    timeframes = ['5min', '15min', '1h', 'day']
    
    for tf in timeframes:
        try:
            print(f"\nFetching {symbol} {tf} data...")
            
            df = handler.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=tf
            )
            
            if not df.empty:
                print(f"✅ {symbol} {tf}: {len(df)} bars fetched")
                print(f"   Date Range: {df.index[0]} to {df.index[-1]}")
                print(f"   Latest Close: {df['close'].iloc[-1]:.5f}")
                print(f"   Sample Data:")
                print(df.tail(3).to_string())
            else:
                print(f"⚠️  No data returned for {symbol} {tf}")
                
        except Exception as e:
            print(f"❌ Error fetching {tf} data: {e}")

def test_live_tick(handler):
    """Test 5: Live Tick Data (Paper Trading Mode)"""
    print("\n" + "="*60)
    print("TEST 5: Get Live Tick Data (Paper Trading)")
    print("="*60)
    
    # Need to create a paper mode handler
    try:
        paper_handler = get_forex_handler(mode='paper')
        
        test_symbols = ['EURUSD', 'GBPUSD', 'XAUUSD']
        
        for symbol in test_symbols:
            tick = paper_handler.get_live_tick(symbol)
            if tick:
                print(f"\n✅ {symbol} Live Tick:")
                print(f"   Bid: {tick['bid']:.5f}")
                print(f"   Ask: {tick['ask']:.5f}")
                print(f"   Last: {tick['last']:.5f}")
                print(f"   Spread: {tick['spread']:.5f}")
                print(f"   Time: {tick['time']}")
            else:
                print(f"⚠️  No tick data for {symbol}")
                
    except Exception as e:
        print(f"❌ Error in live tick test: {e}")

def test_data_format_compatibility(handler):
    """Test 6: Verify Data Format Matches Kite Format"""
    print("\n" + "="*60)
    print("TEST 6: Verify Data Format (Kite Compatibility)")
    print("="*60)
    
    try:
        # Fetch sample data
        df = handler.get_historical_data(
            symbol='EURUSD',
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now(),
            timeframe='5min'
        )
        
        if not df.empty:
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            has_all_cols = all(col in df.columns for col in required_cols)
            
            if has_all_cols:
                print("✅ All required columns present:")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Index type: {type(df.index).__name__}")
                print(f"   Index name: {df.index.name}")
                
                # Check data types
                print("\n   Data Types:")
                print(df.dtypes)
                
                # Check for NaN values
                nan_count = df.isna().sum().sum()
                if nan_count == 0:
                    print("\n✅ No NaN values in data")
                else:
                    print(f"\n⚠️  Found {nan_count} NaN values")
                
                print("\n✅ Data format is compatible with your existing strategies!")
            else:
                missing = [c for c in required_cols if c not in df.columns]
                print(f"❌ Missing columns: {missing}")
        else:
            print("❌ No data to validate format")
            
    except Exception as e:
        print(f"❌ Error in format validation: {e}")

def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# ForexHandler Integration Test Suite")
    print("# Testing MT5 connection and data fetching")
    print("#"*60)
    
    # Test 1: Connection
    handler = test_connection()
    if not handler:
        print("\n❌ Connection failed. Please check:")
        print("   1. MT5 terminal is running and logged in")
        print("   2. .env file has correct MT5 credentials")
        print("   3. MetaTrader5 Python package is installed")
        return
    
    # Test 2: Symbol Loading
    symbols_ok = test_symbol_loading(handler)
    if not symbols_ok:
        print("\n⚠️  Symbol loading had issues, but continuing...")
    
    # Test 3: Symbol Info
    test_symbol_info(handler)
    
    # Test 4: Historical Data
    test_historical_data(handler)
    
    # Test 5: Live Ticks
    test_live_tick(handler)
    
    # Test 6: Format Compatibility
    test_data_format_compatibility(handler)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)
    print("\nIf all tests passed ✅, you're ready to integrate into main_backtest.py!")
    print("If any tests failed ❌, check the error messages above.")
    
    # Cleanup
    handler.shutdown()
    print("\nMT5 connection closed. Tests complete.")

if __name__ == "__main__":
    main()
