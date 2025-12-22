from backend import MarketEngine
import pandas as pd
import numpy as np

def test_fallback():
    print("Initializing Engine...")
    engine = MarketEngine()
    
    print("\n--- Test 1: Direct Synthetic Generation ---")
    try:
        df = engine._generate_synthetic_data("TEST.NS")
        print("Generated DataFrame:")
        print(df.tail())
        print("Columns:", df.columns)
        print("Index Type:", type(df.index))
    except Exception as e:
        print(f"FAILED Generataion: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Test 2: Full Pipeline on Bogus Ticker (Trigger Fallback) ---")
    try:
        # Use a ticker that definitely fails yfinance to trigger fallback
        status = engine.run_live_analysis("INVALID_TICKER_XYZ")
        
        if status:
            print("SUCCESS. Status Keys:", status.keys())
            print("Decision:", status['ai_decision'])
            print("Source:", status.get('data_source'))
        else:
            print("FAILED. Returned None.")
    except Exception as e:
        print(f"FAILED Pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fallback()
