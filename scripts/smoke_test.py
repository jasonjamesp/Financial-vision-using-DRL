import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.data_manager import DataManager
from src.data_pipeline.data_validator import DataValidator
from src.data_pipeline.gaf_encoder import GAFEncoder

def test_pipeline():
    print("--- GAF-PPO Pipeline Smoke Test ---")
    
    # 1. Test Data Fetching
    dm = DataManager()
    print("Fetching ETH-USD data...")
    df = dm.fetch_ohlcv("ETH-USD", days=7)
    
    if df.empty:
        print("FAILED: Data fetch returned empty DataFrame")
        return
    print(f"SUCCESS: Fetched {len(df)} rows")

    # 2. Test Validation
    dv = DataValidator()
    print("Validating data...")
    df_clean = dv.validate_ohlcv(df)
    gaps = dv.detect_gaps(df_clean)
    
    if df_clean.empty:
        print("FAILED: Data validation failed")
        return
    print(f"SUCCESS: Data validated. Found {len(gaps)} potential gaps.")

    # 3. Test GAF Encoding
    ge = GAFEncoder()
    print("Encoding sample window to GAF...")
    sample_window = df_clean['close'].iloc[-64:].values
    img = ge.encode(sample_window)
    
    if img.shape != (64, 64):
        print(f"FAILED: GAF image shape is {img.shape}, expected (64, 64)")
        return
    print("SUCCESS: GAF image generated successfully.")
    
    print("\n--- Pipeline Check PASSED ---")

if __name__ == "__main__":
    test_pipeline()
