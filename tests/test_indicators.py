from src.features.indicators import IndicatorEngine
import pandas as pd
import numpy as np

def test_indicators():
    print("Testing Indicators...")
    data = {
        'high': [100, 105, 110, 108, 112] * 10,
        'low': [90, 95, 100, 98, 102] * 10,
        'close': [95, 100, 105, 102, 108] * 10,
        'volume': [1000, 1200, 1500, 1100, 1300] * 10
    }
    df = pd.DataFrame(data)
    
    df_all = IndicatorEngine.compute_all(df)
    
    # Assert RSI is valid
    assert 'rsi' in df_all.columns
    assert df_all['rsi'].min() >= 0 and df_all['rsi'].max() <= 100
    
    # Assert MACD exists
    assert 'macd' in df_all.columns
    assert 'macd_hist' in df_all.columns
    
    # Assert Bollinger Bands
    assert 'bb_upper' in df_all.columns
    assert 'bb_percent' in df_all.columns
    
    # Assert ADX
    assert 'adx' in df_all.columns

    # Assert EMAs
    assert 'ema_9' in df_all.columns
    assert 'ema_21' in df_all.columns
    
    print("Indicator tests passed!")

if __name__ == "__main__":
    try:
        test_indicators()
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)
