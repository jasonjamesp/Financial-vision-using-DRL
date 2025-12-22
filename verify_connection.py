import yfinance as yf
import requests_cache
import numpy as np
import pandas as pd
import shutil

def verify():
    print(f"yfinance version: {yf.__version__}")
    print(f"numpy version: {np.__version__}")
    
    # Clear any old cache
    try:
        shutil.rmtree('yfinance_cache', ignore_errors=True)
        print("Required Cache cleared.")
    except:
        pass

    session = requests_cache.CachedSession('yfinance_cache', expire_after=3600)
    
    ticker = "RELIANCE.NS"
    print(f"\nAttempting to fetch {ticker} with Cache Session...")
    
    try:
        # dat = yf.Ticker(ticker, session=session) # Incompatible with new yfinance
        dat = yf.Ticker(ticker)
        df = dat.history(period='5d', interval='1h')
        
        if not df.empty:
            print(f"SUCCESS! Downloaded {len(df)} rows.")
            print(df.tail(2)[['Close', 'Volume']])
        else:
            print("FAILED. Dataframe is empty.")
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
