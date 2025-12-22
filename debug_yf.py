import yfinance as yf

def test_fetch(ticker, period, interval):
    print(f"Testing {ticker} ({period}, {interval})...")
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        print(f"Result: {len(df)} rows")
        if not df.empty:
            print(df.head(2))
            print(df.tail(2))
        else:
            print("Empty DataFrame.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test 1: Indian Stock, Hourly
    test_fetch("RELIANCE.NS", "5d", "1h")
    
    # Test 2: US Stock, Hourly (Control)
    test_fetch("AAPL", "5d", "1h")
    
    # Test 3: Daily Data
    test_fetch("RELIANCE.NS", "1mo", "1d")
