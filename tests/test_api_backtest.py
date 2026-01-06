import requests
import json

def test_api_schema():
    print("Testing API Schema...")
    url = "http://localhost:8000/api/training/backtest/run?asset=ETH-USD&days=5"
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        
        assert data['status'] == 'success'
        res = data['result']
        
        # Verify required fields
        required_fields = [
            'asset', 'start_date', 'end_date', 'total_return_pct',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown_pct',
            'win_rate', 'total_trades', 'equity_curve',
            'drawdown_curve', 'price_series', 'trades'
        ]
        
        for field in required_fields:
            assert field in res, f"Missing field: {field}"
            
        print("API Schema test passed!")
        
    except Exception as e:
        print(f"API Test failed: {e}")
        exit(1)

if __name__ == "__main__":
    test_api_schema()
