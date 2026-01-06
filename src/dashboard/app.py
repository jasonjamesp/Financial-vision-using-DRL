from fastapi import FastAPI, WebSocket, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from src.training.training_logger import TrainingLogger
from src.testing.trading_logger import TradingLogger
from src.training.trading_env import TradingEnv
from src.training.ppo_agent import PPOAgent
from src.training.backtester import Backtester
from src.data_pipeline.data_manager import DataManager
from src.data_pipeline.data_validator import DataValidator
from src.features.indicators import IndicatorEngine
from src.features.feature_builder import FeatureBuilder

app = FastAPI()
templates = Jinja2Templates(directory="src/dashboard/templates")

train_logger = TrainingLogger()
trade_logger = TradingLogger()

# Global storage for latest backtest result
latest_backtest_result = None

def load_json_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def save_json_config(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Training Endpoints ---

@app.get("/api/training/backtest/run")
async def run_backtest(asset: str = "ETH-USD", days: int = 30):
    global latest_backtest_result
    try:
        dm = DataManager()
        raw_df = dm.fetch_ohlcv(asset, days=days)
        dv = DataValidator()
        df = dv.validate_ohlcv(raw_df)
        
        if df.empty:
            return {"status": "error", "message": "No data found"}
            
        df = IndicatorEngine.compute_all(df)
        fb = FeatureBuilder()
        feature_df = fb.build_features(df)
        
        env = TradingEnv(df, feature_df, asset_name=asset)
        agent = PPOAgent()
        
        config = load_json_config("config/trading_config.json")
        checkpoint = config.get("model_checkpoint")
        if checkpoint and os.path.exists(checkpoint):
            agent.load(checkpoint)
        
        bt = Backtester(agent, env)
        result = bt.run()
        latest_backtest_result = result
        
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/training/backtest/latest")
async def get_latest_backtest():
    return latest_backtest_result

@app.get("/api/training/diary")
async def get_training_diary(limit: int = 50):
    return train_logger.get_diary(limit=limit)

@app.get("/api/training/stats")
async def get_training_stats():
    return train_logger.get_stats()

@app.get("/api/training/config")
async def get_training_config():
    return load_json_config("config/training_config.json")

@app.post("/api/training/config")
async def update_training_config(config: dict = Body(...)):
    save_json_config("config/training_config.json", config)
    return {"status": "success"}

# --- Testing Endpoints ---

@app.get("/api/testing/trades")
async def get_trading_trades(limit: int = 50):
    return trade_logger.get_trades(limit=limit)

@app.get("/api/testing/performance")
async def get_trading_performance():
    return trade_logger.get_performance()

@app.get("/api/testing/config")
async def get_testing_config():
    return load_json_config("config/trading_config.json")

@app.post("/api/testing/config")
async def update_testing_config(config: dict = Body(...)):
    save_json_config("config/trading_config.json", config)
    return {"status": "success"}

# --- WebSocket for Live Data ---

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            latest_train = train_logger.get_diary(limit=1)
            latest_trade = trade_logger.get_trades(limit=1)
            
            payload = {
                "timestamp": datetime.now().isoformat(),
                "training": latest_train[0] if latest_train else None,
                "testing": latest_trade[0] if latest_trade else None
            }
            
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(2)
    except Exception as e:
        print(f"WebSocket closed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
