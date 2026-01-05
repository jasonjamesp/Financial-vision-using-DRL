from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import json
from datetime import datetime
import random # Placeholder for real simulation

app = FastAPI()
templates = Jinja2Templates(directory="src/dashboard/templates")

# Mock data for initial UI dev
portfolio_data = {
    "balance": 10000.0,
    "equity": 10000.0,
    "pnl": 0.0,
    "pnl_pct": 0.0,
    "positions": [
        {"asset": "ETH-USD", "shares": 0, "value": 0, "avg_price": 0}
    ],
    "alerts": []
}

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/portfolio")
async def get_portfolio():
    return portfolio_data

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Simulate real-time updates
            portfolio_data["equity"] += random.uniform(-10, 10)
            portfolio_data["pnl"] = portfolio_data["equity"] - 10000.0
            portfolio_data["pnl_pct"] = (portfolio_data["pnl"] / 10000.0) * 100
            
            await websocket.send_text(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "portfolio": portfolio_data,
                "price": random.uniform(2000, 2100) # ETH-USD mock
            }))
            await asyncio.sleep(1)
    except Exception as e:
        print(f"WebSocket closed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
