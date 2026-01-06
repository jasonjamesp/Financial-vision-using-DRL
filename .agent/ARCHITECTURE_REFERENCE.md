# 🔒 PRIVATE: GAF-PPO Architecture Reference
# Last Updated: 2026-01-06
# Purpose: Internal reference for understanding system architecture. Update when making major changes.

---

## 📂 Project Structure Overview

```
finance2/
├── config/                    # JSON configurations
│   ├── shared_config.json     # Common settings
│   ├── trading_config.json    # Live trading params
│   └── training_config.json   # Training hyperparams
├── data/
│   ├── cache/                 # SQLite DB (market_data.db)
│   ├── gaf_images/            # Generated GAF images
│   └── raw/                   # CSV backups
├── models/
│   ├── checkpoints/           # Training checkpoints
│   └── trained/               # Final models
├── scripts/
│   ├── run_dashboard.py       # Entry: Starts FastAPI server
│   ├── start_training.py      # Entry: Training orchestrator
│   ├── start_live_test.py     # Entry: Paper trading
│   └── smoke_test.py          # Pipeline verification
├── src/
│   ├── dashboard/             # FastAPI + Jinja2 UI
│   ├── data_pipeline/         # Data fetching, validation, GAF encoding
│   ├── features/              # Technical indicators + ML features
│   ├── risk_management/       # Gatekeeper risk logic
│   ├── testing/               # Live paper trading
│   ├── training/              # PPO, Env, Backtester
│   └── utils/                 # Config, constants
└── tests/                     # Miniature verification tests
```

---

## 🧠 Core Components

### 1. Data Pipeline (`src/data_pipeline/`)

| File | Class/Function | Purpose |
|------|----------------|---------|
| `data_manager.py` | `DataManager` | Fetch from yfinance, SQLite cache with upsert |
| `data_validator.py` | `DataValidator` | Check OHLCV integrity, fill gaps |
| `gaf_encoder.py` | `GAFEncoder` | Convert 64-candle window to 64x64 Gramian Angular Field image |

**Key Notes:**
- yfinance has 60-day limit for 15m candles
- SQLite uses asset+timestamp composite primary key
- GAF encoding uses `pyts.image.GramianAngularField`

---

### 2. Feature Engineering (`src/features/`)

| File | Class | Purpose |
|------|-------|---------|
| `indicators.py` | `IndicatorEngine` | Compute 8 technical indicators (all static methods) |
| `feature_builder.py` | `FeatureBuilder` | Normalize indicators into 12-dim ML feature vector |

**Indicators Computed:**
1. RSI (14)
2. MACD (12/26/9) + Histogram
3. Bollinger Bands (20, 2σ) + BB%
4. ATR (14)
5. OBV
6. ADX (14)
7. Stochastic (14, 3)
8. EMA (9, 21, 50)

**Feature Vector (12 dims):**
- rsi_norm, macd_hist_norm, stoch_k_norm
- bb_pos, atr_norm, adx_norm
- price_ema9_ratio, ema9_21_ratio
- obv_norm, vol_ratio, log_returns, volatility

---

### 3. Training Environment (`src/training/`)

| File | Class | Purpose |
|------|-------|---------|
| `trading_env.py` | `TradingEnv` | Gymnasium env with hybrid observation space |
| `ppo_agent.py` | `PPOAgent`, `ActorCritic` | Multi-input PPO with CNN + MLP fusion |
| `cnn_encoder.py` | `CNNEncoder` | 4-layer CNN for 64x64 GAF → 512 features |
| `reward_function.py` | `RewardFunction` | Risk-adjusted reward with drawdown penalty |
| `backtester.py` | `Backtester` | Simulate agent on historical data, compute metrics |
| `backtest_result.py` | `BacktestResult` | Dataclass for backtest output |
| `walk_forward.py` | `WalkForwardValidator` | Rolling train/test splits |
| `train.py` | `train()` | Main training loop |
| `training_logger.py` | `TrainingLogger` | SQLite logging of episodes |

**Observation Space (Dict):**
```python
{
    "image": Box[1, 64, 64],      # GAF image
    "indicators": Box[12],         # Technical features
    "state": Box[5]               # [balance, shares, net_worth, drawdown, pnl_pct]
}
```

**Action Space:** Discrete(3) → [Hold, Buy All, Sell All]

**ActorCritic Network:**
```
GAF Image (1,64,64) → CNNEncoder → 512
Indicators (12) → Linear(64) → 64
State (5) → Linear(32) → 32
                ↓
        Concatenate (608)
                ↓
    Actor → Softmax(3)
    Critic → Value(1)
```

---

### 4. Live Testing (`src/testing/`)

| File | Class | Purpose |
|------|-------|---------|
| `live_test.py` | `LiveTester` | Real-time paper trading loop |
| `trading_logger.py` | `TradingLogger` | SQLite logging of trades |

**Flow:**
1. Fetch recent 100 candles via DataManager
2. Compute indicators via IndicatorEngine
3. Build features via FeatureBuilder
4. Generate GAF from last 64 candles
5. Agent selects action
6. Log trade decision (no real execution)

---

### 5. Dashboard (`src/dashboard/`)

| File | Purpose |
|------|---------|
| `app.py` | FastAPI endpoints + WebSocket |
| `templates/index.html` | Plotly-powered trading terminal |

**Key Endpoints:**
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Dashboard HTML |
| `/api/training/backtest/run` | GET | Trigger backtest |
| `/api/training/backtest/latest` | GET | Get last backtest result |
| `/api/training/diary` | GET | Training episodes log |
| `/api/testing/trades` | GET | Live trade log |
| `/ws/live` | WS | Real-time updates |

**Backtest Results Modal:**
- Equity Curve (Plotly line)
- Drawdown Curve (Plotly area)
- Price + Trade Markers (Plotly scatter)
- Metrics: Sharpe, Sortino, Max DD, Win Rate

---

### 6. Risk Management (`src/risk_management/`)

| File | Class | Purpose |
|------|-------|---------|
| `risk_scanner.py` | `RiskScanner` | Monitor portfolio health |
| `portfolio_optimizer.py` | `PortfolioOptimizer` | Multi-asset allocation |
| `alerts.py` | `AlertManager` | Notification triggers |

**Risk Thresholds:**
- Max Drawdown > 10% → Exit
- Sharpe < 0.5 → Reduce 50%
- Volatility > 2x 20-day → Reduce 25%
- Win Rate < 40% (50+ trades) → Pause

---

## ⚠️ Known Issues & Potential Errors

### 1. Data Issues
| Issue | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'close'` | yfinance returns `Close` (capitalized) | Check column rename logic in `DataManager` |
| Empty DataFrame | Market closed, no data | Fall back to cached data |
| 60-day limit | yfinance restriction for 15m data | Use daily candles for longer backtests |

### 2. Model Issues
| Issue | Cause | Solution |
|-------|-------|----------|
| `mat1 and mat2 shapes` | Feature dimension mismatch | Ensure `indicator_dim=12` in PPOAgent |
| NaN in gradients | Division by zero in reward | Add epsilon to denominators |
| Agent always HOLDs | Random init, no training | Run full training loop |

### 3. Environment Issues
| Issue | Cause | Solution |
|-------|-------|----------|
| `IndexError: iloc` | Step exceeds df length | Check `done` condition in `step()` |
| NaN in observations | Indicators with insufficient warmup | Fill NaN with `fillna(0)` or `bfill` |

### 4. Dashboard Issues
| Issue | Cause | Solution |
|-------|-------|----------|
| `NameError: app` | Endpoint defined before FastAPI init | Ensure `app = FastAPI()` is at top |
| WebSocket disconnect | Long-running backtest blocks event loop | Use background tasks |
| CORS errors | Browser security | Add `CORSMiddleware` if needed |

### 5. Git/Deployment Issues
| Issue | Cause | Solution |
|-------|-------|----------|
| Push timeout | Large files or slow network | `git config http.postBuffer 524288000` |
| Module not found | Missing PYTHONPATH | `pip install -e .` or set env var |

---

## 🔄 Update Log

| Date | Change | Files Affected |
|------|--------|----------------|
| 2026-01-06 | Added Sortino ratio, drawdown curve | `backtester.py`, `backtest_result.py` |
| 2026-01-06 | Enhanced results modal UI | `index.html` |
| 2026-01-06 | Fixed indicator_dim 11→12 | `ppo_agent.py`, `trading_env.py` |
| 2026-01-06 | Added walk-forward validation | `walk_forward.py` |
| 2026-01-06 | Added technical indicators | `indicators.py`, `feature_builder.py` |
