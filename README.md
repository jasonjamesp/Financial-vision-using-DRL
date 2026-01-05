GAF-PPO Financial Vision Trading System

A production-grade algorithmic trading platform that combines **Gramian Angular Field (GAF)** image encoding with **Proximal Policy Optimization (PPO)** reinforcement learning for intelligent, risk-aware trading decisions.



Overview

This system implements the "Financial Vision" approach from academic research, treating market data as images rather than time series. By encoding price movements into **Gramian Angular Field** images, the PPO agent can leverage convolutional neural networks to identify complex visual patterns that traditional quantitative methods might miss.

### Key Features

- Financial Vision**: Converts 15-minute candlestick windows into 64x64 GAF images
- Deep RL Agent**: Paper-faithful PPO implementation with CNN feature extraction
- Dynamic Risk Scanner**: Real-time monitoring of Sharpe ratio, volatility, and drawdown
- Professional Dashboard**: Live-streaming trading terminal with Plotly charts
- Smart Data Pipeline**: SQLite caching to overcome API rate limits
- Multi-Asset Support**: Crypto (ETH, BTC) and Indian equities (NIFTY 100)

---

Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRADING DASHBOARD (FastAPI)                  │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────────┐ │
│  │ Live     │  │ GAF Image    │  │ Risk Scanner Radar         │ │
│  │ Charts   │  │ Visualizer   │  │ (Drawdown, Sharpe, Vol)    │ │
│  └──────────┘  └──────────────┘  └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PPO TRAINING ENGINE                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ CNN Encoder  │──│ Actor-Critic │──│ Risk-Adjusted Reward │   │
│  │ (64x64 GAF)  │  │   Network    │  │      Function        │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA ACQUISITION PIPELINE                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  yfinance    │──│   SQLite     │──│   GAF Encoder        │   │
│  │  API         │  │   Cache      │  │   (pyts)             │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/jasonjamesp/ppo-backtest.git
cd ppo-backtest
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
python scripts/run_dashboard.py
```

Navigate to `http://localhost:8000` to view the live trading terminal.

### 3. Start Training

```bash
# Set PYTHONPATH for module resolution
# Windows PowerShell:
$env:PYTHONPATH = "$(pwd)"; python src/training/train.py

# Linux/Mac:
PYTHONPATH=$(pwd) python src/training/train.py
```

---

Project Structure

```
ppo-backtest/
├── data/
│   ├── cache/          # SQLite database for cached OHLCV data
│   ├── gaf_images/     # Generated GAF images (training artifacts)
│   └── raw/            # Raw CSV downloads
├── models/
│   ├── checkpoints/    # Model checkpoints during training
│   └── trained/        # Final trained models
├── src/
│   ├── data_pipeline/
│   │   ├── data_manager.py    # Multi-source data fetcher with caching
│   │   ├── data_validator.py  # OHLCV integrity checks
│   │   └── gaf_encoder.py     # Gramian Angular Field generator
│   ├── training/
│   │   ├── cnn_encoder.py     # CNN for GAF feature extraction
│   │   ├── ppo_agent.py       # PPO Actor-Critic implementation
│   │   ├── reward_function.py # Risk-adjusted reward logic
│   │   ├── trading_env.py     # Gymnasium environment
│   │   └── train.py           # Training orchestrator
│   ├── risk_management/
│   │   ├── risk_scanner.py    # Real-time risk monitoring
│   │   ├── portfolio_optimizer.py # Multi-asset allocation
│   │   └── alerts.py          # Notification system
│   ├── dashboard/
│   │   ├── app.py             # FastAPI backend
│   │   └── templates/
│   │       └── index.html     # Trading terminal UI
│   └── utils/
│       └── config.py          # Hyperparameters & settings
├── scripts/
│   ├── run_dashboard.py       # Dashboard launcher
│   └── smoke_test.py          # Pipeline verification
├── requirements.txt
└── README.md
```

---

Configuration

All hyperparameters are centralized in `src/utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPISODES` | 500,000 | Total training episodes |
| `LEARNING_RATE` | 3e-4 | Adam optimizer learning rate |
| `PPO_CLIP` | 0.2 | PPO clipping ratio |
| `GAF_WINDOW_SIZE` | 64 | Candles per GAF image |
| `MAX_DRAWDOWN_THRESHOLD` | 0.10 | Risk scanner drawdown trigger |
| `MIN_SHARPE_THRESHOLD` | 0.5 | Minimum Sharpe for position |

---

Risk Scanner

The "Gatekeeper" risk scanner continuously monitors portfolio health:

| Metric | Threshold | Action |
|--------|-----------|--------|
| **Max Drawdown** | > 10% | Exit position immediately |
| **Sharpe Ratio** | < 0.5 | Reduce position by 50% |
| **Volatility** | > 2x 20-day avg | Reduce position by 25% |
| **Win Rate** | < 40% (50+ trades) | Pause trading |

---

Supported Assets

### Crypto
- `ETH-USD` (Primary training asset per paper)
- `BTC-USD`

### Indian Equities (NIFTY 100)
- `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `ICICIBANK.NS`

---

Testing

Run the smoke test to verify the data pipeline:

```bash
python scripts/smoke_test.py
```

Expected output:
```
--- GAF-PPO Pipeline Smoke Test ---
Fetching ETH-USD data...
SUCCESS: Fetched 651 rows
Validating data...
SUCCESS: Data validated. Found 0 potential gaps.
Encoding sample window to GAF...
SUCCESS: GAF image generated successfully.

--- Pipeline Check PASSED ---
```

---

