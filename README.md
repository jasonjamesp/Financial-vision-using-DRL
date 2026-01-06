# 📈 GAF-PPO Financial Vision Trading System

A production-grade algorithmic trading platform that combines **Gramian Angular Field (GAF)** image encoding with **Proximal Policy Optimization (PPO)** reinforcement learning for intelligent, risk-aware trading decisions.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 Overview

This system implements the "Financial Vision" approach from academic research, treating market data as images rather than time series. By encoding price movements into **Gramian Angular Field** images, the PPO agent can leverage convolutional neural networks to identify complex visual patterns that traditional quantitative methods might miss.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Financial Vision** | Converts 64-candle windows into 64×64 GAF images |
| 🧠 **Hybrid Multi-Input RL** | CNN + Technical Indicators + Portfolio State fusion |
| 📊 **Professional Backtesting** | Sharpe, Sortino, Max Drawdown, Equity Curves |
| 🔴 **Real-time Dashboard** | WebSocket-powered trading terminal with Plotly charts |
| 📈 **Technical Indicators** | RSI, MACD, Bollinger Bands, ATR, OBV, ADX, Stochastic, EMAs |
| ⚠️ **Dynamic Risk Scanner** | Automatic position sizing based on drawdown and volatility |
| 💾 **Smart Data Pipeline** | SQLite caching to overcome yfinance API limits |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          TRADING DASHBOARD (FastAPI)                          │
│  ┌─────────────┐  ┌────────────────────┐  ┌───────────────────────────────┐  │
│  │ Live Charts │  │ Backtest Results   │  │ Risk Scanner Radar            │  │
│  │ (Plotly.js) │  │ (Modal + Metrics)  │  │ (Drawdown, Sharpe, Volatility)│  │
│  └─────────────┘  └────────────────────┘  └───────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │ WebSocket + REST API
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          PPO TRAINING ENGINE                                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────────────────┐  │
│  │ CNN Encoder │──│  Actor-Critic    │──│ Risk-Adjusted Reward Function  │  │
│  │  (64×64)    │  │ (608-dim fusion) │  │ (Sharpe + Drawdown penalty)    │  │
│  └─────────────┘  └──────────────────┘  └─────────────────────────────────┘  │
│         ▲                  ▲                                                  │
│         │                  │                                                  │
│  ┌──────┴──────┐  ┌───────┴────────┐                                         │
│  │ GAF Image   │  │ Tech Indicators │                                        │
│  │ (pyts)      │  │ (12-dim vector) │                                        │
│  └─────────────┘  └────────────────┘                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DATA ACQUISITION PIPELINE                             │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────────────────┐  │
│  │ yfinance    │──│ SQLite Cache     │──│ Indicator Engine                │  │
│  │ (60d limit) │  │ (market_data.db) │  │ (RSI, MACD, BB, ATR, OBV, ADX) │  │
│  └─────────────┘  └──────────────────┘  └─────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA (optional, for GPU acceleration)
- 4GB RAM minimum

### 1. Clone & Install

```bash
git clone https://github.com/jasonjamesp/Financial-vision-using-DRL.git
cd Financial-vision-using-DRL
pip install -r requirements.txt
pip install -e .  # Install as editable package
```

### 2. Launch the Dashboard

```bash
python scripts/run_dashboard.py
```

Navigate to `http://localhost:8000` to view the trading terminal.

### 3. Run a Backtest

1. Open the dashboard
2. Click **BACKTEST** in the Training Panel
3. View equity curve, drawdown chart, and performance metrics

### 4. Start Training

```bash
python scripts/start_training.py
```

---

## 📂 Project Structure

```
Financial-vision-using-DRL/
├── 📁 config/                      # JSON configuration files
│   ├── shared_config.json          # Common settings
│   ├── trading_config.json         # Live trading parameters
│   └── training_config.json        # Training hyperparameters
│
├── 📁 data/
│   ├── cache/                      # SQLite database for OHLCV
│   ├── gaf_images/                 # Generated GAF images
│   └── raw/                        # CSV backups
│
├── 📁 models/
│   ├── checkpoints/                # Model checkpoints during training
│   └── trained/                    # Final trained models
│
├── 📁 scripts/
│   ├── run_dashboard.py            # Launch FastAPI dashboard
│   ├── start_training.py           # Start PPO training
│   ├── start_live_test.py          # Run paper trading
│   └── smoke_test.py               # Verify data pipeline
│
├── 📁 src/
│   ├── 📁 data_pipeline/           # Data acquisition & processing
│   │   ├── data_manager.py         # yfinance fetcher + SQLite cache
│   │   ├── data_validator.py       # OHLCV integrity checks
│   │   └── gaf_encoder.py          # Gramian Angular Field generator
│   │
│   ├── 📁 features/                # Feature engineering
│   │   ├── indicators.py           # Technical indicators (8 types)
│   │   └── feature_builder.py      # ML feature vector (12 dims)
│   │
│   ├── 📁 training/                # RL training components
│   │   ├── ppo_agent.py            # PPO Actor-Critic agent
│   │   ├── cnn_encoder.py          # CNN for GAF feature extraction
│   │   ├── trading_env.py          # Gymnasium environment
│   │   ├── reward_function.py      # Risk-adjusted rewards
│   │   ├── backtester.py           # Historical simulation
│   │   ├── walk_forward.py         # Rolling window validation
│   │   └── train.py                # Training orchestrator
│   │
│   ├── 📁 testing/                 # Paper trading
│   │   ├── live_test.py            # Real-time trading loop
│   │   └── trading_logger.py       # Trade logging
│   │
│   ├── 📁 risk_management/         # Risk controls
│   │   ├── risk_scanner.py         # Portfolio health monitor
│   │   └── alerts.py               # Notification system
│   │
│   ├── 📁 dashboard/               # Web UI
│   │   ├── app.py                  # FastAPI backend
│   │   └── templates/index.html    # Trading terminal UI
│   │
│   └── 📁 utils/
│       └── config.py               # Hyperparameters & constants
│
├── 📁 tests/                       # Verification tests
│   ├── test_indicators.py
│   ├── test_backtester.py
│   └── test_api_backtest.py
│
├── requirements.txt
└── README.md
```

---

## 🧠 Technical Deep Dive

### Observation Space

The PPO agent receives a **hybrid multi-input observation**:

| Component | Shape | Description |
|-----------|-------|-------------|
| `image` | `(1, 64, 64)` | Gramian Angular Field of last 64 candles |
| `indicators` | `(12,)` | Normalized technical indicator vector |
| `state` | `(5,)` | `[balance, shares, net_worth, drawdown, pnl_pct]` |

### Action Space

| Action | ID | Description |
|--------|-----|-------------|
| Hold | 0 | No position change |
| Buy All | 1 | Convert all cash to shares |
| Sell All | 2 | Liquidate all shares |

### Technical Indicators

| Indicator | Period | Use Case |
|-----------|--------|----------|
| RSI | 14 | Overbought/Oversold detection |
| MACD | 12/26/9 | Trend momentum |
| Bollinger Bands | 20, 2σ | Volatility bands |
| ATR | 14 | Volatility measure |
| OBV | - | Volume momentum |
| ADX | 14 | Trend strength |
| Stochastic | 14, 3 | Momentum oscillator |
| EMA | 9, 21, 50 | Trend direction |

### Neural Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ActorCritic Network                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GAF Image (1,64,64)                                         │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────┐                                             │
│  │ CNNEncoder  │──────────────────────────┐                  │
│  │ Conv→Pool×4 │                          │                  │
│  │ Output: 512 │                          │                  │
│  └─────────────┘                          │                  │
│                                           │                  │
│  Indicators (12)                          │                  │
│       │                                   │                  │
│       ▼                                   ▼                  │
│  ┌─────────────┐                   ┌──────────────┐          │
│  │ Linear(64)  │──────────────────▶│ Concatenate  │          │
│  │ ReLU        │                   │    (608)     │          │
│  └─────────────┘                   └──────┬───────┘          │
│                                           │                  │
│  State (5)                                │                  │
│       │                                   │                  │
│       ▼                                   ▼                  │
│  ┌─────────────┐                   ┌──────────────┐          │
│  │ Linear(32)  │──────────────────▶│ Actor Head   │──▶ π(a)  │
│  │ ReLU        │                   │ Linear(256)  │          │
│  └─────────────┘                   │ Softmax(3)   │          │
│                                    └──────────────┘          │
│                                    ┌──────────────┐          │
│                                    │ Critic Head  │──▶ V(s)  │
│                                    │ Linear(256)  │          │
│                                    │ Linear(1)    │          │
│                                    └──────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

All hyperparameters are centralized in `src/utils/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPISODES` | 500,000 | Total training episodes |
| `LEARNING_RATE` | 3e-4 | Adam optimizer learning rate |
| `PPO_CLIP` | 0.2 | PPO clipping ratio |
| `GAF_WINDOW_SIZE` | 64 | Candles per GAF image |
| `IMAGE_SIZE` | (64, 64) | GAF image dimensions |
| `MAX_DRAWDOWN_THRESHOLD` | 0.10 | Risk scanner trigger |
| `MIN_SHARPE_THRESHOLD` | 0.5 | Minimum Sharpe for positions |

---

## 🛡️ Risk Scanner

The "Gatekeeper" risk scanner continuously monitors portfolio health:

| Metric | Threshold | Action |
|--------|-----------|--------|
| **Max Drawdown** | > 10% | Exit position immediately |
| **Sharpe Ratio** | < 0.5 | Reduce position by 50% |
| **Volatility** | > 2× 20-day avg | Reduce position by 25% |
| **Win Rate** | < 40% (50+ trades) | Pause trading |

---

## 📊 Backtesting

The backtesting engine provides professional-grade performance analysis:

### Performance Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Sharpe Ratio** | (μ - rf) / σ × √252 | Risk-adjusted return |
| **Sortino Ratio** | (μ - rf) / σ_down × √252 | Downside risk-adjusted return |
| **Max Drawdown** | max(peak - trough) / peak | Worst peak-to-trough decline |
| **Win Rate** | # wins / # trades | Trade success percentage |

### Running a Backtest

Via Dashboard:
1. Open `http://localhost:8000`
2. Click **BACKTEST** button
3. View results in the modal overlay

Via API:
```bash
curl "http://localhost:8000/api/training/backtest/run?asset=ETH-USD&days=30"
```

---

## 📈 Supported Assets

### Cryptocurrencies
- `ETH-USD` (Primary training asset per research paper)
- `BTC-USD`

### Indian Equities (NIFTY 100)
- `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `ICICIBANK.NS`

---

## 🧪 Testing

### Smoke Test (Pipeline Verification)
```bash
python scripts/smoke_test.py
```

### Unit Tests
```bash
# Set PYTHONPATH and run tests
$env:PYTHONPATH = "."
python tests/test_indicators.py
python tests/test_backtester.py
```

---

## 📚 References

This implementation is based on:

1. **Financial Vision Paper**: Using Gramian Angular Fields to encode time-series as images for RL trading
2. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
3. **Nature DQN CNN**: Mnih et al., "Human-level control through deep RL" (2015)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **Repository**: [GitHub](https://github.com/jasonjamesp/Financial-vision-using-DRL)
- **Documentation**: See `.agent/ARCHITECTURE_REFERENCE.md` for internal architecture details
- **Research Paper**: `Financial_Vision-Based_Reinforcement_Learning_Trad.pdf`

---

**Built with 🧠 by [Jason James P](https://github.com/jasonjamesp)**
