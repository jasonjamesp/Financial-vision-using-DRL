import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
GAF_IMAGE_DIR = DATA_DIR / "gaf_images"
MODEL_DIR = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Data Configuration
CANDLE_INTERVAL = "15m"
GAF_WINDOW_SIZE = 64
IMAGE_SIZE = (64, 64)

# Assets
CRYPTO_ASSETS = ["ETH-USD", "BTC-USD"]
# Sample NIFTY stocks for initial implementation
NIFTY_ASSETS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# Training Hyperparameters
EPISODES = 500_000
STEPS_PER_EPISODE = 1000
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
BATCH_SIZE = 64
EPOCHS_PER_UPDATE = 10
CHECKPOINT_INTERVAL = 10_000

# Risk Management
MAX_DRAWDOWN_THRESHOLD = 0.10
MIN_SHARPE_THRESHOLD = 0.5
RISK_FREE_RATE = 0.03  # Annualized

# API Keys
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
