import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

class FinancialVisionEnv(gym.Env):
    """
    Custom Environment for 'Financial Vision-India' that implements the Gatekeeper Architecture.
    
    Action Space:
        0: HOLD
        1: BUY
        2: SELL
    
    Observation Space (Dict):
        - image: (64, 64, 1) grayscale candlestick chart
        - chips: Discrete(4) (Number of chips currently held: 0, 1, 2, 3)
        - pattern: Discrete(3) (Gatekeeper Signal: 0=Bearish(-1), 1=Neutral(0), 2=Bullish(1))
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir='./data', penalty_factor=0.1, ticker_id=None):
        super(FinancialVisionEnv, self).__init__()
        
        self.data_dir = data_dir
        self.penalty_factor = penalty_factor
        self.ticker_id = ticker_id
        
        # --- Define Spaces ---
        self.action_space = spaces.Discrete(3) # 0=Hold, 1=Buy, 2=Sell
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
            "chips": spaces.Discrete(4), # 0, 1, 2, 3
            # Pattern signals come as -1, 0, 1. We map them: -1->0, 0->1, 1->2 for Discrete(3)
            "pattern": spaces.Discrete(3) 
        })
        
        # --- Internal State ---
        self.current_step = 0
        self.max_steps = 0
        self.chips_held = 0
        self.balance = 0.0 # Not strictly "cash" but profit tracking accumulator
        self.portfolio_value = 0.0
        
        # --- Load Data ---
        self._load_data()

    def _load_data(self):
        """Loads observation images and metadata."""
        images_path = os.path.join(self.data_dir, 'obs_images.npy')
        metadata_path = os.path.join(self.data_dir, 'obs_metadata.npy')
        
        try:
            if os.path.exists(images_path) and os.path.exists(metadata_path):
                self.images = np.load(images_path)
                self.metadata_matrix = np.load(metadata_path)
                
                # Filter by Ticker ID if specified
                if self.ticker_id is not None:
                    # Metadata: [Ticker_ID, Gatekeeper_Signal, Market_Status_Flag, Close_Price]
                    mask = self.metadata_matrix[:, 0] == self.ticker_id
                    if np.sum(mask) == 0:
                        raise ValueError(f"No data found for Ticker ID {self.ticker_id}")
                    
                    self.images = self.images[mask]
                    self.metadata_matrix = self.metadata_matrix[mask]
                    print(f"Loaded {len(self.images)} steps for Ticker ID {self.ticker_id}.")
                else:
                    print(f"Loaded {len(self.images)} steps (All Tickers).")

            else:
                print(f"Warning: Data files not found in {self.data_dir}. Generating dummy data for testing.")
                # Ensure directory exists for potential future saves or just to be safe
                os.makedirs(self.data_dir, exist_ok=True)
                
                # Generating dummy data conforming to the spec
                num_samples = 200 # Increased to have enough for both
                self.images = np.random.randint(0, 256, (num_samples, 64, 64), dtype=np.uint8)
                
                # Metadata: [Ticker_ID, Gatekeeper_Signal, Market_Status_Flag, Close_Price]
                self.metadata_matrix = np.zeros((num_samples, 4))
                
                # Mix Ticker 0 (Reliance) and Ticker 5 (ETH)
                # First half Ticker 0, Second half Ticker 5
                half = num_samples // 2
                self.metadata_matrix[:half, 0] = 0 # RELIANCE
                self.metadata_matrix[half:, 0] = 5 # ETH-USD
                
                self.metadata_matrix[:, 1] = np.random.choice([-1, 0, 1], num_samples) # Patterns
                self.metadata_matrix[:, 2] = np.random.choice([0, 1], num_samples, p=[0.1, 0.9]) # Volatility (0=Lockdown, 1=Safe)
                self.metadata_matrix[:, 3] = np.linspace(100, 200, num_samples) # Prices
            
            self.max_steps = len(self.images) - 1
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise e

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.chips_held = 0
        self.balance = 0.0 # Reset profit tracker
        
        # Initial Portfolio Value
        initial_price = self.metadata_matrix[self.current_step, 3]
        self.portfolio_value = 0 # Starting relative value
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Image: Add channel dimension if needed
        # Assuming existing npy is (N, 64, 64), we need (64, 64, 1)
        img = self.images[self.current_step]
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            
        # Pattern: Map -1->0, 0->1, 1->2
        raw_pattern = int(self.metadata_matrix[self.current_step, 1])
        pattern_obs = raw_pattern + 1 
        
        return {
            "image": img,
            "chips": self.chips_held,
            "pattern": pattern_obs
        }

    def step(self, action):
        """
        Executes one time step within the environment with Gatekeeper Logic.
        """
        
        # 1. Get Current Context
        current_data = self.metadata_matrix[self.current_step]
        raw_pattern = int(current_data[1]) # -1, 0, 1
        volatility_flag = int(current_data[2]) # 0=Lockdown, 1=Safe
        current_price = current_data[3]
        
        explanation = ""
        penalty = 0.0
        applied_action = action
        
        # --- THE GATEKEEPER ---
        
        # Rule 3: Volatility Circuit Breaker
        # FIX: Market Status 0 means High Risk (Lockdown), 1 means Safe.
        if volatility_flag == 0: 
            if action == 1: # Attempted BUY
                penalty = 10.0
                explanation = "Action: BLOCKED (BUY) | Reason: Volatility Lockdown! Penalty applied."
            else:
                explanation = "Action: HOLD | Reason: Volatility Lockdown."
            
            applied_action = 0 # Force HOLD
            
        # Rule 1: The Pattern Gatekeeper (Only applies if not already locked down)
        elif raw_pattern == 0:
            if action != 0:
                explanation = f"Action: BLOCKED ({'BUY' if action==1 else 'SELL'}) | Reason: No Whitelisted Pattern (Signal=0)."
                applied_action = 0 # Force HOLD
            else:
                explanation = "Action: HOLD | Reason: Waiting for Pattern."
                
        # Rule 2: The Three-Chip Limit (Only checks if we are still trying to BUY)
        elif applied_action == 1: # BUY
            if self.chips_held >= 3:
                explanation = "Action: BLOCKED (BUY) | Reason: Max Chips (3) Reached."
                applied_action = 0 # Force HOLD
            else:
                explanation = f"Action: BUY | Reason: {self._get_pattern_name(raw_pattern)} detected + Volatility Safe."

        # Allow SELL if we have chips (and logic hasn't forced hold yet)
        if action == 2: # SELL
            if self.chips_held > 0 and applied_action == 2:
                 explanation = f"Action: SELL | Reason: {self._get_pattern_name(raw_pattern)} detected."
            elif self.chips_held == 0 and applied_action == 2:
                 # Can't sell if no chips, effectively HOLD but no specific penalty rule from prompt, just logic.
                 explanation = "Action: INVALID (SELL) | Reason: No Chips to sell."
                 applied_action = 0

        # --- EXECUTE ACTION ---
        # 0=Hold, 1=Buy, 2=Sell
        prev_portfolio_value = self._calculate_portfolio_value(current_price)
        
        if applied_action == 1: # BUY
            self.chips_held += 1
        elif applied_action == 2: # SELL
            self.chips_held -= 1
            
        # Advance Step
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # --- CALCULATE REWARD ---
        if not terminated:
            next_price = self.metadata_matrix[self.current_step, 3]
            # Return = Change in value of held assets
            price_change = next_price - current_price
            step_return = price_change * self.chips_held
            
            # Derived Reward
            # Use volatility flag (0=Lockdown) -> Penalty if 0
            vol_penalty = 1.0 * self.penalty_factor if volatility_flag == 0 else 0.0
            
            reward = step_return - vol_penalty - penalty
            
            # Logging
            if explanation: 
                 pass
                
        else:
            reward = 0
            
        info = {
            "explanation": explanation,
            "portfolio_value": self._calculate_portfolio_value(current_price), 
            "chips": self.chips_held,
            "volatility_flag": volatility_flag # For verification logging
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _calculate_portfolio_value(self, price):
        return self.chips_held * price

    def _get_pattern_name(self, code):
        mapping = {
            -1: "Bearish Pattern",
            0: "No Pattern",
            1: "Bullish Pattern"
        }
        return mapping.get(code, "Unknown")
