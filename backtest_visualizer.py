import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from gatekeeper_env import FinancialVisionEnv
import os

def run_backtest():
    print("--- Starting Backtest on RELIANCE.NS (Ticker 0) ---")
    
    # 1. Setup
    # Ticker 0 = Reliance
    env = FinancialVisionEnv(data_dir='./data', ticker_id=0)
    model_path = os.path.join("models", "eth_gatekeeper_final.zip")
    
    if not os.path.exists(model_path):
        print("Model not found!")
        return
        
    model = PPO.load(model_path)
    
    # Paper Wallet
    initial_capital = 100000.0
    cash = initial_capital
    chip_lots = [] # List of share counts per chip: [shares_chip1, shares_chip2]
    
    history = []
    
    obs, info = env.reset()
    done = False
    step = 0
    
    # Metrics
    equity_curve = []
    
    while not done:
        # Predict
        action, _ = model.predict(obs) 

        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Detect Execution
        prev_chips = obs['chips']
        curr_chips = next_obs['chips'] # Actual chips held after Gatekeeper
        
        # Get Price at which step occurred
        # env.current_step has been incremented by 1 in step()
        current_price = env.metadata_matrix[env.current_step-1, 3]
        
        trade_event = None
        
        # Rules:
        # 1 Chip = 33000
        chip_allocation = 33000.0
        
        if curr_chips > prev_chips:
            # BUY
            if cash >= chip_allocation:
                shares_bought = chip_allocation / current_price
                cost = shares_bought * current_price
                cash -= cost
                chip_lots.append(shares_bought)
                trade_event = 'BUY'
            else:
                 # Logic if not enough cash? In real env blocked. 
                 # Here we assume chips imply execution success from Env perspective.
                 # If Env says we have the chip, we MUST have bought it.
                 # We force the purchase (Margin/Debt) or assume cash refill for this simulation 
                 # if we strictly follow "Env State".
                 # But let's stick to valid cash. If invalid, maybe skip?
                 # But then wallet drifts from Env. 
                 # Let's force it to keep sync.
                shares_bought = chip_allocation / current_price
                cost = shares_bought * current_price
                cash -= cost
                chip_lots.append(shares_bought)
                trade_event = 'BUY'
                
        elif curr_chips < prev_chips:
            # SELL
            if len(chip_lots) > 0:
                shares_sold = chip_lots.pop() # LIFO
                proceeds = shares_sold * current_price
                cash += proceeds
                trade_event = 'SELL'
            else:
                # Ghost chip? Should not happen if sync
                pass
                
        # Calc Equity
        # Sum of shares in lots
        total_shares = sum(chip_lots)
        portfolio_value = total_shares * current_price
        total_equity = cash + portfolio_value
        equity_curve.append(total_equity)
        
        history.append({
            'step': step,
            'price': current_price,
            'chips': curr_chips,
            'equity': total_equity,
            'event': trade_event
        })
        
        obs = next_obs
        step += 1
        
    # --- Visualization ---
    df = pd.DataFrame(history)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Price (Left)
    ax1.plot(df['step'], df['price'], label='RELIANCE.NS', color='blue', alpha=0.6)
    ax1.set_ylabel('Stock Price (\u20b9)', color='blue') # Rupee symbol might fail in some fonts, stick to 'Rs' or symbol unicode
    ax1.set_xlabel('Step')
    
    # Markers
    buys = df[df['event'] == 'BUY']
    sells = df[df['event'] == 'SELL']
    
    if not buys.empty:
        ax1.scatter(buys['step'], buys['price'], marker='^', color='green', s=100, label='BUY', zorder=5)
    if not sells.empty:
        ax1.scatter(sells['step'], sells['price'], marker='v', color='red', s=100, label='SELL', zorder=5)
    
    # Chips (Right)
    ax2 = ax1.twinx()
    ax2.step(df['step'], df['chips'], color='orange', linestyle='--', alpha=0.5, label='Chips Held')
    ax2.set_ylabel('Chips (0-3)', color='orange')
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_yticks([0, 1, 2, 3])
    
    plt.title('Financial Vision: Reliance Backtest (Trained on ETH)')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    plt.grid(True, alpha=0.3)
    
    output_path = 'backtest_result.png'
    plt.savefig(output_path)
    print(f"Backtest chart saved to {output_path}")
    
    # --- Results ---
    final_equity = equity_curve[-1] if equity_curve else initial_capital
    roi = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Drawdown
    equity_arr = np.array(equity_curve)
    if len(equity_arr) > 0:
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak
        max_dd = drawdown.min() * 100
    else:
        max_dd = 0.0
    
    print("\n--- BACKTEST RESULTS ---")
    print(f"Initial Capital: Rs {initial_capital:,.2f}")
    print(f"Final Equity:    Rs {final_equity:,.2f}")
    print(f"Return on Invest: {roi:.2f}%")
    print(f"Max Drawdown:    {max_dd:.2f}%")

if __name__ == "__main__":
    run_backtest()
