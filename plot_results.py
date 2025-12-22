import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_results():
    log_file = os.path.join('data', 'logs', 'phase3_training_log.csv')
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    # Skip first line if it contains metadata (Monitor wrapper often adds 2 headers)
    # usually: # {json metadata} \n r,l,t
    try:
        df = pd.read_csv(log_file, skiprows=1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 'r' is reward, 'l' is length, 't' is time
    if 'r' not in df.columns:
        print("Columns 'r' (reward) not found. Check CSV format.")
        print(df.head())
        return

    # Calculate Rolling Mean (Window = 100 episodes)
    window = 100
    df['rolling_reward'] = df['r'].rolling(window=window).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df['r'], label='Episode Reward', alpha=0.3, color='gray')
    plt.plot(df['rolling_reward'], label=f'Rolling Mean ({window})', color='blue', linewidth=2)
    
    plt.title('Training Performance: ETH-USD (Gatekeeper)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join('data', 'logs', 'training_plot.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    # plt.show() # Uncomment to show interactively

if __name__ == "__main__":
    plot_training_results()
