import numpy as np
import os

def generate_and_save_data():
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating dummy data for Phase 4 Scanner...")
    num_samples = 200
    images = np.random.randint(0, 256, (num_samples, 64, 64), dtype=np.uint8)
    
    # Metadata: [Ticker_ID, Gatekeeper_Signal, Market_Status_Flag, Close_Price]
    metadata_matrix = np.zeros((num_samples, 4))
    
    # Mix Ticker 0 (Reliance) and Ticker 5 (ETH)
    half = num_samples // 2
    metadata_matrix[:half, 0] = 0 # RELIANCE
    metadata_matrix[half:, 0] = 5 # ETH-USD
    
    metadata_matrix[:, 1] = np.random.choice([-1, 0, 1], num_samples) # Patterns
    metadata_matrix[:, 2] = np.random.choice([0, 1], num_samples, p=[0.1, 0.9]) # Volatility
    metadata_matrix[:, 3] = np.linspace(100, 200, num_samples) # Prices
    
    np.save(os.path.join(data_dir, 'obs_images.npy'), images)
    np.save(os.path.join(data_dir, 'obs_metadata.npy'), metadata_matrix)
    print("Dummy data saved to ./data/")

if __name__ == "__main__":
    generate_and_save_data()
