import numpy as np
import pandas as pd
from pyts.image import GramianAngularField
from src.utils.config import GAF_WINDOW_SIZE, IMAGE_SIZE
import matplotlib.pyplot as plt
import os

class GAFEncoder:
    def __init__(self, image_size=IMAGE_SIZE[0], method='summation'):
        """
        method: 'summation' (GASF) or 'difference' (GADF)
        """
        self.gaf = GramianAngularField(image_size=image_size, method=method)

    def encode(self, data: np.array) -> np.array:
        """
        Encode a 1D time series into a GAF image.
        data: numpy array of shape (n_timestamps,)
        returns: numpy array of shape (image_size, image_size)
        """
        if len(data) < GAF_WINDOW_SIZE:
            # Pad with zeros if data is too short
            pad_width = GAF_WINDOW_SIZE - len(data)
            data = np.pad(data, (pad_width, 0), 'constant')
        
        # Reshape for pyts (n_samples, n_timestamps)
        data_reshaped = data.reshape(1, -1)
        image = self.gaf.fit_transform(data_reshaped)
        
        return image[0]

    def generate_sequence(self, df: pd.DataFrame, target_col: str = 'close') -> list:
        """
        Generate a sequence of GAF images from a sliding window.
        """
        images = []
        series = df[target_col].values
        
        for i in range(GAF_WINDOW_SIZE, len(series)):
            window = series[i-GAF_WINDOW_SIZE:i]
            # Normalize window to [-1, 1] for GAF
            # min_val = window.min()
            # max_val = window.max()
            # if max_val > min_val:
            #     window_norm = 2 * (window - min_val) / (max_val - min_val) - 1
            # else:
            #     window_norm = window * 0
            
            # GAF handles normalization if configured, but manual scaling often helps
            img = self.encode(window)
            images.append(img)
            
        return images

    def save_image(self, image: np.array, path: str):
        plt.imsave(path, image, cmap='rainbow', origin='lower')

if __name__ == "__main__":
    # Test encoding
    encoder = GAFEncoder()
    mock_data = np.sin(np.linspace(0, 10, 64))
    img = encoder.encode(mock_data)
    print(f"GAF image shape: {img.shape}")
    
    # Save test image
    if not os.path.exists("test_gaf.png"):
        encoder.save_image(img, "test_gaf.png")
        print("Test GAF saved as test_gaf.png")
