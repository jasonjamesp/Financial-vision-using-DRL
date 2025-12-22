
import sys

print("Python executable:", sys.executable)

print("\n--- 1. Numpy ---")
try:
    import numpy
    print(f"Success. Version: {numpy.__version__}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- 2. Pandas ---")
try:
    import pandas
    print(f"Success. Version: {pandas.__version__}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- 3. OpenCV ---")
try:
    import cv2
    print(f"Success. Version: {cv2.__version__}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- 4. Mplfinance ---")
try:
    import mplfinance
    print(f"Success. Version: {mplfinance.__version__}")
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- 5. Stable Baselines 3 ---")
try:
    from stable_baselines3 import PPO
    print("Success. Imported PPO.")
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- 6. Backend ---")
try:
    from backend import MarketEngine
    print("Success. Imported MarketEngine.")
except Exception as e:
    print(f"FAILED: {e}")
