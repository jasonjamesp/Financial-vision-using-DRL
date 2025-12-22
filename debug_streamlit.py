
import sys
print(f"Python: {sys.executable}")

try:
    import numpy
    print(f"Numpy: {numpy.__version__}")
except ImportError as e:
    print(f"Numpy Import Failed: {e}")

print("Attempting to import streamlit...")
try:
    import streamlit
    print(f"Streamlit Version: {streamlit.__version__}")
except Exception as e:
    print(f"Streamlit Import FAILED: {e}")
    import traceback
    traceback.print_exc()

print("Attempting to import pandas...")
try:
    import pandas
    print(f"Pandas Version: {pandas.__version__}")
except Exception as e:
    print(f"Pandas Import FAILED: {e}")
