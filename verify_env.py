"""
Environment Verification Script (Clean)
Checks if the environment is correctly set up for PyTorch 2.9 and DA3 (with xformers).
"""
import sys
import os
import importlib.util

def check_package(name):
    spec = importlib.util.find_spec(name)
    return spec is not None

print("=" * 60)
print("Environment Verification (Clean Setup)")
print("=" * 60)

# 1. Check PyTorch
print("\n[1/3] Checking PyTorch...")
try:
    import torch
    print(f"[OK] PyTorch version: {torch.__version__}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[OK] CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"[OK] CUDA version: {torch.version.cuda}")
    
    # Warning if not 2.9
    if not torch.__version__.startswith("2.9"):
        print("! Warning: PyTorch version is not 2.9.x (Expected for xformers 0.0.33)")
except ImportError:
    print("[FAIL] PyTorch not found!")
    sys.exit(1)

# 2. Check xformers status
print("\n[2/3] Checking xformers status...")
has_xformers = check_package("xformers")
if has_xformers:
    print("[OK] xformers package IS installed.")
    try:
        import xformers
        print(f"[OK] xformers version: {xformers.__version__}")
        # Try importing ops to ensure binary compatibility
        from xformers import ops
        print("[OK] xformers.ops imported successfully.")
    except Exception as e:
        print(f"[FAIL] xformers import failed: {e}")
else:
    print("! Warning: xformers package is NOT installed. Performance might be lower.")

# 3. Check DA3 Model Loading (Dry Run)
print("\n[3/3] Checking DA3 Model Loading...")
sys.path.insert(0, os.path.abspath("."))
try:
    from da3_model import DA3Model
    # Dry run initialization only
    print("  Attempting to initialize DA3Model class...")
    # We don't load the heavy weights here, just check import and class definition
    print("[OK] DA3Model class imported successfully.")
    
    # Check if our patch logic is GONE
    import inspect
    source = inspect.getsource(DA3Model._load_model)
    if "sys.modules['xformers'] = None" in source:
        print("! Warning: xformers disabling logic STILL detected in DA3Model (Should be removed).")
    else:
        print("[OK] xformers disabling logic NOT detected (Clean implementation).")
        
except ImportError as e:
    print(f"[FAIL] Failed to import DA3Model: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Verification Complete")
print("=" * 60)
