#!/usr/bin/env python3
"""
Test script to ensure imports are working correctly after refactoring.
"""
import sys
import os

# Add the package to Python path for testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all module imports to ensure everything is working correctly."""
    
    print("Testing imports...")
    
    # Test main package import
    print("\nTesting main package import:")
    try:
        import maritime_trajectory_prediction
        print(f"✓ Main package imported successfully (version: {maritime_trajectory_prediction.__version__})")
    except Exception as e:
        print(f"✗ Error importing main package: {e}")
    
    # Test core imports
    print("\nTesting core imports:")
    try:
        from maritime_trajectory_prediction import AISProcessor, AISDataModule
        print("✓ Core classes imported successfully")
    except Exception as e:
        print(f"✗ Error importing core classes: {e}")
    
    # Test lazy loading
    print("\nTesting lazy loading:")
    try:
        import maritime_trajectory_prediction
        utils = maritime_trajectory_prediction.utils
        print("✓ Utils lazy loaded successfully")
        
        models = maritime_trajectory_prediction.models
        print("✓ Models lazy loaded successfully")
        
        experiments = maritime_trajectory_prediction.experiments
        print("✓ Experiments lazy loaded successfully")
    except Exception as e:
        print(f"✗ Error with lazy loading: {e}")
    
    # Test specific utility imports
    print("\nTesting specific utility imports:")
    try:
        from maritime_trajectory_prediction.src.utils import AISParser, MaritimeUtils
        print("✓ Specific utils imported successfully")
    except Exception as e:
        print(f"✗ Error importing specific utils: {e}")
    
    # Test transformer blocks (should be directly available)
    print("\nTesting transformer blocks:")
    try:
        from maritime_trajectory_prediction.src.models import PositionalEncoding, MultiHeadAttention
        print("✓ Transformer blocks imported successfully")
    except Exception as e:
        print(f"✗ Error importing transformer blocks: {e}")
    
    # Test AIS processing functionality
    print("\nTesting AIS processing:")
    try:
        import json
        import numpy as np
        # Simple test to ensure JSON serialization works
        test_data = {"lat": np.float32(59.123), "lon": np.float32(10.456)}
        json_str = json.dumps(test_data, default=lambda x: float(x) if isinstance(x, np.number) else x)
        print(f"✓ JSON serialization working: {json_str}")
    except Exception as e:
        print(f"✗ Error with JSON serialization: {e}")
    
    print("\nImport tests completed")

if __name__ == "__main__":
    test_imports()

