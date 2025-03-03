"""
Simple test to verify that imports from mmm modules work correctly.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_preprocessing_imports():
    """Test that preprocessing module imports work."""
    try:
        import mmm.preprocessing
        print("Successfully imported mmm.preprocessing")
        print("Available functions and classes:")

        # List all functions in the module
        for name in dir(mmm.preprocessing):
            if not name.startswith("_"):  # Skip private/internal items
                print(f"  - {name}")

    except ImportError as e:
        pytest.fail(f"Failed to import mmm.preprocessing: {str(e)}")


def test_adstock_imports():
    """Test that adstock module imports work."""
    try:
        import mmm.adstock
        print("Successfully imported mmm.adstock")
        print("Available functions and classes:")

        # List all functions in the module
        for name in dir(mmm.adstock):
            if not name.startswith("_"):  # Skip private/internal items
                print(f"  - {name}")

    except ImportError as e:
        pytest.fail(f"Failed to import mmm.adstock: {str(e)}")


if __name__ == "__main__":
    print(f"Python path: {sys.path}")
    test_preprocessing_imports()
    test_adstock_imports()
