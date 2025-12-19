"""
Script to create and save a preprocessor object for testing
This creates a preprocessor.pkl file from existing components
"""

import pickle
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.preprocess_production import ProductionDataPreprocessor

def create_preprocessor(processor_dir='notebooks/processors'):
    """Create and save preprocessor object"""
    print("=" * 80)
    print("CREATING PREPROCESSOR OBJECT")
    print("=" * 80)
    
    # Create preprocessor instance
    preprocessor = ProductionDataPreprocessor()
    
    # Load existing processors
    success = preprocessor.load_processors(processor_dir)
    
    if success:
        # Save the preprocessor object
        preprocessor_path = Path(processor_dir) / 'preprocessor.pkl'
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"\n✅ Preprocessor saved to: {preprocessor_path}")
        return True
    else:
        print("\n❌ Failed to create preprocessor - missing components")
        return False

if __name__ == "__main__":
    processor_dir = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/processors'
    create_preprocessor(processor_dir)
