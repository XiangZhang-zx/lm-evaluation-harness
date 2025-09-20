"""
GSM8K Curated Noise dataset loader
"""

import json
import os
from typing import Dict, Any
from datasets import Dataset


def create_dataset() -> Dict[str, Dataset]:
    """Create the GSM8K curated noise dataset"""
    
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, "test.jsonl")
    
    # Load the test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line.strip()))
    
    # Create datasets
    datasets = {
        "test": Dataset.from_list(test_data)
    }
    
    return datasets


# This function will be called by the lm-evaluation-harness
def load_dataset(dataset_name: str = "main", **kwargs) -> Dict[str, Dataset]:
    """Load the GSM8K curated noise dataset"""
    return create_dataset()
