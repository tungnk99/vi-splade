#!/usr/bin/env python3
"""
Example usage of VNLegalDataset with get_dataset() method

Shows how to use the new simplified API that returns Hugging Face Datasets.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset.vn_legal_dataset import VNLegalDataset


def main():
    """Demonstrate the new get_dataset() API."""
    
    print("🚀 === VN LEGAL DATASET - NEW API DEMO ===")
    print()
    
    # Initialize dataset
    print("📚 Loading VN Legal Dataset...")
    dataset = VNLegalDataset(
        corpus_path="data/vn-legal-doc/cleaned_corpus_data.csv",
        test_path="data/vn-legal-doc/test_data.csv"
        # train_path not provided, so train will be empty
    )
    print()
    
    # Get datasets for each split - TWO WAYS!
    print("🔍 Getting datasets for each split...")
    
    # Method 1: Using get_dataset() method
    print("   Method 1: dataset.get_dataset(split)")
    test_dataset = dataset.get_dataset("test")
    print(f"   ✅ Test dataset: {test_dataset.num_rows:,} samples")
    
    # Method 2: Using bracket notation (NEW!)
    print("   Method 2: dataset[split] (NEW!)")
    train_dataset = dataset["train"]  # Using bracket notation!
    dev_dataset = dataset["dev"]      # Using bracket notation!
    print(f"   ✅ Train dataset: {train_dataset.num_rows:,} samples") 
    print(f"   ✅ Dev dataset: {dev_dataset.num_rows:,} samples")
    print()
    
    # Show dataset features
    print("🏗️ Dataset Features:")
    print(f"   - Features: {test_dataset.features}")
    print(f"   - Column names: {test_dataset.column_names}")
    print()
    
    # Show sample data
    print("📋 Sample Data:")
    if test_dataset.num_rows > 0:
        sample = test_dataset[0]
        print(f"   - Query: {sample['query'][:100]}...")
        print(f"   - Document: {sample['document'][:100]}...")
        print()
    
    # Demonstrate dataset operations
    print("⚙️ Dataset Operations:")
    
    # Take first 5 samples
    small_dataset = test_dataset.select(range(5))
    print(f"   - First 5 samples: {small_dataset.num_rows} samples")
    
    # Filter by query length
    long_queries = test_dataset.filter(lambda x: len(x['query']) > 100)
    print(f"   - Long queries (>100 chars): {long_queries.num_rows:,} samples")
    
    # Map function to add query length
    with_lengths = test_dataset.map(lambda x: {"query_length": len(x['query'])})
    print(f"   - Added query_length column: {with_lengths.column_names}")
    print()
    
    # Show usage examples
    print("💡 Usage Examples:")
    print("```python")
    print("from src.dataset.vn_legal_dataset import VNLegalDataset")
    print()
    print("# Initialize dataset")
    print("dataset = VNLegalDataset(corpus_path, test_path)")
    print()
    print("# Get Hugging Face datasets - TWO WAYS!")
    print("# Method 1: Using get_dataset()")
    print("test_dataset = dataset.get_dataset('test')")
    print("train_dataset = dataset.get_dataset('train')")
    print()
    print("# Method 2: Using bracket notation (cleaner!)")
    print("test_dataset = dataset['test']")
    print("train_dataset = dataset['train']")
    print("dev_dataset = dataset['dev']")
    print()
    print("# Use with DataLoader")
    print("from torch.utils.data import DataLoader")
    print("dataloader = DataLoader(test_dataset, batch_size=32)")
    print()
    print("# Use with Transformers")
    print("from transformers import AutoTokenizer")
    print("tokenizer = AutoTokenizer.from_pretrained('model-name')")
    print("tokenized = test_dataset.map(lambda x: tokenizer(x['query']))")
    print("```")
    print()
    
    print("🎉 === DEMO COMPLETED ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
