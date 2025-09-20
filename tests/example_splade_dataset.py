#!/usr/bin/env python3
"""
Example usage of SPLADE Dataset for Vietnamese Legal Documents

This script demonstrates how to use the SPLADE dataset classes
for training and evaluation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from dataset.splade_dataset import (
        VNLegalCorpusDataset, 
        VNLegalQADataset, 
        VNLegalSpladeDataset,
        create_splade_dataloaders
    )
    import torch
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Please install required packages:")
    print("   pip install torch transformers")
    sys.exit(1)


class SimpleTokenizer:
    """Simple tokenizer for demonstration purposes."""
    
    def __init__(self, vocab_size=10000, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
    
    def __call__(self, text, **kwargs):
        """Tokenize text into simple token IDs."""
        # Simple tokenization: split by space and convert to hash-based IDs
        tokens = text.split()
        
        # Convert tokens to IDs (simple hash-based approach)
        input_ids = [abs(hash(token)) % self.vocab_size for token in tokens]
        attention_mask = [1] * len(input_ids)
        
        # Apply max_length constraint
        max_len = kwargs.get('max_length', self.max_length)
        if kwargs.get('truncation', False):
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
        
        # Apply padding
        if kwargs.get('padding') == 'max_length':
            while len(input_ids) < max_len:
                input_ids.append(0)  # Padding token
                attention_mask.append(0)  # Mask padding
        
        # Return as tensors if requested
        if kwargs.get('return_tensors') == 'pt':
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.tensor([attention_mask])
            }
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def test_corpus_dataset(corpus_path: str, tokenizer):
    """Test the corpus dataset."""
    print("📚 Testing Corpus Dataset...")
    
    try:
        # Create corpus dataset
        corpus_dataset = VNLegalCorpusDataset(
            corpus_path=corpus_path,
            tokenizer=tokenizer,
            max_length=256
        )
        
        print(f"✅ Corpus loaded: {len(corpus_dataset):,} documents")
        
        # Test a single sample
        sample = corpus_dataset[0]
        print(f"✅ Sample keys: {list(sample.keys())}")
        print(f"   - input_ids shape: {sample['input_ids'].shape}")
        print(f"   - attention_mask shape: {sample['attention_mask'].shape}")
        print(f"   - doc_id: {sample['doc_id']}")
        print(f"   - text preview: {sample['doc_text'][:100]}...")
        
        return corpus_dataset
        
    except Exception as e:
        print(f"❌ Corpus dataset test failed: {e}")
        return None


def test_qa_dataset(qa_path: str, corpus_dataset, tokenizer):
    """Test the QA dataset."""
    print("\\n📋 Testing QA Dataset...")
    
    try:
        # Create QA dataset
        qa_dataset = VNLegalQADataset(
            qa_path=qa_path,
            corpus_dataset=corpus_dataset,
            tokenizer=tokenizer,
            max_length=128
        )
        
        print(f"✅ QA loaded: {len(qa_dataset):,} question-answer pairs")
        
        # Test a single sample
        sample = qa_dataset[0]
        print(f"✅ Sample keys: {list(sample.keys())}")
        print(f"   - query_input_ids shape: {sample['query_input_ids'].shape}")
        print(f"   - pos_input_ids shape: {sample['pos_input_ids'].shape}")
        print(f"   - qid: {sample['qid']}")
        print(f"   - question preview: {sample['query_text'][:100]}...")
        
        return qa_dataset
        
    except Exception as e:
        print(f"❌ QA dataset test failed: {e}")
        return None


def test_splade_dataset(corpus_dataset, qa_dataset):
    """Test the SPLADE dataset."""
    print("\\n🔄 Testing SPLADE Dataset...")
    
    try:
        # Create SPLADE dataset
        splade_dataset = VNLegalSpladeDataset(
            corpus_dataset=corpus_dataset,
            qa_dataset=qa_dataset,
            num_negatives=2
        )
        
        print(f"✅ SPLADE dataset: {len(splade_dataset):,} training samples")
        
        # Test a single sample
        sample = splade_dataset[0]
        print(f"✅ Sample keys: {list(sample.keys())}")
        print(f"   - query_input_ids shape: {sample['query_input_ids'].shape}")
        print(f"   - pos_input_ids shape: {sample['pos_input_ids'].shape}")
        print(f"   - Number of negatives: {len(sample['neg_input_ids'])}")
        print(f"   - qid: {sample['qid']}")
        
        return splade_dataset
        
    except Exception as e:
        print(f"❌ SPLADE dataset test failed: {e}")
        return None


def test_dataloader(corpus_path: str, qa_path: str, tokenizer):
    """Test the dataloader creation."""
    print("\\n🚀 Testing DataLoader Creation...")
    
    try:
        # Create dataloaders
        train_loader, val_loader, corpus_dataset = create_splade_dataloaders(
            corpus_path=corpus_path,
            qa_path=qa_path,
            tokenizer=tokenizer,
            batch_size=2,  # Small batch for testing
            max_doc_length=256,
            max_query_length=128,
            num_negatives=2,
            num_workers=0,  # Avoid multiprocessing issues in testing
            split_ratio=0.8
        )
        
        print(f"✅ DataLoaders created successfully!")
        print(f"   - Train loader: {len(train_loader)} batches")
        print(f"   - Val loader: {len(val_loader)} batches")
        
        # Test a single batch
        print("\\n🧪 Testing batch loading...")
        batch = next(iter(train_loader))
        print(f"✅ Batch keys: {list(batch.keys())}")
        print(f"   - query_input_ids shape: {batch['query_input_ids'].shape}")
        print(f"   - pos_input_ids shape: {batch['pos_input_ids'].shape}")
        if batch['neg_input_ids'] is not None:
            print(f"   - neg_input_ids shape: {batch['neg_input_ids'].shape}")
        print(f"   - Batch size: {len(batch['qids'])}")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return None, None


def main():
    """Main function to run all tests."""
    print("🧪 === SPLADE DATASET TESTING ===")
    print()
    
    # Data paths
    corpus_path = "data/vn-legal-doc/cleaned_corpus_data.csv"
    qa_path = "data/vn-legal-doc/test_data.csv"
    
    # Check if files exist
    if not Path(corpus_path).exists():
        print(f"❌ Corpus file not found: {corpus_path}")
        print("💡 Please run process_corpus_csv.py first to create cleaned corpus")
        return
    
    if not Path(qa_path).exists():
        print(f"❌ QA file not found: {qa_path}")
        print("💡 Please ensure test_data.csv exists")
        return
    
    # Create tokenizer
    print("🔧 Creating tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=30000, max_length=512)
    print("✅ Tokenizer created")
    
    # Test corpus dataset
    corpus_dataset = test_corpus_dataset(corpus_path, tokenizer)
    if corpus_dataset is None:
        return
    
    # Test QA dataset
    qa_dataset = test_qa_dataset(qa_path, corpus_dataset, tokenizer)
    if qa_dataset is None:
        return
    
    # Test SPLADE dataset
    splade_dataset = test_splade_dataset(corpus_dataset, qa_dataset)
    if splade_dataset is None:
        return
    
    # Test dataloader
    train_loader, val_loader = test_dataloader(corpus_path, qa_path, tokenizer)
    if train_loader is None:
        return
    
    print("\\n🎉 === ALL TESTS PASSED ===")
    print()
    print("📊 Summary:")
    print(f"   - Corpus documents: {len(corpus_dataset):,}")
    print(f"   - QA pairs: {len(qa_dataset):,}")
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print()
    print("✅ SPLADE dataset is ready for training!")
    print()
    print("🚀 Next steps:")
    print("   1. Install PyTorch and Transformers if not already installed")
    print("   2. Load a real tokenizer (e.g., from Transformers)")
    print("   3. Initialize SPLADE model")
    print("   4. Start training with the created dataloaders")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
