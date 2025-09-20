#!/usr/bin/env python3
"""
Test suite for VNLegalDataset

Comprehensive tests for the Vietnamese Legal Dataset class.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataset.vn_legal_dataset import VNLegalDataset


def test_dataset_loading():
    """Test basic dataset loading functionality."""
    print("🧪 Testing VNLegal Dataset...")
    
    # Data paths
    corpus_path = "data/vn-legal-doc/cleaned_corpus_data.csv"
    test_path = "data/vn-legal-doc/test_data.csv"
    
    # Check if files exist
    if not Path(corpus_path).exists():
        print(f"❌ Corpus file not found: {corpus_path}")
        print("💡 Please run process_corpus_csv.py first")
        return False
    
    if not Path(test_path).exists():
        print(f"❌ Test file not found: {test_path}")
        return False
    
    try:
        # Create dataset
        dataset = VNLegalDataset(
            corpus_path=corpus_path,
            test_path=test_path
        )
        
        print("✅ Dataset created successfully")
        return dataset
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return False


def test_dataset_statistics(dataset):
    """Test dataset statistics functionality."""
    print("\n📊 Testing Dataset Statistics...")
    
    try:
        stats = dataset.stats()
        
        # Check required keys
        required_keys = [
            'total_samples', 'train_samples', 'test_samples', 'dev_samples',
            'unique_queries', 'unique_documents', 'corpus_size', 'data_structure'
        ]
        
        for key in required_keys:
            assert key in stats, f"Missing stats key: {key}"
        
        # Display statistics
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   - {key.replace('_', ' ').title()}:")
                for sub_key, sub_value in value.items():
                    print(f"     • {sub_key}: {sub_value:,}")
            else:
                print(f"   - {key.replace('_', ' ').title()}: {value:,}")
        
        print("✅ Statistics test passed")
        return True
        
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False


def test_dataset_structure(dataset):
    """Test dataset data structure."""
    print("\n🏗️ Testing Dataset Structure...")
    
    try:
        # Test data structure
        data_structure = dataset.get_data_structure()
        
        # Check structure keys
        expected_keys = ["test", "train", "dev"]
        for key in expected_keys:
            assert key in data_structure, f"Missing structure key: {key}"
            assert isinstance(data_structure[key], list), f"Structure key {key} should be list"
        
        # Test split methods
        test_samples = dataset.get_test_samples()
        train_samples = dataset.get_train_samples()
        dev_samples = dataset.get_dev_samples()
        
        assert isinstance(test_samples, list), "Test samples should be list"
        assert isinstance(train_samples, list), "Train samples should be list"
        assert isinstance(dev_samples, list), "Dev samples should be list"
        
        print(f"   - Test samples: {len(test_samples):,}")
        print(f"   - Train samples: {len(train_samples):,}")
        print(f"   - Dev samples: {len(dev_samples):,}")
        
        print("✅ Structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False


def test_dataset_iteration(dataset):
    """Test dataset iteration functionality."""
    print("\n🔄 Testing Dataset Iteration...")
    
    try:
        # Test iteration
        sample_count = 0
        for sample in dataset:
            sample_count += 1
            if sample_count >= 3:  # Only check first 3
                break
            
            # Verify sample structure
            required_keys = ['query', 'document', 'qid', 'cid', 'split']
            for key in required_keys:
                assert key in sample, f"Missing sample key: {key}"
            
            # Verify data types
            assert isinstance(sample['query'], str), "Query should be string"
            assert isinstance(sample['document'], str), "Document should be string"
            assert isinstance(sample['qid'], (int, float)), "QID should be numeric"
            assert isinstance(sample['cid'], (int, float)), "CID should be numeric"
            assert isinstance(sample['split'], str), "Split should be string"
        
        print(f"✅ Iteration test passed! Checked {sample_count} samples")
        return True
        
    except Exception as e:
        print(f"❌ Iteration test failed: {e}")
        return False


def test_dataset_indexing(dataset):
    """Test dataset indexing functionality."""
    print("\n🎯 Testing Dataset Indexing...")
    
    try:
        # Test indexing
        if len(dataset) > 0:
            first_sample = dataset[0]
            print(f"   - First sample QID: {first_sample['qid']}")
            
            # Test last sample
            last_sample = dataset[len(dataset) - 1]
            print(f"   - Last sample QID: {last_sample['qid']}")
            
            # Test middle sample
            mid_idx = len(dataset) // 2
            mid_sample = dataset[mid_idx]
            print(f"   - Middle sample QID: {mid_sample['qid']}")
        
        print("✅ Indexing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Indexing test failed: {e}")
        return False


def test_dataset_filtering(dataset):
    """Test dataset filtering functionality."""
    print("\n🔍 Testing Dataset Filtering...")
    
    try:
        # Test split filtering
        train_samples = dataset.get_train_samples()
        test_samples = dataset.get_test_samples()
        dev_samples = dataset.get_dev_samples()
        
        print(f"   - Train samples: {len(train_samples):,}")
        print(f"   - Test samples: {len(test_samples):,}")
        print(f"   - Dev samples: {len(dev_samples):,}")
        
        # Test get_split_data method
        for split in ["test", "train", "dev"]:
            split_data = dataset.get_split_data(split)
            print(f"   - {split.title()} split: {len(split_data):,} samples")
        
        # Test sample lookup by QID
        if len(dataset) > 0:
            first_sample = dataset[0]
            qid = first_sample['qid']
            qid_samples = dataset.get_sample_by_qid(qid)
            print(f"   - Samples for QID {qid}: {len(qid_samples)}")
        
        print("✅ Filtering test passed")
        return True
        
    except Exception as e:
        print(f"❌ Filtering test failed: {e}")
        return False


def test_dataset_preview(dataset):
    """Test dataset preview functionality."""
    print("\n📋 Testing Dataset Preview...")
    
    try:
        print("=" * 60)
        dataset.preview(3)
        print("=" * 60)
        
        print("✅ Preview test passed")
        return True
        
    except Exception as e:
        print(f"❌ Preview test failed: {e}")
        return False


def test_document_lookup(dataset):
    """Test document lookup functionality."""
    print("\n📄 Testing Document Lookup...")
    
    try:
        # Test document lookup by CID
        if len(dataset) > 0:
            first_sample = dataset[0]
            cid = first_sample['cid']
            
            # Lookup document by CID
            doc_text = dataset.get_document_by_cid(cid)
            if doc_text:
                print(f"   - Found document for CID {cid}: {len(doc_text)} chars")
                print(f"   - Document preview: {doc_text[:100]}...")
            else:
                print(f"   - No document found for CID {cid}")
        
        print("✅ Document lookup test passed")
        return True
        
    except Exception as e:
        print(f"❌ Document lookup test failed: {e}")
        return False


def run_all_tests():
    """Run all tests for VNLegalDataset."""
    print("🚀 === VN LEGAL DATASET TEST SUITE ===")
    print()
    
    # Test 1: Dataset loading
    dataset = test_dataset_loading()
    if not dataset:
        print("\n❌ Dataset loading failed - stopping tests")
        return False
    
    # Test 2: Statistics
    if not test_dataset_statistics(dataset):
        print("\n❌ Statistics test failed")
        return False
    
    # Test 3: Structure
    if not test_dataset_structure(dataset):
        print("\n❌ Structure test failed")
        return False
    
    # Test 4: Iteration
    if not test_dataset_iteration(dataset):
        print("\n❌ Iteration test failed")
        return False
    
    # Test 5: Indexing
    if not test_dataset_indexing(dataset):
        print("\n❌ Indexing test failed")
        return False
    
    # Test 6: Filtering
    if not test_dataset_filtering(dataset):
        print("\n❌ Filtering test failed")
        return False
    
    # Test 7: Preview
    if not test_dataset_preview(dataset):
        print("\n❌ Preview test failed")
        return False
    
    # Test 8: Document lookup
    if not test_document_lookup(dataset):
        print("\n❌ Document lookup test failed")
        return False
    
    print("\n🎉 === ALL TESTS PASSED ===")
    print()
    print("💡 Usage examples:")
    print("```python")
    print("from src.dataset.vn_legal_dataset import VNLegalDataset")
    print()
    print("# Load dataset")
    print("dataset = VNLegalDataset(corpus_path, test_path)")
    print()
    print("# Get data structure")
    print("data = dataset.get_data_structure()")
    print("test_data = data['test']")
    print("train_data = data['train']")
    print()
    print("# Iterate over samples")
    print("for sample in dataset:")
    print("    query = sample['query']")
    print("    document = sample['document']")
    print("    # Process query-document pair...")
    print("```")
    
    return True


if __name__ == "__main__":
    try:
        success = run_all_tests()
        if success:
            print("\n✅ Test suite completed successfully!")
        else:
            print("\n❌ Test suite failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error in test suite: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
