#!/usr/bin/env python3
"""
Dataset Manager Example

Demonstrating enhanced dataset manager capabilities:
- Auto-download from Hugging Face Hub
- Auto-conversion to standard format 
- Building datasets in different formats (query-doc, triplets, reranking)
- Loading datasets from configs

Usage:
    python examples/example_dataset_manager.py
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_process.dataset_manager import DatasetManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_auto_download_and_conversion():
    """Demo auto-download and conversion capabilities."""
    print("\n" + "="*70)
    print("🚀 DEMO: Auto-Download and Conversion")
    print("="*70)
    
    # Initialize with auto features enabled
    manager = DatasetManager(
        auto_download=True,
        auto_convert=True,
        converted_base_dir="clean_data"
    )
    
    print(f"\n📋 Available datasets: {manager.list_available_datasets()}")
    
    # Try loading a dataset (should auto-download and convert if needed)
    dataset_id = "vinli_triplet"  # Start with smaller dataset
    try:
        print(f"\n📖 Loading dataset: {dataset_id}")
        
        # This will auto-download if not present, then auto-convert
        info = manager.get_dataset_info(dataset_id)
        print(f"Status: {'✅ Available' if info['available'] else '📥 Will auto-download'}")
        
        # Load a split - should trigger auto-download/convert
        dataset = manager.load_dataset(dataset_id, "train")
        print(f"✅ Successfully loaded: {len(dataset):,} samples")
        
        # Show a sample
        if len(dataset) > 0:
            print(f"\n🔍 Sample data:")
            sample = dict(dataset[0])
            for key, value in sample.items():
                # Truncate long text for display
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_query_doc_format():
    """Demo building query-document pairs."""
    print("\n" + "="*70)
    print("📋 DEMO: Query-Document Pairs")
    print("="*70)
    
    manager = DatasetManager(auto_download=True, auto_convert=True)
    
    # Try different datasets
    datasets_to_try = ["vn_legal_retrieval", "vinli_triplet"]
    
    for dataset_id in datasets_to_try:
        try:
            print(f"\n🔨 Building query-doc pairs for: {dataset_id}")
            
            # Build query-document format
            query_doc_datasets = manager.build_dataset(
                dataset_id=dataset_id,
                format='query_doc',
                max_samples=50  # Limit for demo
            )
            
            for split_name, dataset in query_doc_datasets.items():
                print(f"  ✅ {split_name}: {len(dataset):,} query-doc pairs")
                
                if len(dataset) > 0:
                    sample = dict(dataset[0])
                    print(f"    Sample keys: {list(sample.keys())}")
                    
                    # Show truncated sample
                    for key, value in sample.items():
                        if isinstance(value, str) and len(value) > 80:
                            value = value[:80] + "..."
                        print(f"      {key}: {value}")
                    break  # Just show first split
            
        except Exception as e:
            print(f"  ⚠️ Failed for {dataset_id}: {e}")


def demo_triple_format():
    """Demo building triplet format."""
    print("\n" + "="*70)
    print("🔗 DEMO: Triplet Format (Query, Positive, Negative)")
    print("="*70)
    
    manager = DatasetManager(auto_download=True, auto_convert=True)
    
    # Try datasets that are suitable for triplets
    datasets_to_try = ["vinli_triplet", "vn_legal_retrieval"]
    
    for dataset_id in datasets_to_try:
        try:
            print(f"\n🔨 Building triplets for: {dataset_id}")
            
            # Build triplet format
            triple_datasets = manager.build_dataset(
                dataset_id=dataset_id,
                format='triple',
                max_samples=20,  # Fewer samples for demo
                negative_sampling=True,
                negative_ratio=2.0  # 2 negatives per positive
            )
            
            for split_name, dataset in triple_datasets.items():
                print(f"  ✅ {split_name}: {len(dataset):,} triplets")
                
                if len(dataset) > 0:
                    sample = dict(dataset[0])
                    print(f"    Sample structure: {list(sample.keys())}")
                    
                    # Show truncated triplet
                    for key in ['query', 'positive', 'negative']:
                        if key in sample:
                            value = str(sample[key])
                            if len(value) > 60:
                                value = value[:60] + "..."
                            print(f"      {key}: {value}")
                    break
            
        except Exception as e:
            print(f"  ⚠️ Failed for {dataset_id}: {e}")


def demo_reranking_format():
    """Demo building reranking format."""
    print("\n" + "="*70)
    print("📊 DEMO: Reranking Format (Query, Document, Score)")
    print("="*70)
    
    manager = DatasetManager(auto_download=True, auto_convert=True)
    
    dataset_id = "vn_legal_retrieval"
    
    try:
        print(f"\n🔨 Building reranking format for: {dataset_id}")
        
        # Build reranking format
        reranking_datasets = manager.build_dataset(
            dataset_id=dataset_id,
            format='reranking',
            max_samples=30
        )
        
        for split_name, dataset in reranking_datasets.items():
            print(f"  ✅ {split_name}: {len(dataset):,} reranking pairs")
            
            if len(dataset) > 0:
                sample = dict(dataset[0])
                print(f"    Structure: {list(sample.keys())}")
                
                # Show sample with scores
                query = str(sample.get('query', ''))[:50] + "..."
                document = str(sample.get('document', ''))[:50] + "..."
                score = sample.get('score', 'N/A')
                
                print(f"      query: {query}")
                print(f"      document: {document}")
                print(f"      score: {score}")
                break
    
    except Exception as e:
        print(f"  ⚠️ Failed: {e}")


def demo_dataset_statistics():
    """Demo dataset statistics and info."""
    print("\n" + "="*70)
    print("📊 DEMO: Dataset Statistics")
    print("="*70)
    
    manager = DatasetManager(auto_download=True, auto_convert=True)
    
    # Show info for all configured datasets
    for dataset_id in manager.list_available_datasets():
        try:
            print(f"\n📈 Dataset: {dataset_id}")
            
            info = manager.get_dataset_info(dataset_id)
            print(f"  Name: {info['name']}")
            print(f"  Available: {'✅' if info['available'] else '❌'}")
            
            if info['available']:
                try:
                    stats = manager.get_dataset_statistics(dataset_id)
                    print(f"  Total rows: {stats['total_rows']:,}")
                    print(f"  Total size: {stats['total_size_mb']} MB")
                    
                    for split_name, split_stats in stats['splits'].items():
                        print(f"    {split_name}: {split_stats['rows']:,} rows")
                except:
                    print("  (Stats not available)")
        
        except Exception as e:
            print(f"  ❌ Error getting info: {e}")


def demo_combined_workflow():
    """Demo a complete workflow."""
    print("\n" + "="*70)
    print("🔄 DEMO: Complete Workflow")
    print("="*70)
    
    manager = DatasetManager(auto_download=True, auto_convert=True)
    
    try:
        # 1. List and check datasets
        print("\n1️⃣ Listing available datasets...")
        available = manager.list_available_datasets()
        print(f"   Found: {available}")
        
        # 2. Pick a dataset and ensure it's ready
        dataset_id = "vinli_triplet"  # Good for demo
        print(f"\n2️⃣ Working with: {dataset_id}")
        
        # 3. Load original format
        print("\n3️⃣ Loading original format...")
        original_dataset = manager.load_dataset(dataset_id, "train")
        print(f"   Original: {len(original_dataset):,} samples")
        
        # 4. Build different formats
        print("\n4️⃣ Building different formats...")
        
        # Query-doc pairs
        query_doc = manager.build_dataset(dataset_id, format='query_doc', max_samples=10)
        print(f"   Query-doc: {sum(len(d) for d in query_doc.values())} pairs")
        
        # Triplets
        triplets = manager.build_dataset(dataset_id, format='triple', max_samples=5)
        print(f"   Triplets: {sum(len(d) for d in triplets.values())} triplets")
        
        # 5. Show usage examples
        print("\n5️⃣ Usage examples...")
        
        if query_doc and 'train' in query_doc:
            qd_dataset = query_doc['train']
            print(f"   Query-doc ready for training: {len(qd_dataset)} samples")
            print(f"   Columns: {list(qd_dataset.column_names)}")
        
        if triplets and 'train' in triplets:
            trip_dataset = triplets['train']
            print(f"   Triplets ready for training: {len(trip_dataset)} samples")
            print(f"   Columns: {list(trip_dataset.column_names)}")
        
        print("\n✅ Workflow completed successfully!")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")


def main():
    """Run all demos."""
    print("🎯 DATASET MANAGER ENHANCED DEMO")
    print("=" * 70)
    print("Demonstrating new capabilities:")
    print("- Auto-download from Hugging Face Hub")  
    print("- Auto-conversion to standard format")
    print("- Building datasets in multiple formats")
    print("- Seamless config-based management")
    
    # Run individual demos
    demo_auto_download_and_conversion()
    demo_query_doc_format()
    demo_triple_format()
    demo_reranking_format()
    demo_dataset_statistics()
    demo_combined_workflow()
    
    print("\n" + "="*70)
    print("🎉 ALL DEMOS COMPLETED!")
    print("="*70)
    print("\n💡 Key features demonstrated:")
    print("  ✅ Auto-download missing datasets from HF")
    print("  ✅ Auto-convert to standardized format")
    print("  ✅ Build query-document pairs")
    print("  ✅ Build triplets with negative sampling")
    print("  ✅ Build reranking format with scores")
    print("  ✅ Seamless integration with configs")
    
    print("\n🚀 Ready for production use!")


if __name__ == "__main__":
    main()
