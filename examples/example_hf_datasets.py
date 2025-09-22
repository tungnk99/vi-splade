#!/usr/bin/env python3
"""
Example: Using Hugging Face Datasets with SPLADE Training

This example shows how to:
1. Download datasets from Hugging Face Hub
2. Use them with SPLADE training
3. Combine multiple datasets for training

Usage:
    python examples/example_hf_datasets.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_process.dataset_manager import DatasetManager
from src.model.splade import SpladeModel
from src.trainer.splade_trainer import SpladeTrainer


def download_and_setup_datasets():
    """Download datasets if not available."""
    print("📥 === DATASET DOWNLOAD & SETUP ===")
    
    # Check which datasets are available
    manager = DatasetManager()
    datasets_to_check = ["vn_legal_retrieval", "vinli_triplet"]
    
    missing_datasets = []
    for dataset_id in datasets_to_check:
        info = manager.get_dataset_info(dataset_id)
        if info and info["available"]:
            print(f"✅ {dataset_id}: Available in data/{dataset_id}/")
        else:
            print(f"❌ {dataset_id}: Not available")
            missing_datasets.append(dataset_id)
    
    if missing_datasets:
        print(f"\n💡 To download missing datasets:")
        for dataset_id in missing_datasets:
            print(f"   python data_process/download_datasets.py --dataset {dataset_id}")
        print(f"   python data_process/download_datasets.py --dataset all")
        return False
    
    return True


def example_load_and_inspect():
    """Load datasets and inspect their structure."""
    print("\n🔍 === DATASET INSPECTION ===")
    
    manager = DatasetManager()
    
    # Try to load legal retrieval dataset
    try:
        dataset_id = "vn_legal_retrieval"
        
        # Get basic info
        info = manager.get_dataset_info(dataset_id)
        print(f"📋 Dataset: {info['name']}")
        print(f"   Path: {info['local_path']}")
        print(f"   Splits: {info['splits']}")
        
        # Load train split
        train_data = manager.load_dataset(dataset_id, "train", as_hf_dataset=False)
        print(f"\n📊 Train data structure:")
        print(f"   Rows: {len(train_data):,}")
        print(f"   Columns: {list(train_data.columns)}")
        
        # Show sample
        print(f"\n📖 Sample data:")
        sample = manager.get_sample_data(dataset_id, "train", n_samples=2)
        for idx, row in sample.iterrows():
            print(f"   Sample {idx + 1}:")
            for col in sample.columns:
                value = str(row[col])[:100]
                print(f"     {col}: {value}...")
            print()
        
        # Get statistics
        stats = manager.get_dataset_statistics(dataset_id)
        print(f"📈 Dataset statistics:")
        print(f"   Total rows: {stats['total_rows']:,}")
        print(f"   Total size: {stats['total_size_mb']} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False


def example_convert_for_training():
    """Convert HF dataset to format suitable for SPLADE training."""
    print("\n🔄 === CONVERT FOR SPLADE TRAINING ===")
    
    manager = DatasetManager()
    
    try:
        # Load dataset
        dataset_id = "vn_legal_retrieval"
        train_data = manager.load_dataset(dataset_id, "train", as_hf_dataset=False)
        test_data = manager.load_dataset(dataset_id, "test", as_hf_dataset=False)
        
        print(f"📖 Loaded data:")
        print(f"   Train: {len(train_data):,} samples")
        print(f"   Test: {len(test_data):,} samples")
        
        # Convert to VNLegalDataset format if needed
        output_dir = Path("data/converted_for_training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if data has the expected columns
        print(f"\n🔍 Checking data format...")
        print(f"   Train columns: {list(train_data.columns)}")
        
        if "query" in train_data.columns and "document" in train_data.columns:
            print("✅ Data already in query-document format")
            
            # Save as CSV for VNLegalDataset compatibility
            train_csv = output_dir / "train_data.csv"
            test_csv = output_dir / "test_data.csv"
            
            # Prepare data with required columns
            train_converted = train_data.copy()
            test_converted = test_data.copy()
            
            # Rename columns to match VNLegalDataset expectations
            if "query" in train_converted.columns:
                train_converted = train_converted.rename(columns={"query": "question"})
                test_converted = test_converted.rename(columns={"query": "question"})
            
            if "document" in train_converted.columns:
                train_converted = train_converted.rename(columns={"document": "context_list"})
                test_converted = test_converted.rename(columns={"document": "context_list"})
            
            # Add required IDs if missing
            if "qid" not in train_converted.columns:
                train_converted["qid"] = range(len(train_converted))
            if "cid" not in train_converted.columns:
                train_converted["cid"] = range(len(train_converted))
            
            if "qid" not in test_converted.columns:
                test_converted["qid"] = range(len(test_converted))
            if "cid" not in test_converted.columns:
                test_converted["cid"] = range(len(test_converted))
            
            # Save converted data
            train_converted.to_csv(train_csv, index=False)
            test_converted.to_csv(test_csv, index=False)
            
            print(f"💾 Converted datasets saved:")
            print(f"   {train_csv}")
            print(f"   {test_csv}")
            
            return str(train_csv), str(test_csv)
            
        else:
            print("❌ Data format not suitable for direct conversion")
            return None, None
            
    except Exception as e:
        print(f"❌ Error converting dataset: {e}")
        return None, None


def example_train_with_hf_data():
    """Train SPLADE model with HF dataset."""
    print("\n🚀 === SPLADE TRAINING WITH HF DATA ===")
    
    # Convert dataset first
    train_csv, test_csv = example_convert_for_training()
    
    if not train_csv or not test_csv:
        print("❌ Could not convert dataset for training")
        return
    
    try:
        # Create SPLADE model
        print("\n🔧 Creating SPLADE model...")
        splade_model = SpladeModel(
            model_name_or_path="vinai/phobert-base-v2",
            model_type="splade",
            eval=False
        )
        print(f"✅ Model created: {splade_model.model_name_or_path}")
        
        # Initialize trainer
        trainer = SpladeTrainer(
            model=splade_model,
            output_dir="models/splade-hf-data",
            run_name="splade-hf-legal-data"
        )
        
        # Training configuration (small for demo)
        training_config = {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 4,
            "learning_rate": 2e-5,
            "fp16": False,
            "eval_steps": 50,
            "save_steps": 50,
            "logging_steps": 20
        }
        
        print(f"\n⚙️ Training configuration:")
        for key, value in training_config.items():
            print(f"   {key}: {value}")
        
        # Create corpus CSV (use test data as corpus for demo)
        corpus_csv = Path("data/converted_for_training/corpus_data.csv") 
        import pandas as pd
        test_df = pd.read_csv(test_csv)
        
        # Create corpus format
        corpus_data = []
        for idx, row in test_df.iterrows():
            corpus_data.append({
                "cid": row.get("cid", idx),
                "text": row.get("context_list", "")
            })
        
        corpus_df = pd.DataFrame(corpus_data)
        corpus_df.to_csv(corpus_csv, index=False)
        print(f"💾 Created corpus file: {corpus_csv}")
        
        # Start training
        print(f"\n🏃‍♂️ Starting SPLADE training...")
        print(f"   Using converted HF data from data/converted_for_training/")
        
        model = trainer.train(
            corpus_path=str(corpus_csv),
            test_path=test_csv,
            training_args=training_config
        )
        
        print(f"\n🎉 Training completed!")
        print(f"   Model saved in: models/splade-hf-data/")
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_combine_datasets():
    """Combine multiple HF datasets."""
    print("\n🔄 === COMBINING MULTIPLE DATASETS ===")
    
    manager = DatasetManager()
    
    # Check available datasets
    available = []
    for dataset_id in ["vn_legal_retrieval", "vinli_triplet", "vinli_reranking"]:
        info = manager.get_dataset_info(dataset_id)
        if info and info["available"]:
            available.append(dataset_id)
            print(f"✅ {dataset_id}: Available")
        else:
            print(f"❌ {dataset_id}: Not available")
    
    if len(available) >= 2:
        print(f"\n🔄 Combining {len(available)} datasets...")
        
        # Combine with sampling
        combined = manager.create_combined_dataset(
            dataset_configs=[
                {"dataset_id": dataset_id, "split": "train"} 
                for dataset_id in available
            ],
            output_path="data/combined_hf_datasets.parquet",
            sample_ratio={
                "vn_legal_retrieval": 1.0,  # Use all legal data
                "vinli_triplet": 0.3,       # Sample 30% of triplet data
                "vinli_reranking": 0.5      # Sample 50% of reranking data
            }
        )
        
        print(f"✅ Combined dataset created:")
        print(f"   Total samples: {len(combined):,}")
        print(f"   Saved to: data/combined_hf_datasets.parquet")
        print(f"   Source datasets: {combined['source_dataset'].unique()}")
        
        return combined
    else:
        print("❌ Need at least 2 datasets for combining")
        return None


def main():
    """Main demo function."""
    print("🤗 === HUGGING FACE DATASETS WITH SPLADE ===")
    print()
    
    # Step 1: Check and download datasets
    if not download_and_setup_datasets():
        print("\n⏹️ Please download datasets first, then re-run this example")
        return
    
    # Step 2: Inspect datasets
    if not example_load_and_inspect():
        print("\n❌ Could not load datasets")
        return
    
    # Step 3: Combine datasets (optional)
    example_combine_datasets()
    
    # Step 4: Train with HF data
    print("\n" + "="*60)
    choice = input("🚀 Start SPLADE training with HF data? (y/n): ").strip().lower()
    
    if choice == 'y':
        model = example_train_with_hf_data()
        if model:
            print("\n🎊 SUCCESS! SPLADE model trained with HF datasets")
        else:
            print("\n❌ Training failed")
    else:
        print("\n✅ Demo completed (training skipped)")
    
    print(f"\n📚 Next steps:")
    print(f"   - Check downloaded data in data/ folders")
    print(f"   - Use DatasetManager in your scripts")
    print(f"   - Combine datasets as needed")
    print(f"   - Train with different configurations")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
