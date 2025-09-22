#!/usr/bin/env python3
"""
Example: Source Conversion & SPLADE Training Workflow

Demo workflow hoàn chỉnh:
1. Download datasets từ HF Hub
2. Convert về VNLegalDataset format
3. Load qua DatasetManager
4. Train SPLADE model

Usage:
    python examples/example_source_conversion.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_process.dataset_converters import run_by_config
from data_process.dataset_manager import DatasetManager
from src.model.splade import SpladeModel
from src.trainer.splade_trainer import SpladeTrainer


def step1_download_sources():
    """Step 1: Download datasets from HF Hub (nếu chưa có)."""
    print("📥 === STEP 1: DOWNLOAD SOURCES ===")
    
    import subprocess
    import os
    
    # Check if HF_TOKEN is set
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN not set. Please run:")
        print("   export HF_TOKEN='your_token'")
        return False
    
    print(f"✅ HF_TOKEN found: {hf_token[:8]}...")
    
    # Check which datasets are already downloaded
    from data_process.download_datasets import HFDatasetDownloader
    
    try:
        downloader = HFDatasetDownloader()
        available_datasets = []
        
        for dataset_id in ["vn_legal_retrieval", "vinli_triplet"]:
            config = downloader.datasets.get(dataset_id)
            if config:
                local_path = Path(config["local_path"])
                if local_path.exists() and any(local_path.glob("*.parquet")):
                    print(f"✅ {dataset_id}: Already downloaded")
                    available_datasets.append(dataset_id)
                else:
                    print(f"❌ {dataset_id}: Not downloaded")
        
        if len(available_datasets) < 2:
            print("\n💡 Need to download datasets first:")
            print("   python data_process/download_datasets.py --dataset vn_legal_retrieval")
            print("   python data_process/download_datasets.py --dataset vinli_triplet")
            return False
        
        print(f"\n✅ Found {len(available_datasets)} datasets ready for conversion")
        return True
        
    except Exception as e:
        print(f"❌ Error checking downloads: {e}")
        return False


def step2_convert_sources():
    """Step 2: Convert sources to VNLegalDataset format."""
    print("\n🔄 === STEP 2: CONVERT SOURCES ===")
    
    # Convert all available sources
    print("🔄 Converting all available sources...")
    
    try:
        results = run_by_config("all", output_base_dir="clean_data")
        
        # Check results
        successful = sum(1 for r in results.values() if r)
        total = len(results)
        
        print(f"\n📊 Conversion Summary:")
        print(f"   Total sources: {total}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {total - successful}")
        
        for source_name, files in results.items():
            status = "✅" if files else "❌"
            print(f"   {status} {source_name}: {len(files)} files")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        return False


def step3_load_via_manager():
    """Step 3: Load datasets via DatasetManager."""
    print("\n📖 === STEP 3: LOAD VIA MANAGER ===")
    
    # Initialize manager to read from converted data
    manager = DatasetManager(converted_base_dir="clean_data")
    
    print("📋 Available datasets:")
    for dataset_id in manager.list_available_datasets():
        info = manager.get_dataset_info(dataset_id)
        if info:
            status = "✅" if info["available"] else "❌"
            print(f"   {status} {dataset_id}")
            if info["available"]:
                print(f"      Converted path: {info['converted_path']}")
                if "file_info" in info:
                    for split, split_info in info["file_info"].items():
                        print(f"      {split}: {split_info['rows']:,} rows")
    
    # Try loading a dataset
    try:
        dataset_id = "vn_legal_retrieval"
        
        print(f"\n📖 Loading {dataset_id}...")
        
        # Load corpus
        corpus = manager.load_dataset(dataset_id, "corpus", as_hf_dataset=False)
        print(f"✅ Corpus loaded: {len(corpus):,} documents")
        print(f"   Columns: {list(corpus.columns)}")
        print(f"   Sample: {corpus.iloc[0]['text'][:100]}...")
        
        # Load train data
        train_data = manager.load_dataset(dataset_id, "train", as_hf_dataset=False)
        print(f"✅ Train data loaded: {len(train_data):,} samples")
        print(f"   Columns: {list(train_data.columns)}")
        print(f"   Sample question: {train_data.iloc[0]['question']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def step4_train_splade():
    """Step 4: Train SPLADE với converted data."""
    print("\n🚀 === STEP 4: SPLADE TRAINING ===")
    
    try:
        # 1. Create SPLADE model
        print("🔧 Creating SPLADE model...")
        splade_model = SpladeModel(
            model_name_or_path="vinai/phobert-base-v2",
            model_type="splade",
            eval=False
        )
        print(f"✅ Model created: {splade_model.model_name_or_path}")
        
        # 2. Initialize trainer
        trainer = SpladeTrainer(
            model=splade_model,
            output_dir="models/splade-converted-data",
            run_name="splade-converted-vn-legal"
        )
        
        # 3. Prepare paths to converted data
        converted_base = Path("clean_data/vn_legal_retrieval")
        corpus_path = converted_base / "corpus.csv"
        train_path = converted_base / "train.csv"
        
        if not corpus_path.exists() or not train_path.exists():
            print(f"❌ Converted data not found:")
            print(f"   Corpus: {corpus_path}")
            print(f"   Train: {train_path}")
            return False
        
        print(f"📂 Using converted data:")
        print(f"   Corpus: {corpus_path}")
        print(f"   Train: {train_path}")
        
        # 4. Training configuration (small for demo)
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
        
        print(f"⚙️ Training configuration:")
        for key, value in training_config.items():
            print(f"   {key}: {value}")
        
        # 5. Start training
        print(f"\n🏃‍♂️ Starting SPLADE training...")
        print(f"   Using converted VNLegalDataset format")
        
        model = trainer.train(
            corpus_path=str(corpus_path),
            test_path=str(train_path),  # Use train as test for demo
            training_args=training_config
        )
        
        print(f"\n🎉 Training completed!")
        print(f"   Model saved in: models/splade-converted-data/")
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main demo workflow."""
    print("🔧 === SOURCE CONVERSION & SPLADE TRAINING WORKFLOW ===")
    print()
    
    # Step 1: Check downloads
    print("Checking downloaded sources...")
    if not step1_download_sources():
        print("\n⏹️ Please download sources first, then re-run")
        return
    
    # Step 2: Convert sources
    print("\nConverting sources to VNLegalDataset format...")
    if not step2_convert_sources():
        print("\n❌ Source conversion failed")
        return
    
    # Step 3: Load via manager
    print("\nLoading via DatasetManager...")
    if not step3_load_via_manager():
        print("\n❌ Data loading failed")
        return
    
    # Step 4: Train SPLADE
    print("\nReady for SPLADE training!")
    choice = input("🚀 Start training? (y/n): ").strip().lower()
    
    if choice == 'y':
        model = step4_train_splade()
        if model:
            print("\n🎊 SUCCESS! Complete workflow finished")
            print("\n📊 Workflow Summary:")
            print("✅ Downloaded HF datasets")
            print("✅ Converted to VNLegalDataset format")  
            print("✅ Loaded via unified DatasetManager")
            print("✅ Trained SPLADE model")
            print("\n📁 Generated files:")
            print("   clean_data/*/: Converted datasets")
            print("   models/splade-converted-data/: Trained model")
        else:
            print("\n❌ Training failed")
    else:
        print("\n✅ Workflow demo completed (training skipped)")
    
    print(f"\n💡 Next steps:")
    print(f"   - Check converted data: ls clean_data/")
    print(f"   - Use DatasetManager in your scripts")
    print(f"   - Train with different datasets")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Workflow interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
