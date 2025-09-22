import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv(".env", override=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class HFDatasetDownloader:
    """Hugging Face dataset downloader with parallel support."""
    
    def __init__(self, config_path: str = "data_process/dataset_configs.json"):
        """
        Initialize downloader with config file.
        
        Args:
            config_path: Path to dataset configuration JSON file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.datasets = self.config["datasets"]
        self.settings = self.config["download_settings"]
        
        # Create cache directory
        cache_dir = Path(self.settings["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📚 Loaded {len(self.datasets)} dataset configurations")
    
    def _load_config(self) -> Dict:
        """Load dataset configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_datasets(self) -> None:
        """List all available datasets."""
        print("📋 Available Datasets:")
        print("=" * 60)
        
        for dataset_id, config in self.datasets.items():
            print(f"🔹 {dataset_id}")
            print(f"   Name: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   Hub ID: {config['hub_id']}")
            print(f"   Splits: {', '.join(config['splits'].keys())}")
            print(f"   Use case: {config['use_case']}")
            print()
    
    def check_auth(self) -> bool:
        """Check if user is authenticated with Hugging Face."""
        try:
            # Get HF token from environment
            import os
            hf_token = os.getenv('HF_TOKEN')
            
            if hf_token:
                # Test token by trying to get user info
                from huggingface_hub import whoami
                user = whoami(token=hf_token)
                logger.info(f"✅ Authenticated via ENV token as: {user['name']}")
                return True
            else:
                # Fallback to saved token
                from huggingface_hub import whoami
                user = whoami()
                logger.info(f"✅ Authenticated via saved token as: {user['name']}")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ Authentication check failed: {e}")
            logger.info("💡 Options:")
            logger.info("   1. Set environment: export HF_TOKEN='your_token'")
            logger.info("   2. Or run: huggingface-cli login")
            return False
    
    def download_dataset(self, dataset_id: str, force_redownload: bool = False) -> bool:
        """
        Download a specific dataset.
        
        Args:
            dataset_id: Dataset identifier from config
            force_redownload: Force redownload even if exists
            
        Returns:
            bool: Success status
        """
        if dataset_id not in self.datasets:
            logger.error(f"❌ Dataset '{dataset_id}' not found in config")
            return False
        
        config = self.datasets[dataset_id]
        hub_id = config["hub_id"]
        local_path = Path(config["local_path"])
        
        # Create local directory
        local_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📥 Downloading {config['name']} ({hub_id})")
        
        try:
            # Check if already exists
            if not force_redownload and self._dataset_exists(local_path, config["splits"]):
                logger.info(f"✅ Dataset already exists at {local_path}")
                return True
            
            # Download each split
            success_count = 0
            total_splits = len(config["splits"])
            
            with tqdm(total=total_splits, desc=f"Downloading {dataset_id}") as pbar:
                for split_name, split_path in config["splits"].items():
                    try:
                        # Download split with token if available
                        import os
                        hf_token = os.getenv('HF_TOKEN')
                        
                        dataset = load_dataset(
                            hub_id,
                            split=split_name,
                            cache_dir=self.settings["cache_dir"],
                            token=hf_token  # Will be None if not set, which is fine
                        )
                        
                        # Convert to pandas and save
                        df = dataset.to_pandas()
                        output_file = local_path / f"{split_name}.parquet"
                        df.to_parquet(output_file, index=False)
                        
                        logger.info(f"  ✅ {split_name}: {len(df):,} rows → {output_file}")
                        success_count += 1
                        
                    except Exception as e:
                        logger.error(f"  ❌ Failed to download {split_name}: {e}")
                    
                    pbar.update(1)
            
            # Create metadata file
            self._save_metadata(local_path, config, success_count, total_splits)
            
            if success_count == total_splits:
                logger.info(f"🎉 Successfully downloaded {config['name']}")
                return True
            else:
                logger.warning(f"⚠️ Partial download: {success_count}/{total_splits} splits")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_id}: {e}")
            return False
    
    def _dataset_exists(self, local_path: Path, splits: Dict[str, str]) -> bool:
        """Check if dataset already exists locally."""
        for split_name in splits.keys():
            split_file = local_path / f"{split_name}.parquet"
            if not split_file.exists():
                return False
        return True
    
    def _save_metadata(self, local_path: Path, config: Dict, success_count: int, total_splits: int):
        """Save dataset metadata."""
        metadata = {
            "dataset_name": config["name"],
            "hub_id": config["hub_id"],
            "description": config["description"],
            "download_status": {
                "success_splits": success_count,
                "total_splits": total_splits,
                "complete": success_count == total_splits
            },
            "splits": list(config["splits"].keys()),
            "use_case": config["use_case"],
            "language": config["language"]
        }
        
        metadata_file = local_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def download_all(self, force_redownload: bool = False) -> Dict[str, bool]:
        """
        Download all datasets.
        
        Args:
            force_redownload: Force redownload even if exists
            
        Returns:
            Dict[str, bool]: Download status for each dataset
        """
        results = {}
        
        if self.settings.get("parallel_downloads", 1) > 1:
            # Parallel download
            with ThreadPoolExecutor(max_workers=self.settings["parallel_downloads"]) as executor:
                future_to_dataset = {
                    executor.submit(self.download_dataset, dataset_id, force_redownload): dataset_id
                    for dataset_id in self.datasets.keys()
                }
                
                for future in as_completed(future_to_dataset):
                    dataset_id = future_to_dataset[future]
                    try:
                        results[dataset_id] = future.result()
                    except Exception as e:
                        logger.error(f"❌ {dataset_id} failed: {e}")
                        results[dataset_id] = False
        else:
            # Sequential download
            for dataset_id in self.datasets.keys():
                results[dataset_id] = self.download_dataset(dataset_id, force_redownload)
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        logger.info(f"📊 Download Summary: {successful}/{total} datasets successful")
        
        return results
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """Get information about a specific dataset."""
        if dataset_id not in self.datasets:
            return None
        
        config = self.datasets[dataset_id]
        local_path = Path(config["local_path"])
        
        info = {
            "id": dataset_id,
            "name": config["name"],
            "description": config["description"],
            "hub_id": config["hub_id"],
            "local_path": str(local_path),
            "splits": config["splits"],
            "use_case": config["use_case"],
            "language": config["language"],
            "exists_locally": self._dataset_exists(local_path, config["splits"])
        }
        
        # Add file sizes if exists
        if info["exists_locally"]:
            info["file_info"] = {}
            for split_name in config["splits"].keys():
                split_file = local_path / f"{split_name}.parquet"
                if split_file.exists():
                    df = pd.read_parquet(split_file)
                    info["file_info"][split_name] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "size_mb": round(split_file.stat().st_size / 1024 / 1024, 2)
                    }
        
        return info


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Download datasets from Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset",
        help="Dataset ID to download (or 'all' for all datasets)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets"
    )
    parser.add_argument(
        "--info",
        help="Show information about a specific dataset"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if dataset exists"
    )
    parser.add_argument(
        "--config",
        default="data_process/dataset_configs.json",
        help="Path to dataset config file"
    )
    
    args = parser.parse_args()
    
    try:
        downloader = HFDatasetDownloader(args.config)
        
        # Check authentication if required
        if downloader.settings.get("require_auth", False):
            if not downloader.check_auth():
                logger.error("❌ Authentication required. Run: huggingface-cli login")
                return 1
        
        if args.list:
            downloader.list_datasets()
            return 0
        
        if args.info:
            info = downloader.get_dataset_info(args.info)
            if info:
                print(f"📋 Dataset Information: {args.info}")
                print("=" * 50)
                for key, value in info.items():
                    if key == "file_info":
                        print(f"File Info:")
                        for split, split_info in value.items():
                            print(f"  {split}: {split_info['rows']:,} rows, {split_info['size_mb']} MB")
                    else:
                        print(f"{key}: {value}")
            else:
                logger.error(f"❌ Dataset '{args.info}' not found")
                return 1
            return 0
        
        if args.dataset:
            if args.dataset.lower() == "all":
                results = downloader.download_all(args.force)
                failed = [k for k, v in results.items() if not v]
                if failed:
                    logger.error(f"❌ Failed datasets: {failed}")
                    return 1
            else:
                success = downloader.download_dataset(args.dataset, args.force)
                if not success:
                    return 1
        else:
            parser.print_help()
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
