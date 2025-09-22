#!/usr/bin/env python3
"""
Dataset Manager

Utility class to load and manage datasets from Hugging Face Hub with auto-download support.
Provides unified interface for all dataset types and dataset building capabilities.

Usage:
    from data_process.dataset_manager import DatasetManager
    
    manager = DatasetManager()
    dataset = manager.load_dataset('vn_legal_retrieval', split='train')
    print(f"Loaded {len(dataset)} samples")
    
    # Build dataset formats
    query_doc_dataset = manager.build_dataset('vn_legal_retrieval', format='query_doc')
    triple_dataset = manager.build_dataset('vinli_triplet', format='triple')
"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Tuple
from datasets import Dataset, load_dataset
from tqdm import tqdm
import numpy as np

import logging
logger = logging.getLogger(__name__)


class DatasetManager:
    """Manager for datasets with auto-download, conversion, and building capabilities."""
    
    def __init__(
        self, 
        config_path: str = "data_process/dataset_configs.json",
        converted_base_dir: str = "clean_data",
        auto_download: bool = True,
        auto_convert: bool = True
    ):
        """
        Initialize dataset manager.
        
        Args:
            config_path: Path to dataset configuration file
            converted_base_dir: Base directory for converted datasets
            auto_download: Automatically download missing datasets from HF
            auto_convert: Automatically convert downloaded datasets
        """
        self.config_path = Path(config_path)
        self.converted_base_dir = Path(converted_base_dir)
        self.auto_download = auto_download
        self.auto_convert = auto_convert
        self.config = self._load_config()
        self.datasets = self.config["datasets"]
        self.download_settings = self.config.get("download_settings", {})
        
        # Create necessary directories
        self.converted_base_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(self.download_settings.get("cache_dir", "data/.cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📚 Dataset manager initialized with {len(self.datasets)} datasets")
        logger.info(f"📂 Using converted data from: {self.converted_base_dir}")
        logger.info(f"🔄 Auto-download: {auto_download}, Auto-convert: {auto_convert}")
    
    def _load_config(self) -> Dict:
        """Load dataset configuration."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset IDs."""
        return list(self.datasets.keys())
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """Get detailed information about a dataset."""
        if dataset_id not in self.datasets:
            return None
        
        config = self.datasets[dataset_id]
        local_path = Path(config["local_path"])
        
        converted_path = self.converted_base_dir / dataset_id
        
        info = {
            "id": dataset_id,
            "name": config["name"],
            "description": config["description"],
            "original_path": config["local_path"],
            "converted_path": str(converted_path),
            "use_case": config["use_case"],
            "language": config["language"],
            "available": self._check_dataset_exists(dataset_id)
        }
        
        if info["available"]:
            info["split_info"] = {}
            for split_name in config["splits"].keys():
                split_file = local_path / f"{split_name}.parquet"
                if split_file.exists():
                    df = pd.read_parquet(split_file)
                    info["split_info"][split_name] = {
                        "rows": len(df),
                        "columns": list(df.columns)
                    }
        
        return info
    
    def _check_dataset_exists(self, dataset_id: str) -> bool:
        """Check if converted dataset exists locally."""
        converted_path = self.converted_base_dir / dataset_id
        if not converted_path.exists():
            return False
        
        # Check for standard files: corpus.csv + at least one of train/test/dev.csv
        corpus_file = converted_path / "corpus.csv"
        data_files = []
        for split in ['train.csv', 'test.csv', 'dev.csv']:
            if (converted_path / split).exists():
                data_files.append(split)
        
        # Must have corpus and at least one data file
        return corpus_file.exists() and len(data_files) > 0
    
    def load_dataset(
        self, 
        dataset_id: str, 
        split: str,
        as_hf_dataset: bool = True
    ) -> Union[Dataset, pd.DataFrame]:
        """
        Load a specific converted dataset split with auto-download support.
        
        Args:
            dataset_id: Dataset identifier
            split: Split name (train, test, dev, corpus, etc.)
            as_hf_dataset: Return as HuggingFace Dataset (True) or pandas DataFrame (False)
            
        Returns:
            Dataset or DataFrame: Loaded dataset
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset '{dataset_id}' not found. Available: {self.list_available_datasets()}")
        
        # Try to ensure dataset is available
        self._ensure_dataset_available(dataset_id)
        
        converted_path = self.converted_base_dir / dataset_id
        
        if not self._check_dataset_exists(dataset_id):
            raise FileNotFoundError(
                f"Dataset '{dataset_id}' could not be downloaded or converted.\n"
                f"Please check your configuration and network connection."
            )
        
        # Handle different split types with standard naming
        if split == "corpus":
            split_file = converted_path / "corpus.csv"
        else:
            split_file = converted_path / f"{split}.csv"
        
        if not split_file.exists():
            # List available splits
            available_splits = []
            if (converted_path / "corpus.csv").exists():
                available_splits.append("corpus")
            
            for standard_split in ['train', 'test', 'dev']:
                if (converted_path / f"{standard_split}.csv").exists():
                    available_splits.append(standard_split)
            
            raise ValueError(f"Split '{split}' not found for {dataset_id}. Available: {available_splits}")
        
        # Load CSV file
        df = pd.read_csv(split_file)
        logger.info(f"📖 Loaded {dataset_id}/{split}: {len(df):,} rows from converted data")
        
        if as_hf_dataset:
            return Dataset.from_pandas(df)
        else:
            return df
    
    def load_all_splits(
        self, 
        dataset_id: str,
        as_hf_dataset: bool = True
    ) -> Dict[str, Union[Dataset, pd.DataFrame]]:
        """
        Load all splits of a dataset.
        
        Args:
            dataset_id: Dataset identifier
            as_hf_dataset: Return as HuggingFace Dataset (True) or pandas DataFrame (False)
            
        Returns:
            Dict mapping split names to datasets
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset '{dataset_id}' not found")
        
        config = self.datasets[dataset_id]
        splits = {}
        
        for split_name in config["splits"].keys():
            try:
                splits[split_name] = self.load_dataset(dataset_id, split_name, as_hf_dataset)
            except FileNotFoundError:
                logger.warning(f"⚠️ Split '{split_name}' not found for {dataset_id}")
        
        return splits
    
    def get_sample_data(self, dataset_id: str, split: str, n_samples: int = 5) -> pd.DataFrame:
        """
        Get sample data from a dataset for inspection.
        
        Args:
            dataset_id: Dataset identifier
            split: Split name
            n_samples: Number of samples to return
            
        Returns:
            DataFrame with sample data
        """
        df = self.load_dataset(dataset_id, split, as_hf_dataset=False)
        return df.head(n_samples)
    
    def search_datasets_by_use_case(self, use_case: str) -> List[str]:
        """
        Find datasets by use case.
        
        Args:
            use_case: Use case to search for
            
        Returns:
            List of dataset IDs matching the use case
        """
        matching = []
        for dataset_id, config in self.datasets.items():
            if config.get("use_case") == use_case:
                matching.append(dataset_id)
        
        return matching
    
    def get_dataset_statistics(self, dataset_id: str) -> Dict:
        """
        Get comprehensive statistics for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Dictionary with dataset statistics
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset '{dataset_id}' not found")
        
        config = self.datasets[dataset_id]
        local_path = Path(config["local_path"])
        
        stats = {
            "dataset_id": dataset_id,
            "name": config["name"],
            "splits": {},
            "total_rows": 0,
            "total_size_mb": 0
        }
        
        for split_name in config["splits"].keys():
            split_file = local_path / f"{split_name}.parquet"
            if split_file.exists():
                df = pd.read_parquet(split_file)
                file_size_mb = split_file.stat().st_size / 1024 / 1024
                
                split_stats = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                    "size_mb": round(file_size_mb, 2)
                }
                
                # Text statistics if text columns exist
                text_cols = [col for col in df.columns if any(
                    keyword in col.lower() 
                    for keyword in ['text', 'query', 'document', 'sentence', 'content']
                )]
                
                if text_cols:
                    split_stats["text_stats"] = {}
                    for col in text_cols:
                        text_lengths = df[col].astype(str).str.len()
                        split_stats["text_stats"][col] = {
                            "avg_length": round(text_lengths.mean(), 1),
                            "max_length": text_lengths.max(),
                            "min_length": text_lengths.min()
                        }
                
                stats["splits"][split_name] = split_stats
                stats["total_rows"] += len(df)
                stats["total_size_mb"] += file_size_mb
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        
        return stats
    
    def _ensure_dataset_available(self, dataset_id: str) -> bool:
        """
        Ensure dataset is available locally, download and convert if needed.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            bool: Success status
        """
        if self._check_dataset_exists(dataset_id):
            return True
        
        logger.info(f"🔍 Dataset '{dataset_id}' not found locally")
        
        # Check if raw data exists
        config = self.datasets[dataset_id]
        raw_path = Path(config["local_path"])
        
        if not self._raw_dataset_exists(dataset_id) and self.auto_download:
            logger.info(f"📥 Auto-downloading '{dataset_id}' from Hugging Face...")
            if not self._download_from_hf(dataset_id):
                logger.error(f"❌ Failed to download '{dataset_id}'")
                return False
        
        if self._raw_dataset_exists(dataset_id) and self.auto_convert:
            logger.info(f"🔄 Auto-converting '{dataset_id}' to standard format...")
            if not self._convert_dataset(dataset_id):
                logger.error(f"❌ Failed to convert '{dataset_id}'")
                return False
        
        return self._check_dataset_exists(dataset_id)
    
    def _raw_dataset_exists(self, dataset_id: str) -> bool:
        """
        Check if raw downloaded dataset exists.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            bool: Whether raw dataset exists
        """
        config = self.datasets[dataset_id]
        raw_path = Path(config["local_path"])
        
        if not raw_path.exists():
            return False
        
        # Check if splits exist
        for split_name in config["splits"].keys():
            split_file = raw_path / f"{split_name}.parquet"
            if not split_file.exists():
                return False
        
        return True
    
    def _download_from_hf(self, dataset_id: str) -> bool:
        """
        Download dataset from Hugging Face Hub.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            bool: Success status
        """
        config = self.datasets[dataset_id]
        hub_id = config["hub_id"]
        local_path = Path(config["local_path"])
        
        local_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get HF token if available
            hf_token = os.getenv('HF_TOKEN')
            
            success_count = 0
            total_splits = len(config["splits"])
            
            logger.info(f"📥 Downloading {config['name']} ({hub_id})")
            
            with tqdm(total=total_splits, desc=f"Downloading {dataset_id}") as pbar:
                for split_name in config["splits"].keys():
                    try:
                        dataset = load_dataset(
                            hub_id,
                            split=split_name,
                            cache_dir=self.download_settings.get("cache_dir", "data/.cache"),
                            token=hf_token
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
            
            return success_count == total_splits
            
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_id}: {e}")
            return False
    
    def _convert_dataset(self, dataset_id: str) -> bool:
        """
        Convert raw dataset to standard format.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            bool: Success status
        """
        try:
            from .dataset_converters import run_by_config
            
            logger.info(f"🔄 Converting {dataset_id} to standard format...")
            
            # Run conversion
            results = run_by_config(
                source_name=dataset_id,
                config_path=str(self.config_path),
                output_base_dir=str(self.converted_base_dir)
            )
            
            if results and len(results) > 0:
                logger.info(f"✅ Successfully converted {dataset_id}")
                return True
            else:
                logger.error(f"❌ Conversion failed for {dataset_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error converting {dataset_id}: {e}")
            return False
    
    def build_dataset(
        self,
        dataset_id: str,
        format: Literal['query_doc', 'triple', 'reranking'] = 'query_doc',
        splits: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        negative_sampling: bool = True,
        negative_ratio: float = 1.0
    ) -> Dict[str, Dataset]:
        """
        Build dataset in specific format for training/evaluation.
        
        Args:
            dataset_id: Dataset identifier
            format: Output format ('query_doc', 'triple', 'reranking')
            splits: Specific splits to build (default: all available)
            max_samples: Maximum number of samples per split
            negative_sampling: Whether to include negative sampling for triplets
            negative_ratio: Ratio of negative to positive samples
            
        Returns:
            Dict[str, Dataset]: Built datasets by split
        """
        logger.info(f"🏗️ Building {format} dataset for {dataset_id}")
        
        # Ensure dataset is available
        if not self._ensure_dataset_available(dataset_id):
            raise RuntimeError(f"Could not make dataset '{dataset_id}' available")
        
        # Get available splits
        available_splits = self._get_available_splits(dataset_id)
        if splits is None:
            splits = [s for s in available_splits if s != 'corpus']
        
        built_datasets = {}
        
        for split in splits:
            if split not in available_splits:
                logger.warning(f"⚠️ Split '{split}' not available for {dataset_id}")
                continue
            
            logger.info(f"🔨 Building {split} split in {format} format...")
            
            if format == 'query_doc':
                dataset = self._build_query_doc_format(dataset_id, split, max_samples)
            elif format == 'triple':
                dataset = self._build_triple_format(
                    dataset_id, split, max_samples, negative_sampling, negative_ratio
                )
            elif format == 'reranking':
                dataset = self._build_reranking_format(dataset_id, split, max_samples)
            else:
                raise ValueError(f"Unknown format: {format}")
            
            if dataset is not None:
                built_datasets[split] = dataset
                logger.info(f"✅ Built {split}: {len(dataset):,} samples")
        
        logger.info(f"🎉 Dataset building completed: {len(built_datasets)} splits")
        return built_datasets
    
    def _get_available_splits(self, dataset_id: str) -> List[str]:
        """
        Get list of available splits for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            List[str]: Available split names
        """
        converted_path = self.converted_base_dir / dataset_id
        available_splits = []
        
        # Check standard splits
        for split_name in ['train', 'test', 'dev', 'corpus']:
            if split_name == 'corpus':
                split_file = converted_path / "corpus.csv"
            else:
                split_file = converted_path / f"{split_name}.csv"
            
            if split_file.exists():
                available_splits.append(split_name)
        
        return available_splits
    
    def _build_query_doc_format(
        self, 
        dataset_id: str, 
        split: str, 
        max_samples: Optional[int] = None
    ) -> Optional[Dataset]:
        """
        Build query-document pairs format.
        
        Args:
            dataset_id: Dataset identifier
            split: Split name
            max_samples: Maximum samples to include
            
        Returns:
            Dataset: Query-document pairs
        """
        try:
            # Load split data
            df = self.load_dataset(dataset_id, split, as_hf_dataset=False)
            
            # Load corpus if available
            corpus_df = None
            try:
                corpus_df = self.load_dataset(dataset_id, 'corpus', as_hf_dataset=False)
            except:
                logger.info("📝 No separate corpus found, using inline documents")
            
            query_doc_pairs = []
            
            for _, row in df.iterrows():
                query = row.get('question', '')
                
                if corpus_df is not None and 'cid' in row:
                    # Use corpus lookup
                    try:
                        cids = eval(row['cid']) if isinstance(row['cid'], str) else [row['cid']]
                        for cid in cids:
                            doc_row = corpus_df[corpus_df['cid'] == cid]
                            if not doc_row.empty:
                                document = doc_row.iloc[0]['text']
                                query_doc_pairs.append({
                                    'query': query,
                                    'document': document,
                                    'label': row.get('label', 1)
                                })
                    except:
                        # Fallback to context_list
                        if 'context_list' in row:
                            try:
                                contexts = eval(row['context_list']) if isinstance(row['context_list'], str) else [row['context_list']]
                                for doc in contexts:
                                    query_doc_pairs.append({
                                        'query': query,
                                        'document': doc,
                                        'label': row.get('label', 1)
                                    })
                            except:
                                pass
                else:
                    # Use inline documents
                    if 'context_list' in row:
                        try:
                            contexts = eval(row['context_list']) if isinstance(row['context_list'], str) else [row['context_list']]
                            for doc in contexts:
                                query_doc_pairs.append({
                                    'query': query,
                                    'document': doc,
                                    'label': row.get('label', 1)
                                })
                        except:
                            pass
            
            if max_samples and len(query_doc_pairs) > max_samples:
                query_doc_pairs = np.random.choice(query_doc_pairs, max_samples, replace=False).tolist()
            
            if query_doc_pairs:
                return Dataset.from_list(query_doc_pairs)
            else:
                logger.warning(f"⚠️ No query-doc pairs found for {dataset_id}/{split}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error building query-doc format: {e}")
            return None
    
    def _build_triple_format(
        self, 
        dataset_id: str, 
        split: str, 
        max_samples: Optional[int] = None,
        negative_sampling: bool = True,
        negative_ratio: float = 1.0
    ) -> Optional[Dataset]:
        """
        Build triplet format (query, positive_doc, negative_doc).
        
        Args:
            dataset_id: Dataset identifier
            split: Split name
            max_samples: Maximum samples to include
            negative_sampling: Whether to include negative sampling
            negative_ratio: Ratio of negatives to positives
            
        Returns:
            Dataset: Triplet format dataset
        """
        try:
            # Check if dataset already has triplet structure
            df = self.load_dataset(dataset_id, split, as_hf_dataset=False)
            
            # If already has negative labels, use existing structure
            if 'label' in df.columns and 'negative' in df['label'].values:
                return self._build_from_existing_triplets(df, max_samples)
            
            # Otherwise, build triplets from query-doc pairs
            corpus_df = None
            try:
                corpus_df = self.load_dataset(dataset_id, 'corpus', as_hf_dataset=False)
            except:
                logger.warning("📝 No corpus found for negative sampling")
                return None
            
            if corpus_df is None or len(corpus_df) < 2:
                logger.warning("⚠️ Insufficient corpus for negative sampling")
                return None
            
            triplets = []
            corpus_texts = corpus_df['text'].tolist()
            
            for _, row in df.iterrows():
                query = row.get('question', '')
                
                # Get positive documents
                positive_docs = []
                if 'context_list' in row:
                    try:
                        contexts = eval(row['context_list']) if isinstance(row['context_list'], str) else [row['context_list']]
                        positive_docs.extend(contexts)
                    except:
                        pass
                
                # Create triplets with negative sampling
                for pos_doc in positive_docs:
                    if negative_sampling:
                        # Sample negatives
                        available_negatives = [doc for doc in corpus_texts if doc != pos_doc]
                        n_negatives = min(int(negative_ratio), len(available_negatives))
                        
                        if n_negatives > 0:
                            negative_docs = np.random.choice(available_negatives, n_negatives, replace=False)
                            
                            for neg_doc in negative_docs:
                                triplets.append({
                                    'query': query,
                                    'positive': pos_doc,
                                    'negative': neg_doc
                                })
            
            if max_samples and len(triplets) > max_samples:
                triplets = np.random.choice(triplets, max_samples, replace=False).tolist()
            
            if triplets:
                return Dataset.from_list(triplets)
            else:
                logger.warning(f"⚠️ No triplets generated for {dataset_id}/{split}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error building triple format: {e}")
            return None
    
    def _build_from_existing_triplets(self, df: pd.DataFrame, max_samples: Optional[int] = None) -> Optional[Dataset]:
        """
        Build triplets from existing positive/negative labeled data.
        
        Args:
            df: DataFrame with label column
            max_samples: Maximum samples to include
            
        Returns:
            Dataset: Triplet format dataset
        """
        try:
            # Group by query to find positive/negative pairs
            triplets = []
            
            for query, group in df.groupby('question'):
                positive_docs = []
                negative_docs = []
                
                for _, row in group.iterrows():
                    if 'context_list' in row:
                        try:
                            contexts = eval(row['context_list']) if isinstance(row['context_list'], str) else [row['context_list']]
                            if row.get('label') == 'positive' or row.get('label') == 1:
                                positive_docs.extend(contexts)
                            elif row.get('label') == 'negative' or row.get('label') == 0:
                                negative_docs.extend(contexts)
                        except:
                            pass
                
                # Create triplets
                for pos_doc in positive_docs:
                    for neg_doc in negative_docs:
                        triplets.append({
                            'query': query,
                            'positive': pos_doc,
                            'negative': neg_doc
                        })
            
            if max_samples and len(triplets) > max_samples:
                triplets = np.random.choice(triplets, max_samples, replace=False).tolist()
            
            return Dataset.from_list(triplets) if triplets else None
            
        except Exception as e:
            logger.error(f"❌ Error building from existing triplets: {e}")
            return None
    
    def _build_reranking_format(
        self, 
        dataset_id: str, 
        split: str, 
        max_samples: Optional[int] = None
    ) -> Optional[Dataset]:
        """
        Build reranking format with relevance scores.
        
        Args:
            dataset_id: Dataset identifier
            split: Split name
            max_samples: Maximum samples to include
            
        Returns:
            Dataset: Reranking format dataset
        """
        try:
            # Similar to query_doc but preserve/generate relevance scores
            df = self.load_dataset(dataset_id, split, as_hf_dataset=False)
            
            reranking_data = []
            
            for _, row in df.iterrows():
                query = row.get('question', '')
                score = row.get('score', row.get('relevance_score', 1.0))
                
                if 'context_list' in row:
                    try:
                        contexts = eval(row['context_list']) if isinstance(row['context_list'], str) else [row['context_list']]
                        for doc in contexts:
                            reranking_data.append({
                                'query': query,
                                'document': doc,
                                'score': float(score)
                            })
                    except:
                        pass
            
            if max_samples and len(reranking_data) > max_samples:
                reranking_data = np.random.choice(reranking_data, max_samples, replace=False).tolist()
            
            return Dataset.from_list(reranking_data) if reranking_data else None
            
        except Exception as e:
            logger.error(f"❌ Error building reranking format: {e}")
            return None
    
    def create_combined_dataset(
        self,
        dataset_configs: List[Dict[str, str]],
        output_path: str,
        sample_ratio: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Combine multiple datasets into one.
        
        Args:
            dataset_configs: List of {"dataset_id": str, "split": str} configs
            output_path: Path to save combined dataset
            sample_ratio: Optional sampling ratio for each dataset
            
        Returns:
            Combined DataFrame
        """
        combined_data = []
        
        for config in dataset_configs:
            dataset_id = config["dataset_id"]
            split = config["split"]
            
            df = self.load_dataset(dataset_id, split, as_hf_dataset=False)
            
            # Add source information
            df["source_dataset"] = dataset_id
            df["source_split"] = split
            
            # Apply sampling if specified
            if sample_ratio and dataset_id in sample_ratio:
                ratio = sample_ratio[dataset_id]
                df = df.sample(frac=ratio, random_state=42)
                logger.info(f"📊 Sampled {len(df):,} rows from {dataset_id}/{split} (ratio: {ratio})")
            
            combined_data.append(df)
        
        # Combine all datasets
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Save combined dataset
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(output_path, index=False)
        
        logger.info(f"💾 Saved combined dataset: {len(combined_df):,} rows → {output_path}")
        
        return combined_df


def demo_usage():
    """Demonstrate how to use DatasetManager."""
    print("🔍 === DATASET MANAGER DEMO ===")
    
    manager = DatasetManager(auto_download=True, auto_convert=True)
    
    # List available datasets
    print("\n📋 Available datasets:")
    for dataset_id in manager.list_available_datasets():
        info = manager.get_dataset_info(dataset_id)
        status = "✅ Available" if info["available"] else "❌ Not downloaded"
        print(f"  {dataset_id}: {info['name']} - {status}")
    
    # Try to load a dataset (with auto-download)
    try:
        dataset_id = "vn_legal_retrieval"
        split = "train"
        
        print(f"\n📖 Loading {dataset_id}/{split}...")
        dataset = manager.load_dataset(dataset_id, split)
        print(f"  Loaded {len(dataset):,} rows")
        
        # Show sample
        print(f"\n🔍 Sample data:")
        sample = manager.get_sample_data(dataset_id, split, n_samples=2)
        print(sample.head())
        
        # Demo dataset building
        print(f"\n🏗️ Building datasets in different formats...")
        
        # Query-Doc format
        print("\n📋 Building Query-Document pairs:")
        query_doc_datasets = manager.build_dataset(dataset_id, format='query_doc', max_samples=100)
        for split_name, dataset in query_doc_datasets.items():
            print(f"  {split_name}: {len(dataset):,} query-doc pairs")
            if len(dataset) > 0:
                print(f"    Sample: {dataset[0]}")
        
        # Triple format
        print("\n🔗 Building Triplets:")
        try:
            triple_datasets = manager.build_dataset(dataset_id, format='triple', max_samples=50)
            for split_name, dataset in triple_datasets.items():
                print(f"  {split_name}: {len(dataset):,} triplets")
                if len(dataset) > 0:
                    print(f"    Sample keys: {list(dataset[0].keys())}")
        except Exception as e:
            print(f"  ⚠️ Triplet building failed: {e}")
        
        # Show statistics
        print(f"\n📊 Dataset statistics:")
        stats = manager.get_dataset_statistics(dataset_id)
        for split_name, split_stats in stats["splits"].items():
            print(f"  {split_name}: {split_stats['rows']:,} rows, {split_stats['size_mb']} MB")
        
    except Exception as e:
        print(f"❌ Could not load dataset: {e}")
        print("💡 The manager should auto-download and convert datasets")


if __name__ == "__main__":
    demo_usage()
