#!/usr/bin/env python3
"""
Dataset Converters

Mỗi dạng dataset có converter function riêng để convert về VNLegalDataset format.
Support run theo config hoặc run từng file cụ thể.

Usage:
    # Run theo config
    python data_process/dataset_converters.py --config --source vn_legal_retrieval
    python data_process/dataset_converters.py --config --source all
    
    # Run từng file
    python data_process/dataset_converters.py --file data/file.parquet --type legal_qa --output data/converted/custom
    python data_process/dataset_converters.py --file data/file.csv --type vn_legal --output data/converted/custom
"""

import json
import pandas as pd
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def convert_legal_qa(input_path: str, output_dir: str, source_name: str = "legal_qa") -> Dict[str, str]:
    """
    Convert Legal QA dataset (query-document pairs) to VNLegalDataset format.
    
    Expected input format:
    - query, document columns
    - Or question, context columns
    - Or similar variations
    
    Args:
        input_path: Path to input file/directory
        output_dir: Output directory
        source_name: Source name for metadata
        
    Returns:
        Dict[str, str]: Created output files
    """
    logger.info(f"⚖️ Converting Legal QA: {input_path}")
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    corpus_data = []
    cid_counter = 0
    
    # Handle single file or directory
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = list(input_path.glob("*.parquet")) + list(input_path.glob("*.csv"))
    
    if not files_to_process:
        logger.error(f"❌ No files found in {input_path}")
        return {}
    
    logger.info(f"📂 Found {len(files_to_process)} files to process")
    
    for file_path in files_to_process:
        split_name = file_path.stem
        logger.info(f"📖 Processing {split_name}: {file_path}")
        
        # Load data
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            logger.warning(f"⚠️ Unsupported file format: {file_path}")
            continue
        
        logger.info(f"   Loaded {len(df):,} rows")
        logger.info(f"   Columns: {list(df.columns)}")
        
        # Convert to VNLegalDataset format
        converted_data = []
        
        for idx, row in df.iterrows():
            # Extract query and document with flexible column names
            query = _extract_text_flexible(row, ['query', 'question', 'text', 'anchor'])
            document = _extract_text_flexible(row, ['document', 'context', 'passage', 'positive', 'text'])
            
            if not query or not document:
                continue
            
            # Add to corpus
            doc_cid = cid_counter
            corpus_data.append({
                "cid": doc_cid,
                "text": document
            })
            cid_counter += 1
            
            # Create QA entry
            converted_data.append({
                "qid": len(converted_data),
                "question": query,
                "context_list": f'["{document}"]',  # String representation of list
                "cid": f"[{doc_cid}]"  # String representation of list
            })
        
        # Save split file với standard naming
        if converted_data:
            standard_name = _map_to_standard_split(split_name)
            split_file = output_dir / f"{standard_name}.csv"
            split_df = pd.DataFrame(converted_data)
            split_df.to_csv(split_file, index=False)
            output_files[standard_name] = str(split_file)
            logger.info(f"✅ Saved {standard_name}: {len(converted_data):,} samples → {split_file}")
    
    # Save corpus
    if corpus_data:
        corpus_file = output_dir / "corpus.csv"
        corpus_df = pd.DataFrame(corpus_data)
        # Remove duplicates based on text
        corpus_df = corpus_df.drop_duplicates(subset=['text']).reset_index(drop=True)
        corpus_df['cid'] = range(len(corpus_df))  # Reassign CIDs after dedup
        corpus_df.to_csv(corpus_file, index=False)
        output_files["corpus"] = str(corpus_file)
        logger.info(f"✅ Saved corpus: {len(corpus_df):,} unique documents → {corpus_file}")
    
    # Save metadata
    _save_metadata(output_dir, {
        "converter": "legal_qa",
        "source_name": source_name,
        "input_path": str(input_path),
        "output_files": output_files,
        "total_samples": sum(len(pd.read_csv(f)) for f in output_files.values() if "corpus" not in f),
        "corpus_size": len(corpus_data) if corpus_data else 0
    })
    
    return output_files


def convert_triplets(input_path: str, output_dir: str, source_name: str = "triplets") -> Dict[str, str]:
    """
    Convert Triplet dataset (anchor, positive, negative) to VNLegalDataset format.
    
    Expected input format:
    - anchor, positive, negative columns
    - Or query, positive, negative columns
    
    Args:
        input_path: Path to input file/directory
        output_dir: Output directory  
        source_name: Source name for metadata
        
    Returns:
        Dict[str, str]: Created output files
    """
    logger.info(f"🔗 Converting Triplets: {input_path}")
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    corpus_data = []
    cid_counter = 0
    
    # Handle single file or directory
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = list(input_path.glob("*.parquet")) + list(input_path.glob("*.csv"))
    
    if not files_to_process:
        logger.error(f"❌ No files found in {input_path}")
        return {}
    
    for file_path in files_to_process:
        split_name = file_path.stem
        logger.info(f"📖 Processing {split_name}: {file_path}")
        
        # Load data
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            continue
        
        logger.info(f"   Loaded {len(df):,} rows")
        logger.info(f"   Columns: {list(df.columns)}")
        
        converted_data = []
        
        for idx, row in df.iterrows():
            # Extract triplet components
            anchor = _extract_text_flexible(row, ['anchor', 'query', 'question'])
            positive = _extract_text_flexible(row, ['positive', 'pos', 'document'])
            negative = _extract_text_flexible(row, ['negative', 'neg', 'hard_negative'])
            
            if not all([anchor, positive, negative]):
                continue
            
            # Add positive document to corpus
            pos_cid = cid_counter
            corpus_data.append({"cid": pos_cid, "text": positive})
            cid_counter += 1
            
            # Add negative document to corpus  
            neg_cid = cid_counter
            corpus_data.append({"cid": neg_cid, "text": negative})
            cid_counter += 1
            
            # Create positive pair
            converted_data.append({
                "qid": len(converted_data),
                "question": anchor,
                "context_list": f'["{positive}"]',
                "cid": f"[{pos_cid}]",
                "label": "positive"
            })
            
            # Create negative pair (for hard negative training)
            converted_data.append({
                "qid": len(converted_data),
                "question": anchor,
                "context_list": f'["{negative}"]',
                "cid": f"[{neg_cid}]",
                "label": "negative"
            })
        
        # Save split file với standard naming
        if converted_data:
            standard_name = _map_to_standard_split(split_name)
            split_file = output_dir / f"{standard_name}.csv"
            split_df = pd.DataFrame(converted_data)
            split_df.to_csv(split_file, index=False)
            output_files[standard_name] = str(split_file)
            logger.info(f"✅ Saved {standard_name}: {len(converted_data):,} samples → {split_file}")
    
    # Save corpus
    if corpus_data:
        corpus_file = output_dir / "corpus.csv"
        corpus_df = pd.DataFrame(corpus_data)
        # Remove duplicates
        corpus_df = corpus_df.drop_duplicates(subset=['text']).reset_index(drop=True)
        corpus_df['cid'] = range(len(corpus_df))
        corpus_df.to_csv(corpus_file, index=False)
        output_files["corpus"] = str(corpus_file)
        logger.info(f"✅ Saved corpus: {len(corpus_df):,} unique documents → {corpus_file}")
    
    # Save metadata
    _save_metadata(output_dir, {
        "converter": "triplets",
        "source_name": source_name,
        "input_path": str(input_path),
        "output_files": output_files,
        "total_samples": sum(len(pd.read_csv(f)) for f in output_files.values() if "corpus" not in f),
        "corpus_size": len(corpus_data) if corpus_data else 0
    })
    
    return output_files


def convert_reranking(input_path: str, output_dir: str, source_name: str = "reranking") -> Dict[str, str]:
    """
    Convert Reranking dataset to VNLegalDataset format.
    
    Expected input format:
    - query, document, relevance_score columns
    - Or similar variations with scores
    
    Args:
        input_path: Path to input file/directory
        output_dir: Output directory
        source_name: Source name for metadata
        
    Returns:
        Dict[str, str]: Created output files
    """
    logger.info(f"🔀 Converting Reranking: {input_path}")
    
    # Reranking is similar to legal_qa but preserves relevance scores
    return convert_legal_qa(input_path, output_dir, source_name)


def convert_vn_legal_csv(input_path: str, output_dir: str, source_name: str = "vn_legal") -> Dict[str, str]:
    """
    Convert existing VN Legal CSV files to standardized VNLegalDataset format.
    
    Expected input format:
    - corpus: cid, text
    - qa: qid, question, context_list, cid
    
    Args:
        input_path: Path to input file/directory
        output_dir: Output directory
        source_name: Source name for metadata
        
    Returns:
        Dict[str, str]: Created output files
    """
    logger.info(f"📋 Converting VN Legal CSV: {input_path}")
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # Handle single file or directory
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = list(input_path.glob("*.csv"))
    
    if not files_to_process:
        logger.error(f"❌ No CSV files found in {input_path}")
        return {}
    
    for csv_file in files_to_process:
        logger.info(f"📖 Processing: {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"   Loaded {len(df):,} rows")
            logger.info(f"   Columns: {list(df.columns)}")
            
            # Detect file type and convert
            if _is_corpus_file(df):
                output_file = _convert_corpus_csv(df, output_dir, csv_file.stem)
                if output_file:
                    output_files["corpus"] = output_file
            
            elif _is_qa_file(df):
                output_file = _convert_qa_csv(df, output_dir, csv_file.stem)
                if output_file:
                    output_files[csv_file.stem] = output_file
            
            else:
                logger.warning(f"⚠️ Unknown CSV format: {csv_file.name}")
                logger.info(f"   Available columns: {list(df.columns)}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {csv_file.name}: {e}")
    
    # Save metadata
    if output_files:
        _save_metadata(output_dir, {
            "converter": "vn_legal_csv",
            "source_name": source_name,
            "input_path": str(input_path),
            "output_files": output_files,
            "total_files": len(files_to_process)
        })
    
    return output_files


def convert_benchmark(input_path: str, output_dir: str, source_name: str = "benchmark") -> Dict[str, str]:
    """
    Convert Benchmark dataset to VNLegalDataset format.
    
    Expected input format:
    - Similar to legal_qa but may have additional benchmark-specific columns
    
    Args:
        input_path: Path to input file/directory
        output_dir: Output directory
        source_name: Source name for metadata
        
    Returns:
        Dict[str, str]: Created output files
    """
    logger.info(f"📊 Converting Benchmark: {input_path}")
    
    # Benchmark datasets are typically similar to legal_qa
    return convert_legal_qa(input_path, output_dir, source_name)


# Helper functions
def _map_to_standard_split(split_name: str) -> str:
    """Map various split names to standard format."""
    split_name = split_name.lower()
    
    # Standard mappings
    if split_name in ['train', 'training', 'train_data']:
        return 'train'
    elif split_name in ['test', 'testing', 'test_data']:
        return 'test'
    elif split_name in ['dev', 'development', 'validation', 'valid', 'val', 'dev_data']:
        return 'dev'
    elif split_name in ['corpus', 'corpus_data', 'documents', 'docs']:
        return 'corpus'
    else:
        # Default mapping for unknown split names
        if 'train' in split_name:
            return 'train'
        elif 'test' in split_name:
            return 'test'
        elif 'dev' in split_name or 'val' in split_name:
            return 'dev'
        else:
            return split_name  # Keep original if can't map


def _extract_text_flexible(row: pd.Series, possible_columns: List[str]) -> Optional[str]:
    """Extract text from row using flexible column name matching."""
    for col in possible_columns:
        if col in row and pd.notna(row[col]):
            text = str(row[col]).strip()
            if text and text.lower() not in ['nan', 'none', '']:
                return text
    return None


def _is_corpus_file(df: pd.DataFrame) -> bool:
    """Check if dataframe is corpus format."""
    required_cols = ['cid', 'text']
    return all(col in df.columns for col in required_cols)


def _is_qa_file(df: pd.DataFrame) -> bool:
    """Check if dataframe is QA format."""
    required_cols = ['qid', 'question']
    return all(col in df.columns for col in required_cols)


def _convert_corpus_csv(df: pd.DataFrame, output_dir: Path, filename: str) -> Optional[str]:
    """Convert corpus CSV to standard format."""
    logger.info(f"📚 Converting corpus CSV...")
    
    output_file = output_dir / "corpus.csv"
    
    # Ensure required columns
    if 'cid' not in df.columns:
        df['cid'] = range(len(df))
    
    if 'text' not in df.columns:
        logger.error("❌ No 'text' column found in corpus file")
        return None
    
    # Clean and save
    clean_df = df[['cid', 'text']].copy()
    clean_df['text'] = clean_df['text'].astype(str).str.strip()
    clean_df = clean_df[clean_df['text'] != ''].reset_index(drop=True)
    
    clean_df.to_csv(output_file, index=False)
    logger.info(f"✅ Saved corpus: {len(clean_df):,} documents → {output_file}")
    
    return str(output_file)


def _convert_qa_csv(df: pd.DataFrame, output_dir: Path, filename: str) -> Optional[str]:
    """Convert QA CSV to standard format."""
    logger.info(f"❓ Converting QA CSV...")
    
    # Map filename to standard split name
    standard_name = _map_to_standard_split(filename)
    output_file = output_dir / f"{standard_name}.csv"
    
    # Ensure required columns
    if 'qid' not in df.columns:
        df['qid'] = range(len(df))
    
    if 'question' not in df.columns:
        logger.error("❌ No 'question' column found in QA file")
        return None
    
    # Handle context_list and cid columns
    if 'context_list' not in df.columns:
        context_col = _extract_column_flexible(df, ['context', 'document', 'passage', 'text'])
        if context_col:
            df['context_list'] = df[context_col].apply(lambda x: f'["{x}"]')
        else:
            logger.error("❌ No context column found")
            return None
    
    if 'cid' not in df.columns:
        df['cid'] = range(len(df))
        df['cid'] = df['cid'].apply(lambda x: f"[{x}]")
    
    # Select and clean data
    clean_df = df[['qid', 'question', 'context_list', 'cid']].copy()
    clean_df['question'] = clean_df['question'].astype(str).str.strip()
    clean_df = clean_df[clean_df['question'] != ''].reset_index(drop=True)
    
    clean_df.to_csv(output_file, index=False)
    logger.info(f"✅ Saved QA: {len(clean_df):,} samples → {output_file}")
    
    return str(output_file)


def _extract_column_flexible(df: pd.DataFrame, possible_columns: List[str]) -> Optional[str]:
    """Find first available column from possible names."""
    for col in possible_columns:
        if col in df.columns:
            return col
    return None


def _save_metadata(output_dir: Path, metadata: Dict[str, Any]):
    """Save conversion metadata."""
    metadata_file = output_dir / "conversion_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Metadata saved: {metadata_file}")


# Converter registry
CONVERTERS = {
    "legal_qa": convert_legal_qa,
    "triplets": convert_triplets,
    "reranking": convert_reranking,
    "benchmark": convert_benchmark,
    "vn_legal": convert_vn_legal_csv,
}


def run_by_config(source_name: str, config_path: str = "data_process/dataset_configs.json", output_base_dir: str = "clean_data"):
    """Run conversion based on dataset config."""
    logger.info(f"🔧 Running conversion by config for: {source_name}")
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    datasets = config["datasets"]
    
    if source_name == "all":
        # Convert all datasets
        results = {}
        for dataset_id, dataset_config in datasets.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Converting {dataset_id}")
            logger.info(f"{'='*60}")
            
            try:
                result = _convert_single_by_config(dataset_id, dataset_config, output_base_dir)
                results[dataset_id] = result
            except Exception as e:
                logger.error(f"❌ {dataset_id}: {e}")
                results[dataset_id] = {}
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("CONVERSION SUMMARY")
        logger.info(f"{'='*60}")
        
        successful = sum(1 for r in results.values() if r)
        logger.info(f"📊 Total: {len(results)}, Successful: {successful}")
        
        return results
    
    else:
        # Convert single dataset
        if source_name not in datasets:
            logger.error(f"❌ Source '{source_name}' not found in config")
            return {}
        
        dataset_config = datasets[source_name]
        return _convert_single_by_config(source_name, dataset_config, output_base_dir)


def _convert_single_by_config(dataset_id: str, dataset_config: Dict, output_base_dir: str) -> Dict[str, str]:
    """Convert single dataset based on config."""
    input_path = dataset_config["local_path"]
    output_dir = Path(output_base_dir) / dataset_id
    use_case = dataset_config.get("use_case", "unknown")
    
    logger.info(f"📂 Input: {input_path}")
    logger.info(f"📂 Output: {output_dir}")
    logger.info(f"🎯 Use case: {use_case}")
    
    # Check if input exists
    if not Path(input_path).exists():
        logger.error(f"❌ Input path not found: {input_path}")
        return {}
    
    # Map use case to converter
    converter_map = {
        "legal_qa": "legal_qa",
        "training_triplets": "triplets", 
        "reranking": "reranking",
        "benchmark_retrieval": "benchmark",
        "unknown": "legal_qa"  # Default fallback
    }
    
    converter_type = converter_map.get(use_case, "legal_qa")
    converter_func = CONVERTERS[converter_type]
    
    logger.info(f"🔧 Using converter: {converter_type}")
    
    return converter_func(input_path, str(output_dir), dataset_id)


def run_by_file(file_path: str, converter_type: str, output_dir: str, source_name: str = "custom"):
    """Run conversion for specific file."""
    logger.info(f"📁 Running conversion by file:")
    logger.info(f"   File: {file_path}")
    logger.info(f"   Type: {converter_type}")
    logger.info(f"   Output: {output_dir}")
    
    if converter_type not in CONVERTERS:
        logger.error(f"❌ Unknown converter type: {converter_type}")
        logger.info(f"Available types: {list(CONVERTERS.keys())}")
        return {}
    
    if not Path(file_path).exists():
        logger.error(f"❌ File not found: {file_path}")
        return {}
    
    converter_func = CONVERTERS[converter_type]
    return converter_func(file_path, output_dir, source_name)


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description="Convert datasets to VNLegalDataset format")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--config", action="store_true", help="Run by config")
    mode_group.add_argument("--file", help="Run by specific file")
    
    # Config mode options
    parser.add_argument("--source", help="Source name (for config mode)")
    parser.add_argument("--config-path", default="data_process/dataset_configs.json", help="Config file path")
    parser.add_argument("--output-base", default="data/converted", help="Base output directory (for config mode)")
    
    # File mode options
    parser.add_argument("--type", help="Converter type (for file mode)")
    parser.add_argument("--output", help="Output directory (for file mode)")
    parser.add_argument("--name", default="custom", help="Source name (for file mode)")
    
    # General options
    parser.add_argument("--list-types", action="store_true", help="List available converter types")
    parser.add_argument("--list-sources", action="store_true", help="List available sources in config")
    
    args = parser.parse_args()
    
    # List options
    if args.list_types:
        print("📋 Available converter types:")
        for conv_type in CONVERTERS.keys():
            print(f"   {conv_type}")
        return
    
    if args.list_sources:
        try:
            with open(args.config_path, 'r') as f:
                config = json.load(f)
            print("📋 Available sources in config:")
            for source_id, source_config in config["datasets"].items():
                use_case = source_config.get("use_case", "unknown")
                exists = "✅" if Path(source_config["local_path"]).exists() else "❌"
                print(f"   {exists} {source_id}: {use_case}")
        except Exception as e:
            print(f"❌ Error reading config: {e}")
        return
    
    # Mode execution
    if args.config:
        if not args.source:
            print("❌ --source required for config mode")
            return
        
        results = run_by_config(args.source, args.config_path, args.output_base)
        
        if results:
            print(f"\n🎉 Conversion completed! Check: {args.output_base}")
        else:
            print("❌ Conversion failed")
    
    elif args.file:
        if not args.type or not args.output:
            print("❌ --type and --output required for file mode")
            return
        
        results = run_by_file(args.file, args.type, args.output, args.name)
        
        if results:
            print(f"\n🎉 File conversion completed! Check: {args.output}")
            for file_type, file_path in results.items():
                print(f"   {file_type}: {file_path}")
        else:
            print("❌ File conversion failed")


if __name__ == "__main__":
    main()
