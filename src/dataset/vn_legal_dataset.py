"""
Vietnamese Legal Dataset for SPLADE

Single class dataset that loads corpus, test, and train data
and provides get_dataset() method to return Hugging Face Dataset.
"""

import pandas as pd
import ast
from typing import Dict, List, Optional, Literal
from pathlib import Path
from datasets import Dataset


class VNLegalDataset:
    """
    Vietnamese Legal dataset class for SPLADE training.
    
    Loads corpus, test, and train data and provides iteration over 
    query-document pairs in simple dict format.
    """
    
    def __init__(
        self, 
        corpus_path: str,
        test_path: Optional[str] = None,
        train_path: Optional[str] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            corpus_path (str): Path to corpus CSV file
            test_path (str, optional): Path to test CSV file
            train_path (str, optional): Path to train CSV file
        """
        self.corpus_path = corpus_path
        self.test_path = test_path
        self.train_path = train_path
        
        # Load data
        self._load_data()
        
        # Prepare samples with new structure
        self._prepare_samples()
        
    def _load_data(self):
        """Load all data files."""
        print("📚 Loading data files...")
        
        # Load corpus
        self.corpus_df = pd.read_csv(self.corpus_path)
        print(f"   - Corpus: {len(self.corpus_df):,} documents")
        
        # Create document ID to text mapping
        self.doc_id_to_text = {}
        for _, row in self.corpus_df.iterrows():
            doc_id = row['cid']
            doc_text = str(row['text'])
            self.doc_id_to_text[doc_id] = doc_text
        
        # Load test data if provided
        self.test_df = None
        if self.test_path and Path(self.test_path).exists():
            self.test_df = pd.read_csv(self.test_path)
            print(f"   - Test: {len(self.test_df):,} questions")
        
        # Load train data if provided  
        self.train_df = None
        if self.train_path and Path(self.train_path).exists():
            self.train_df = pd.read_csv(self.train_path)
            print(f"   - Train: {len(self.train_df):,} questions")
        
        print("✅ Data loaded successfully!")
    
    def _prepare_samples(self):
        """Prepare all query-document samples with structured format."""
        print("🔄 Preparing samples...")
        
        # Initialize structured data
        self.data = {
            "test": [],
            "train": [],
            "dev": []
        }
        
        # Process test data
        if self.test_df is not None:
            self.data["test"] = self._process_qa_data(self.test_df, "test")
        
        # Process train data
        if self.train_df is not None:
            self.data["train"] = self._process_qa_data(self.train_df, "train")
        
        # For compatibility, keep a flat samples list
        self.samples = []
        for split_name, split_data in self.data.items():
            self.samples.extend(split_data)
        
        total_samples = sum(len(split_data) for split_data in self.data.values())
        print(f"✅ Prepared {total_samples:,} query-document pairs")
        print(f"   - Test: {len(self.data['test']):,}")
        print(f"   - Train: {len(self.data['train']):,}")
        print(f"   - Dev: {len(self.data['dev']):,}")
    
    def _process_qa_data(self, qa_df: pd.DataFrame, split: str) -> List[Dict]:
        """Process QA data and create query-document pairs."""
        split_samples = []
        
        for _, row in qa_df.iterrows():
            question = str(row['question'])
            qid = row['qid']
            
            # Parse context list (stored as string representation of list)
            try:
                context_list = ast.literal_eval(row['context_list'])
                if not isinstance(context_list, list):
                    context_list = [str(context_list)]
            except:
                context_list = [str(row['context_list'])]
            
            # Parse CID list - handle different formats
            cid_raw = str(row['cid'])
            try:
                # Try to parse as list first
                if cid_raw.startswith('[') and cid_raw.endswith(']'):
                    # Handle numpy-style arrays like '[63069 63069]'
                    if ' ' in cid_raw and ',' not in cid_raw:
                        # Convert space-separated to comma-separated
                        cid_raw = cid_raw.replace(' ', ', ')
                    cid_list = ast.literal_eval(cid_raw)
                    if not isinstance(cid_list, list):
                        cid_list = [int(cid_list)]
                else:
                    # Single value
                    cid_list = [int(cid_raw)]
            except:
                # Fallback: try to extract numbers
                import re
                numbers = re.findall(r'\d+', cid_raw)
                cid_list = [int(num) for num in numbers] if numbers else [0]
            
            # Create samples for each context-document pair
            for context, cid in zip(context_list, cid_list):
                # Get document from corpus if available
                document_text = self.doc_id_to_text.get(cid, str(context))
                
                sample = {
                    'query': question,
                    'document': document_text,
                    'qid': qid,
                    'cid': cid,
                    'split': split
                }
                
                split_samples.append(sample)
        
        return split_samples
    
    def get_dataset(self, split: Literal["train", "test", "dev"]) -> Dataset:
        """
        Get Hugging Face Dataset for a specific split.
        
        Args:
            split: Dataset split ("train", "test", or "dev")
            
        Returns:
            Dataset: Hugging Face Dataset with 'query' and 'document' columns
        """
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found. Available: {list(self.data.keys())}")
        
        split_data = self.data[split]
        
        # Extract queries and documents
        queries = []
        documents = []
        
        for sample in split_data:
            queries.append(sample['query'])
            documents.append(sample['document'])
        
        # Create Hugging Face Dataset
        dataset = Dataset.from_dict({
            "query": queries,
            "document": documents,
        })
        
        print(f"Created {split} dataset with {len(dataset)} samples")
        return dataset
    
    def __getitem__(self, split: str) -> Dataset:
        """
        Get dataset using bracket notation: dataset["test"]
        
        Args:
            split: Dataset split ("train", "test", or "dev")
            
        Returns:
            Dataset: Hugging Face Dataset with 'query' and 'document' columns
        """
        return self.get_dataset(split)
    


# Example usage
if __name__ == "__main__":
    # Simple usage example
    dataset = VNLegalDataset(
        corpus_path="data/vn-legal-doc/cleaned_corpus_data.csv",
        test_path="data/vn-legal-doc/test_data.csv"
    )
    
    # Get Hugging Face datasets - both ways work!
    try:
        # Method 1: Using get_dataset()
        test_dataset = dataset.get_dataset("test")
        print(f"Method 1 - Test dataset: {test_dataset}")
        
        # Method 2: Using bracket notation (new!)
        test_dataset_v2 = dataset["test"]
        print(f"Method 2 - Test dataset: {test_dataset_v2}")
        
        print(f"Features: {test_dataset.features}")
        
        # Show first sample
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            print(f"First sample:")
            print(f"  Query: {sample['query'][:100]}...")
            print(f"  Document: {sample['document'][:100]}...")
        
        # Try both methods for train dataset
        train_dataset = dataset.get_dataset("train")
        train_dataset_v2 = dataset["train"]
        print(f"Train dataset (method 1): {train_dataset}")
        print(f"Train dataset (method 2): {train_dataset_v2}")
        
    except Exception as e:
        print(f"Error: {e}")