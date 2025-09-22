# Dataset Manager - Enhanced Version

Dataset Manager đã được nâng cấp với các tính năng mới mạnh mẽ:

## 🚀 Tính năng mới

### 1. Auto-Download từ Hugging Face Hub
- Tự động download dataset từ HF nếu chưa có ở local
- Hỗ trợ authentication với HF_TOKEN
- Download song song với progress bar

### 2. Auto-Convert sang định dạng chuẩn
- Tự động convert dataset đã download sang định dạng VNLegalDataset
- Mapping thông minh các format khác nhau
- Tạo corpus và split files chuẩn

### 3. Build Dataset theo format
- **Query-Document pairs**: Cho training retrieval models
- **Triplets**: Query + Positive + Negative docs
- **Reranking**: Query + Document + Relevance score

### 4. Quản lý thông qua configs
- Load dataset configurations từ JSON
- Mapping automatic giữa use cases và converters
- Hỗ trợ nhiều dataset sources

## 📖 Cách sử dụng

### Basic Usage

```python
from data_process.dataset_manager import DatasetManager

# Khởi tạo với auto features
manager = DatasetManager(
    auto_download=True,    # Tự động download nếu thiếu
    auto_convert=True      # Tự động convert sang format chuẩn
)

# Load dataset (sẽ auto-download nếu cần)
dataset = manager.load_dataset('vn_legal_retrieval', 'train')
print(f"Loaded {len(dataset)} samples")
```

### Building Different Formats

```python
# 1. Query-Document pairs
query_doc_datasets = manager.build_dataset(
    dataset_id='vn_legal_retrieval',
    format='query_doc',
    max_samples=1000
)

# 2. Triplets với negative sampling
triple_datasets = manager.build_dataset(
    dataset_id='vinli_triplet', 
    format='triple',
    negative_sampling=True,
    negative_ratio=2.0  # 2 negatives per positive
)

# 3. Reranking format
reranking_datasets = manager.build_dataset(
    dataset_id='vn_legal_retrieval',
    format='reranking'
)
```

### Advanced Usage

```python
# Kiểm tra dataset info
info = manager.get_dataset_info('vinli_triplet')
print(f"Available: {info['available']}")
print(f"Splits: {info.get('split_info', {}).keys()}")

# Get statistics
stats = manager.get_dataset_statistics('vn_legal_retrieval')
print(f"Total rows: {stats['total_rows']}")
print(f"Size: {stats['total_size_mb']} MB")

# List available datasets
available = manager.list_available_datasets()
print(f"Configured datasets: {available}")
```

## 🔧 Configuration

Dataset configurations trong `dataset_configs.json`:

```json
{
  "datasets": {
    "vn_legal_retrieval": {
      "name": "Vietnamese Legal Document Retrieval Data",
      "description": "Vietnamese legal document retrieval dataset for question-answering",
      "hub_id": "YuITC/Vietnamese-Legal-Doc-Retrieval-Data",
      "splits": {
        "train": "train_data.parquet",
        "test": "test_data.parquet"
      },
      "local_path": "data/vn_legal_retrieval",
      "file_format": "parquet",
      "use_case": "legal_qa",
      "language": "vi"
    }
  },
  "download_settings": {
    "cache_dir": "data/.cache",
    "require_auth": true,
    "parallel_downloads": 4,
    "verify_checksums": true
  }
}
```

## 🏗️ Dataset Formats

### Query-Document Pairs
```python
{
    'query': 'Câu hỏi về luật...',
    'document': 'Nội dung tài liệu pháp lý...',
    'label': 1  # Relevance score
}
```

### Triplets
```python
{
    'query': 'Câu hỏi về luật...',
    'positive': 'Tài liệu liên quan...',
    'negative': 'Tài liệu không liên quan...'
}
```

### Reranking
```python
{
    'query': 'Câu hỏi về luật...',
    'document': 'Tài liệu để rank...',
    'score': 0.85  # Relevance score
}
```

## 🎯 Use Cases

### 1. Training Retrieval Models
```python
# Lấy query-doc pairs cho training
datasets = manager.build_dataset('vn_legal_retrieval', format='query_doc')
train_data = datasets['train']

# Sử dụng với PyTorch DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(train_data, batch_size=32)
```

### 2. Training với Triplet Loss
```python
# Lấy triplets cho contrastive learning
triplets = manager.build_dataset('vinli_triplet', format='triple')
train_triplets = triplets['train']

# Có sẵn query, positive, negative
for batch in train_triplets:
    query = batch['query']
    positive = batch['positive'] 
    negative = batch['negative']
    # Train model với triplet loss
```

### 3. Evaluation với Reranking
```python
# Lấy data với scores để evaluate
rerank_data = manager.build_dataset('vn_legal_retrieval', format='reranking')
test_data = rerank_data['test']

# Evaluate ranking performance
for item in test_data:
    predicted_score = model.predict(item['query'], item['document'])
    true_score = item['score']
    # Compute ranking metrics
```

## 🔄 Auto-Download Flow

1. **Check Local**: Kiểm tra dataset có sẵn local không
2. **Download**: Nếu không có và `auto_download=True`, download từ HF
3. **Convert**: Nếu có raw data và `auto_convert=True`, convert sang format chuẩn
4. **Load**: Load dataset đã convert

## 🌟 Environment Setup

```bash
# Set HF token (optional, for private datasets)
export HF_TOKEN="your_huggingface_token"

# Or create .env file
echo "HF_TOKEN=your_huggingface_token" > .env
```

## 📝 Examples

Xem file `examples/example_dataset_manager.py` để demo đầy đủ các tính năng:

```bash
python examples/example_dataset_manager.py
```

## 🚀 Quick Start

```python
# All-in-one example
from data_process.dataset_manager import DatasetManager

manager = DatasetManager(auto_download=True, auto_convert=True)

# Build training data
train_data = manager.build_dataset(
    'vn_legal_retrieval', 
    format='query_doc',
    splits=['train']
)['train']

# Ready to use!
print(f"Training data: {len(train_data)} samples")
print(f"Sample: {train_data[0]}")
```

Với Dataset Manager nâng cấp, việc quản lý và sử dụng datasets trở nên đơn giản và linh hoạt hơn rất nhiều! 🎉