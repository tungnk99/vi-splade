# Vietnamese Legal Document Data Processing

Thư viện xử lý dữ liệu văn bản pháp luật Việt Nam với các tính năng làm sạch, chuẩn hóa và chuyển đổi định dạng.

## Tính năng chính

### 1. VNLegalDocProcessor - Xử lý văn bản
- **Làm sạch văn bản**: Loại bỏ ký tự điều khiển, chuẩn hóa Unicode cho tiếng Việt
- **Chuẩn hóa cấu trúc pháp lý**: Điều, Chương, Mục, Khoản
- **Tách câu thông minh**: Nhận biết cấu trúc pháp lý đặc biệt
- **Trích xuất thành phần pháp lý**: Tự động nhận diện các điều khoản, chương mục

### 2. DataConverter - Chuyển đổi định dạng  
- **Parquet → JSON**: Định dạng có cấu trúc, dễ đọc
- **Parquet → CSV**: Tương thích với Excel, Google Sheets
- **Parquet → TXT**: Văn bản thuần túy với metadata
- **Tạo tóm tắt**: Báo cáo thống kê tổng quan

### 3. DataAnalyzer - Phân tích dữ liệu
- **Thống kê tổng quan**: Kích thước, cột, chất lượng dữ liệu
- **Phân tích văn bản**: Từ vựng, độ dài, mật độ thuật ngữ pháp lý
- **So sánh dataset**: Corpus vs Test data
- **Báo cáo chi tiết**: Phân tích toàn diện xuất ra file

## Cài đặt

```bash
pip install pandas pyarrow numpy
```

## Sử dụng nhanh

### Chuyển đổi dữ liệu sang định dạng dễ đọc

```python
from data_process import DataConverter

# Khởi tạo converter
converter = DataConverter(output_dir="output")

# Chuyển đổi tất cả định dạng
files = converter.convert_all_formats(
    "data/vn-legal-doc/corpus_data.parquet",
    text_column="text",
    limit=100  # Chỉ lấy 100 bản ghi đầu
)

print("Các file đã tạo:", files)
```

### Xử lý và làm sạch văn bản

```python
from data_process import VNLegalDocProcessor
import pandas as pd

# Khởi tạo processor
processor = VNLegalDocProcessor()

# Đọc dữ liệu
df = pd.read_parquet("data/vn-legal-doc/corpus_data.parquet")

# Xử lý batch
processed_df = processor.process_corpus_batch(df.head(10), "text")

# Kết quả có thêm cột: processed_text, word_count, sentence_count
print(processed_df.columns)
```

### Phân tích dữ liệu

```python
from data_process import DataAnalyzer
import pandas as pd

# Khởi tạo analyzer
analyzer = DataAnalyzer()

# Đọc dữ liệu
corpus_df = pd.read_parquet("data/vn-legal-doc/corpus_data.parquet")
test_df = pd.read_parquet("data/vn-legal-doc/test_data.parquet")

# Phân tích tổng quan
overview = analyzer.analyze_dataset_overview(corpus_df)
print(f"Tổng số văn bản: {overview['basic_info']['total_records']}")

# Phân tích văn bản chi tiết
text_analysis = analyzer.analyze_text_content(corpus_df, "text")
print(f"Từ vựng độc nhất: {text_analysis['vocabulary_analysis']['unique_words']}")

# So sánh datasets
comparison = analyzer.analyze_corpus_vs_test(corpus_df, test_df)

# Tạo báo cáo
analyzer.generate_analysis_report({
    'corpus': overview,
    'text_analysis': text_analysis,
    'comparison': comparison
}, "analysis_report.txt")
```

## Chạy demo hoàn chỉnh

```bash
cd data_process
python example_usage.py
```

Script này sẽ:
1. ✅ Chuyển đổi dữ liệu sang JSON, CSV, TXT
2. ✅ Làm sạch và chuẩn hóa văn bản
3. ✅ Phân tích thống kê chi tiết
4. ✅ Trích xuất thành phần pháp lý
5. ✅ Tạo báo cáo tổng hợp

## Cấu trúc thư mục sau khi chạy

```
processed_data/
├── corpus_data.json           # Dữ liệu corpus dạng JSON
├── corpus_data.csv            # Dữ liệu corpus dạng CSV  
├── corpus_data_text.txt       # Văn bản thuần với metadata
├── corpus_data_summary.txt    # Tóm tắt thống kê corpus
├── test_data.json            # Dữ liệu test dạng JSON
├── test_data.csv             # Dữ liệu test dạng CSV
├── test_data_text.txt        # Văn bản test thuần
├── test_data_summary.txt     # Tóm tắt thống kê test
└── comprehensive_analysis_report.txt  # Báo cáo phân tích đầy đủ
```

## Tính năng nâng cao

### Trích xuất thành phần pháp lý

```python
processor = VNLegalDocProcessor()

# Xử lý một văn bản với trích xuất thành phần
result = processor.process_document(text, extract_elements=True)

# Kết quả chứa:
# - cleaned_text: văn bản đã làm sạch
# - normalized_text: văn bản đã chuẩn hóa  
# - sentences: danh sách câu
# - legal_elements: các thành phần pháp lý (Điều, Chương, Mục...)
# - statistics: thống kê từ, câu, ký tự
```

### Tùy chỉnh patterns pháp lý

```python
processor = VNLegalDocProcessor()

# Thêm pattern tùy chỉnh
processor.legal_patterns['custom_pattern'] = r'Nghị định\\s+\\d+/\\d+/.*'

# Sử dụng để trích xuất
elements = processor.extract_legal_elements(text)
```

## Lưu ý

- Tất cả các file output sử dụng encoding UTF-8 để đảm bảo hiển thị đúng tiếng Việt
- Dữ liệu được làm sạch và chuẩn hóa theo tiêu chuẩn Unicode NFC  
- Hỗ trợ xử lý batch cho hiệu suất cao với dữ liệu lớn
- Các báo cáo phân tích được xuất ra định dạng text dễ đọc

## Hỗ trợ

Thư viện được thiết kế đặc biệt cho văn bản pháp luật Việt Nam với:
- Nhận diện cấu trúc pháp lý (Điều, Chương, Mục...)
- Từ điển thuật ngữ pháp lý Việt Nam
- Xử lý Unicode tiếng Việt chính xác
- Phân tích thống kê phù hợp với văn bản pháp lý


