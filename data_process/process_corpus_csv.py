#!/usr/bin/env python3
"""
Xử lý Corpus CSV - Làm sạch dữ liệu corpus từ CSV và xuất ra CSV

Input: corpus_data.csv
Output: cleaned_corpus_data.csv (cấu trúc cột giống input, chỉ thay text đã làm sạch)
"""

import pandas as pd
import re
import unicodedata
from pathlib import Path


def clean_vietnamese_text(text):
    """
    Làm sạch văn bản tiếng Việt
    
    Args:
        text (str): Văn bản gốc
        
    Returns:
        str: Văn bản đã làm sạch
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Chuẩn hóa Unicode (quan trọng cho tiếng Việt)
    text = unicodedata.normalize('NFC', text.strip())
    
    # Loại bỏ ký tự điều khiển
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)
    
    # Loại bỏ khoảng trắng thừa ở đầu và cuối
    text = text.strip()
    
    # Chuẩn hóa dấu câu
    text = re.sub(r'\.{3,}', '...', text)  # Nhiều dấu chấm -> ba chấm
    text = re.sub(r'[!]{2,}', '!', text)   # Nhiều dấu ! -> một dấu !
    text = re.sub(r'[?]{2,}', '?', text)   # Nhiều dấu ? -> một dấu ?
    text = re.sub(r'[-]{2,}', '-', text)   # Nhiều dấu - -> một dấu -
    
    return text


def normalize_legal_structure(text):
    """
    Chuẩn hóa cấu trúc văn bản pháp luật
    
    Args:
        text (str): Văn bản chứa cấu trúc pháp luật
        
    Returns:
        str: Văn bản đã chuẩn hóa cấu trúc
    """
    # Chuẩn hóa tham chiếu Điều
    text = re.sub(r'Điều\s+(\d+)([a-z]*)', r'Điều \1\2', text)
    
    # Chuẩn hóa tham chiếu Chương
    text = re.sub(r'Chương\s+([IVXLCDM]+)', r'Chương \1', text)
    
    # Chuẩn hóa tham chiếu Mục
    text = re.sub(r'Mục\s+(\d+)', r'Mục \1', text)
    
    # Chuẩn hóa khoản (số + dấu chấm + khoảng trắng)
    text = re.sub(r'(\d+)\s*\.\s+', r'\1. ', text)
    
    # Chuẩn hóa điểm (chữ cái + dấu ngoặc đóng + khoảng trắng)
    text = re.sub(r'([a-z])\s*\)\s+', r'\1) ', text)
    
    return text


def process_corpus_csv(input_csv="corpus_data.csv", output_csv="cleaned_corpus_data.csv", text_column="text"):
    """
    Xử lý corpus từ file CSV và xuất ra CSV đã làm sạch
    Giữ nguyên cấu trúc cột như input, chỉ thay thế nội dung cột text
    
    Args:
        input_csv (str): Đường dẫn file CSV đầu vào
        output_csv (str): Đường dẫn file CSV đầu ra
        text_column (str): Tên cột chứa văn bản
    """
    
    print("🏛️ === XỬ LÝ CORPUS CSV PHÁP LUẬT VIỆT NAM ===")
    print()
    
    # Kiểm tra file đầu vào
    if not Path(input_csv).exists():
        print(f"❌ Không tìm thấy file: {input_csv}")
        return
    
    # Đọc CSV
    print(f"📖 Đọc corpus từ: {input_csv}")
    try:
        df = pd.read_csv(input_csv, encoding='utf-8')
    except UnicodeDecodeError:
        # Thử với encoding khác nếu utf-8 không được
        df = pd.read_csv(input_csv, encoding='latin-1')
    
    print(f"📊 Kích thước corpus: {df.shape}")
    print(f"📝 Các cột: {list(df.columns)}")
    
    # Kiểm tra cột text
    if text_column not in df.columns:
        print(f"❌ Không tìm thấy cột '{text_column}' trong file CSV")
        print(f"💡 Các cột có sẵn: {list(df.columns)}")
        return
    
    print()
    print(f"🚀 Bắt đầu xử lý {len(df):,} văn bản...")
    
    # Tạo DataFrame copy để giữ nguyên cấu trúc
    processed_df = df.copy()
    
    # Danh sách lưu văn bản đã làm sạch
    cleaned_texts = []
    empty_count = 0
    total_words = 0
    total_chars = 0
    
    # Xử lý từng văn bản
    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"📝 Đã xử lý: {idx:,}/{len(df):,} văn bản ({idx/len(df)*100:.1f}%)")
        
        # Lấy văn bản gốc
        original_text = str(row[text_column])
        
        # Làm sạch văn bản
        cleaned_text = clean_vietnamese_text(original_text)
        
        # Chuẩn hóa cấu trúc pháp luật
        if cleaned_text:
            cleaned_text = normalize_legal_structure(cleaned_text)
            # Tính thống kê
            total_words += len(cleaned_text.split())
            total_chars += len(cleaned_text)
        else:
            empty_count += 1
        
        # Lưu văn bản đã làm sạch
        cleaned_texts.append(cleaned_text)
    
    # Thay thế cột text bằng văn bản đã làm sạch
    processed_df[text_column] = cleaned_texts
    
    # Tạo thư mục output nếu cần
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Lưu CSV đã xử lý với cấu trúc cột giống input
    print(f"💾 Lưu corpus đã xử lý tại: {output_csv}")
    processed_df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # Thống kê kết quả
    valid_count = len(df) - empty_count
    avg_words = total_words / valid_count if valid_count > 0 else 0
    avg_chars = total_chars / valid_count if valid_count > 0 else 0
    
    print()
    print("✅ === HOÀN THÀNH XỬ LÝ CORPUS ===")
    print()
    print("📊 Thống kê:")
    print(f"   • Tổng số văn bản: {len(df):,}")
    print(f"   • Văn bản hợp lệ: {valid_count:,}")
    print(f"   • Văn bản trống: {empty_count:,}")
    print(f"   • Tỷ lệ hợp lệ: {valid_count/len(df)*100:.1f}%")
    print(f"   • Trung bình từ/văn bản: {avg_words:.1f}")
    print(f"   • Trung bình ký tự/văn bản: {avg_chars:.1f}")
    print()
    print("📁 Files được tạo:")
    print(f"   📄 {output_csv} - Corpus đã làm sạch")
    print()
    print("💡 Cấu trúc file output:")
    print(f"   - Giữ nguyên tất cả cột như input: {list(df.columns)}")
    print(f"   - Cột '{text_column}' chứa văn bản đã làm sạch")
    print("   - Không thêm cột mới")
    
    return processed_df


def main():
    """Hàm chính để chạy xử lý corpus"""
    
    # Tìm file corpus_data.csv
    possible_paths = [
        "corpus_data.csv",
        "data/vn-legal-doc/corpus_data.csv",
        "converted_data/corpus_data.csv"
    ]
    
    input_file = None
    for path in possible_paths:
        if Path(path).exists():
            input_file = path
            break
    
    if input_file:
        # Tạo tên file output dựa trên input
        if "data/vn-legal-doc/" in input_file:
            output_file = "data/vn-legal-doc/cleaned_corpus_data.csv"
        else:
            output_file = "cleaned_corpus_data.csv"
        
        process_corpus_csv(input_file, output_file)
    else:
        print(f"❌ Không tìm thấy file corpus_data.csv trong các vị trí:")
        for path in possible_paths:
            print(f"   - {path}")
        print("💡 Hãy đảm bảo file corpus_data.csv tồn tại")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
