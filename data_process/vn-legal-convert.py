#!/usr/bin/env python3
"""
Simple Data Converter for Vietnamese Legal Documents

Chuyển đổi dữ liệu đơn giản:
- Corpus -> JSON
- Test data -> CSV  
"""

import pandas as pd
import json
from pathlib import Path
import sys


def convert_corpus_to_json(input_path: str, output_path: str = None, limit: int = None):
    """
    Chuyển đổi corpus data sang JSON
    
    Args:
        input_path (str): Đường dẫn file parquet corpus
        output_path (str): Đường dẫn file JSON output (tùy chọn)
        limit (int): Giới hạn số bản ghi (tùy chọn)
    """
    print(f"📖 Đang đọc corpus data từ: {input_path}")
    
    # Đọc dữ liệu
    df = pd.read_parquet(input_path)
    
    if limit:
        df = df.head(limit)
        print(f"📝 Lấy {limit} bản ghi đầu tiên")
    
    # Tạo tên file output nếu không có
    if output_path is None:
        output_path = "corpus_data.json"
    
    # Chuyển đổi sang JSON với format đẹp
    print(f"🔄 Đang chuyển đổi {len(df)} bản ghi sang JSON...")
    
    # Convert to JSON với indent để dễ đọc
    json_data = df.to_json(orient='records', force_ascii=False, indent=2)
    
    # Ghi file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_data)
    
    print(f"✅ Hoàn thành! File JSON đã lưu tại: {output_path}")
    print(f"📊 Số lượng: {len(df)} văn bản pháp luật")
    return output_path


def convert_test_to_csv(input_path: str, output_path: str = None, limit: int = None):
    """
    Chuyển đổi test data sang CSV
    
    Args:
        input_path (str): Đường dẫn file parquet test
        output_path (str): Đường dẫn file CSV output (tùy chọn)
        limit (int): Giới hạn số bản ghi (tùy chọn)
    """
    print(f"📖 Đang đọc test data từ: {input_path}")
    
    # Đọc dữ liệu
    df = pd.read_parquet(input_path)
    
    if limit:
        df = df.head(limit)
        print(f"📝 Lấy {limit} bản ghi đầu tiên")
    
    # Tạo tên file output nếu không có
    if output_path is None:
        output_path = "test_data.csv"
    
    # Chuyển đổi sang CSV
    print(f"🔄 Đang chuyển đổi {len(df)} bản ghi sang CSV...")
    
    # Xử lý cột context_list nếu có (convert array thành string)
    if 'context_list' in df.columns:
        df['context_list'] = df['context_list'].astype(str)
    
    # Ghi file CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ Hoàn thành! File CSV đã lưu tại: {output_path}")
    print(f"📊 Số lượng: {len(df)} câu hỏi test")
    return output_path


def convert_train_to_csv(input_path: str, output_path: str = None, limit: int = None):
    """
    Chuyển đổi train data sang CSV
    
    Args:
        input_path (str): Đường dẫn file parquet train
        output_path (str): Đường dẫn file CSV output (tùy chọn)
        limit (int): Giới hạn số bản ghi (tùy chọn)
    """
    print(f"📖 Đang đọc train data từ: {input_path}")
    
    # Đọc dữ liệu
    df = pd.read_parquet(input_path)
    
    if limit:
        df = df.head(limit)
        print(f"📝 Lấy {limit} bản ghi đầu tiên")
    
    # Tạo tên file output nếu không có
    if output_path is None:
        output_path = "train_data.csv"
    
    # Chuyển đổi sang CSV
    print(f"🔄 Đang chuyển đổi {len(df)} bản ghi sang CSV...")
    
    # Xử lý cột context_list nếu có (convert array thành string)
    if 'context_list' in df.columns:
        df['context_list'] = df['context_list'].astype(str)
    
    # Ghi file CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ Hoàn thành! File CSV đã lưu tại: {output_path}")
    print(f"📊 Số lượng: {len(df)} câu hỏi train")
    return output_path


def convert_corpus_to_csv(input_path: str, output_path: str = None, limit: int = None):
    """
    Chuyển đổi corpus data sang CSV (thay vì JSON)
    
    Args:
        input_path (str): Đường dẫn file parquet corpus
        output_path (str): Đường dẫn file CSV output (tùy chọn)
        limit (int): Giới hạn số bản ghi (tùy chọn)
    """
    print(f"📖 Đang đọc corpus data từ: {input_path}")
    
    # Đọc dữ liệu
    df = pd.read_parquet(input_path)
    
    if limit:
        df = df.head(limit)
        print(f"📝 Lấy {limit} bản ghi đầu tiên")
    
    # Tạo tên file output nếu không có
    if output_path is None:
        output_path = "corpus_data.csv"
    
    # Chuyển đổi sang CSV
    print(f"🔄 Đang chuyển đổi {len(df)} bản ghi sang CSV...")
    
    # Ghi file CSV với encoding UTF-8
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ Hoàn thành! File CSV đã lưu tại: {output_path}")
    print(f"📊 Số lượng: {len(df)} văn bản pháp luật")
    return output_path


def process_corpus_normalization(input_path: str, output_path: str = None):
    """
    Chuẩn hóa corpus data
    
    Args:
        input_path (str): Đường dẫn file parquet corpus  
        output_path (str): Đường dẫn file CSV output đã chuẩn hóa
    """
    import re
    
    print(f"🔧 Đang chuẩn hóa corpus từ: {input_path}")
    
    # Đọc dữ liệu
    df = pd.read_parquet(input_path)
    
    if output_path is None:
        output_path = "corpus_normalized.csv"
    
    print(f"🔄 Đang chuẩn hóa {len(df)} văn bản...")
    
    # Chuẩn hóa text
    def normalize_text(text):
        if pd.isna(text):
            return ""
        
        # Chuyển về string nếu chưa phải
        text = str(text)
        
        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text)
        
        # Loại bỏ khoảng trắng đầu cuối
        text = text.strip()
        
        # Loại bỏ ký tự đặc biệt không mong muốn (giữ lại dấu câu cơ bản)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\-\"\']', ' ', text)
        
        # Loại bỏ khoảng trắng thừa lần nữa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # Áp dụng chuẩn hóa cho cột text
    df['text_normalized'] = df['text'].apply(normalize_text)
    
    # Loại bỏ các dòng có text rỗng sau khi chuẩn hóa
    df = df[df['text_normalized'].str.len() > 0]
    
    # Tạo DataFrame cuối cùng với cột đã chuẩn hóa
    df_final = df[['cid', 'text', 'text_normalized']].copy()
    
    # Ghi file CSV
    df_final.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✅ Hoàn thành! File CSV đã chuẩn hóa lưu tại: {output_path}")
    print(f"📊 Số lượng sau chuẩn hóa: {len(df_final)} văn bản")
    print(f"🗑️  Đã loại bỏ: {len(df) - len(df_final)} văn bản rỗng")
    
    return output_path


def main():
    """Hàm chính để chuyển đổi tất cả dữ liệu"""
    
    print("🚀 === CHUYỂN ĐỔI DỮ LIỆU VN-LEGAL-DOC ===")
    print()
    
    # Đường dẫn dữ liệu
    data_dir = Path("data/vn-legal-doc")
    corpus_file = data_dir / "corpus_data.parquet"
    test_file = data_dir / "test_data.parquet"
    train_file = data_dir / "train_data.parquet"
    
    # Kiểm tra file tồn tại
    if not corpus_file.exists():
        print(f"❌ Không tìm thấy file corpus: {corpus_file}")
        return
    
    if not test_file.exists():
        print(f"❌ Không tìm thấy file test: {test_file}")
        return
        
    if not train_file.exists():
        print(f"❌ Không tìm thấy file train: {train_file}")
        return
    
    # Tạo thư mục output
    output_dir = Path("converted_data")
    output_dir.mkdir(exist_ok=True)
    
    print("📁 Tạo thư mục output: converted_data/")
    print()
    
    # 1. Chuyển đổi Corpus -> JSON
    print("1️⃣ CHUYỂN ĐỔI CORPUS -> JSON")
    print("-" * 40)
    
    corpus_json = output_dir / "corpus_data.json"
    convert_corpus_to_json(
        str(corpus_file), 
        str(corpus_json)
        # Bỏ limit để lấy tất cả dữ liệu
    )
    print()
    
    # 2. Chuyển đổi Test -> CSV  
    print("2️⃣ CHUYỂN ĐỔI TEST DATA -> CSV")
    print("-" * 40)
    
    test_csv = output_dir / "test_data.csv"
    convert_test_to_csv(
        str(test_file),
        str(test_csv)
        # Bỏ limit để lấy tất cả dữ liệu
    )
    print()
    
    # 3. Chuyển đổi Train -> CSV
    print("3️⃣ CHUYỂN ĐỔI TRAIN DATA -> CSV")
    print("-" * 40)
    
    train_csv = output_dir / "train_data.csv"
    convert_train_to_csv(
        str(train_file),
        str(train_csv)
        # Bỏ limit để lấy tất cả dữ liệu
    )
    print()
    
    # Tóm tắt
    print("🎉 === HOÀN THÀNH CHUYỂN ĐỔI ===")
    print()
    print("📁 Files đã tạo:")
    print(f"   📄 {corpus_json} - Corpus dạng JSON")
    print(f"   📄 {test_csv} - Test data dạng CSV")
    print(f"   📄 {train_csv} - Train data dạng CSV")
    print()
    print("💡 Bạn có thể:")
    print("   - Mở file JSON bằng text editor hoặc VS Code")
    print("   - Mở file CSV bằng Excel, Google Sheets")
    print("   - Sử dụng trong code Python với pandas")
    print()


def convert_corpus_csv_only():
    """Hàm để chỉ chuyển corpus sang CSV"""
    
    print("🚀 === CHUYỂN ĐỔI CORPUS SANG CSV ===")
    print()
    
    # Đường dẫn dữ liệu
    data_dir = Path("data/vn-legal-doc")
    corpus_file = data_dir / "corpus_data.parquet"
    
    # Kiểm tra file tồn tại
    if not corpus_file.exists():
        print(f"❌ Không tìm thấy file corpus: {corpus_file}")
        return
    
    # Tạo thư mục output
    output_dir = Path("converted_data")
    output_dir.mkdir(exist_ok=True)
    
    print("📁 Sử dụng thư mục: converted_data/")
    print()
    
    # Chuyển đổi Corpus -> CSV
    print("📄 CHUYỂN ĐỔI CORPUS -> CSV")
    print("-" * 40)
    
    corpus_csv = output_dir / "corpus_data.csv"
    convert_corpus_to_csv(
        str(corpus_file), 
        str(corpus_csv)
    )
    print()
    
    # Chuẩn hóa corpus
    print("🔧 CHUẨN HÓA CORPUS")
    print("-" * 40)
    
    corpus_normalized = output_dir / "corpus_normalized.csv"
    process_corpus_normalization(
        str(corpus_file),
        str(corpus_normalized)
    )
    print()
    
    # Tóm tắt
    print("🎉 === HOÀN THÀNH ===")
    print()
    print("📁 Files đã tạo:")
    print(f"   📄 {corpus_csv} - Corpus dạng CSV gốc")
    print(f"   📄 {corpus_normalized} - Corpus đã chuẩn hóa")
    print()


if __name__ == "__main__":
    import sys
    
    try:
        # Nếu có argument "corpus", chỉ chuyển corpus
        if len(sys.argv) > 1 and sys.argv[1] == "corpus":
            convert_corpus_csv_only()
        else:
            main()
    except KeyboardInterrupt:
        print("\\n⏹️  Dừng chuyển đổi")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
