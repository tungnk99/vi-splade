#!/usr/bin/env python3
"""
Run Conversions Script

Easy interface để run conversions với options khác nhau.
Supports both config-based và file-based conversions.

Usage:
    # Run theo config
    python data_process/run_conversions.py config --source vn_legal_retrieval
    python data_process/run_conversions.py config --source all
    
    # Run từng file  
    python data_process/run_conversions.py file --input data/file.parquet --type legal_qa --output data/converted/custom
    
    # Interactive mode
    python data_process/run_conversions.py interactive
    
    # List options
    python data_process/run_conversions.py list --types
    python data_process/run_conversions.py list --sources
"""

import sys
import json
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_process.dataset_converters import (
    run_by_config, 
    run_by_file, 
    CONVERTERS
)


def interactive_mode():
    """Interactive conversion mode."""
    print("🔧 === INTERACTIVE CONVERSION MODE ===")
    print()
    
    # Choose mode
    print("Choose conversion mode:")
    print("1. Config-based (use dataset_configs.json)")
    print("2. File-based (convert specific file)")
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "1":
        interactive_config_mode()
    elif choice == "2":
        interactive_file_mode()
    else:
        print("❌ Invalid choice")


def interactive_config_mode():
    """Interactive config-based conversion."""
    print("\n📋 === CONFIG-BASED CONVERSION ===")
    
    # Load and show available sources
    try:
        with open("data_process/dataset_configs.json", 'r') as f:
            config = json.load(f)
        
        print("\nAvailable sources:")
        sources = []
        for i, (source_id, source_config) in enumerate(config["datasets"].items(), 1):
            use_case = source_config.get("use_case", "unknown")
            exists = "✅" if Path(source_config["local_path"]).exists() else "❌"
            print(f"{i:2d}. {exists} {source_id}: {use_case}")
            sources.append(source_id)
        
        print(f"{len(sources)+1:2d}. all (convert all sources)")
        
        # Get user choice
        choice = input(f"\nEnter choice (1-{len(sources)+1}): ").strip()
        
        try:
            choice_num = int(choice)
            if choice_num == len(sources) + 1:
                source_name = "all"
            elif 1 <= choice_num <= len(sources):
                source_name = sources[choice_num - 1]
            else:
                print("❌ Invalid choice")
                return
        except ValueError:
            print("❌ Invalid choice")
            return
        
        # Get output directory
        output_dir = input("Output directory (default: clean_data): ").strip()
        if not output_dir:
            output_dir = "clean_data"
        
        # Run conversion
        print(f"\n🔄 Converting {source_name}...")
        results = run_by_config(source_name, output_base_dir=output_dir)
        
        if results:
            print(f"\n✅ Conversion completed! Check: {output_dir}")
        else:
            print("❌ Conversion failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def interactive_file_mode():
    """Interactive file-based conversion."""
    print("\n📁 === FILE-BASED CONVERSION ===")
    
    # Get input file
    input_file = input("Input file path: ").strip()
    if not input_file or not Path(input_file).exists():
        print("❌ File not found")
        return
    
    # Show available converter types
    print("\nAvailable converter types:")
    converter_list = list(CONVERTERS.keys())
    for i, conv_type in enumerate(converter_list, 1):
        print(f"{i}. {conv_type}")
    
    # Get converter type choice
    choice = input(f"\nEnter choice (1-{len(converter_list)}): ").strip()
    
    try:
        choice_num = int(choice)
        if 1 <= choice_num <= len(converter_list):
            converter_type = converter_list[choice_num - 1]
        else:
            print("❌ Invalid choice")
            return
    except ValueError:
        print("❌ Invalid choice")
        return
    
    # Get output directory
    output_dir = input("Output directory: ").strip()
    if not output_dir:
        print("❌ Output directory required")
        return
    
    # Get source name
    source_name = input("Source name (default: custom): ").strip()
    if not source_name:
        source_name = "custom"
    
    # Run conversion
    print(f"\n🔄 Converting {input_file}...")
    results = run_by_file(input_file, converter_type, output_dir, source_name)
    
    if results:
        print(f"\n✅ Conversion completed! Check: {output_dir}")
        for file_type, file_path in results.items():
            print(f"   {file_type}: {file_path}")
    else:
        print("❌ Conversion failed")


def list_types():
    """List available converter types."""
    print("📋 Available converter types:")
    
    # Load converter configs if available
    try:
        with open("data_process/converter_configs.json", 'r') as f:
            configs = json.load(f)
        
        for conv_type, config in configs["converter_types"].items():
            print(f"\n🔧 {conv_type}:")
            print(f"   Name: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   Expected columns: {config['expected_columns']}")
            print(f"   Use cases: {', '.join(config['use_cases'])}")
    
    except FileNotFoundError:
        # Fallback to simple list
        for conv_type in CONVERTERS.keys():
            print(f"   {conv_type}")


def list_sources():
    """List available sources in config."""
    print("📋 Available sources in config:")
    
    try:
        with open("data_process/dataset_configs.json", 'r') as f:
            config = json.load(f)
        
        for source_id, source_config in config["datasets"].items():
            use_case = source_config.get("use_case", "unknown")
            local_path = source_config["local_path"]
            exists = "✅" if Path(local_path).exists() else "❌"
            
            print(f"\n{exists} {source_id}:")
            print(f"   Use case: {use_case}")
            print(f"   Path: {local_path}")
            print(f"   Description: {source_config.get('description', 'N/A')}")
    
    except FileNotFoundError:
        print("❌ Config file not found: data_process/dataset_configs.json")
    except Exception as e:
        print(f"❌ Error reading config: {e}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Run dataset conversions")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Config mode
    config_parser = subparsers.add_parser("config", help="Run by config")
    config_parser.add_argument("--source", required=True, help="Source name or 'all'")
    config_parser.add_argument("--config-path", default="data_process/dataset_configs.json", help="Config file path")
    config_parser.add_argument("--output", default="clean_data", help="Output base directory")
    
    # File mode
    file_parser = subparsers.add_parser("file", help="Run by file")
    file_parser.add_argument("--input", required=True, help="Input file path")
    file_parser.add_argument("--type", required=True, help="Converter type")
    file_parser.add_argument("--output", required=True, help="Output directory")
    file_parser.add_argument("--name", default="custom", help="Source name")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Interactive mode")
    
    # List mode
    list_parser = subparsers.add_parser("list", help="List available options")
    list_group = list_parser.add_mutually_exclusive_group(required=True)
    list_group.add_argument("--types", action="store_true", help="List converter types")
    list_group.add_argument("--sources", action="store_true", help="List sources in config")
    
    args = parser.parse_args()
    
    if args.command == "config":
        print(f"🔧 Running config-based conversion for: {args.source}")
        results = run_by_config(args.source, args.config_path, args.output)
        
        if results:
            print(f"\n🎉 Conversion completed! Check: {args.output}")
        else:
            print("❌ Conversion failed")
    
    elif args.command == "file":
        print(f"📁 Running file-based conversion:")
        print(f"   Input: {args.input}")
        print(f"   Type: {args.type}")
        print(f"   Output: {args.output}")
        
        results = run_by_file(args.input, args.type, args.output, args.name)
        
        if results:
            print(f"\n🎉 Conversion completed! Check: {args.output}")
            for file_type, file_path in results.items():
                print(f"   {file_type}: {file_path}")
        else:
            print("❌ Conversion failed")
    
    elif args.command == "interactive":
        interactive_mode()
    
    elif args.command == "list":
        if args.types:
            list_types()
        elif args.sources:
            list_sources()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
