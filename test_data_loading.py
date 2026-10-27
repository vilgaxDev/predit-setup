#!/usr/bin/env python3
"""Test script to verify data files can be loaded"""
import pandas as pd
from pathlib import Path
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Test loading data files
LOCAL_DATA_DIR = Path(__file__).parent

data_files = {
    "maternity": "COMBINE_MOH_333_3replaced_KCHR.xlsx",
    "anc1": "MOH_405_ANC final JAN 2026.xlsx",
    "anc2": "MOH_405_ANC_20 version 2.xlsx",
    "under5": "MOH_204A_UNDER5 revised version on jan 2026.xlsx",
    "anonymized": "anonymized_data_kg_export.xlsx"
}

print(f"Working directory: {Path.cwd()}")
print(f"LOCAL_DATA_DIR: {LOCAL_DATA_DIR}")
print(f"Files in directory:")
for f in LOCAL_DATA_DIR.glob("*.xlsx"):
    print(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")

print("\n" + "="*50)
print("Testing file loading...")
print("="*50)

for key, filename in data_files.items():
    filepath = LOCAL_DATA_DIR / filename
    print(f"\n{key}: {filename}")
    print(f"  Path: {filepath}")
    print(f"  Exists: {filepath.exists()}")

    if filepath.exists():
        try:
            all_sheets = pd.read_excel(filepath, sheet_name=None, engine='openpyxl')
            print(f"  [OK] Loaded successfully!")
            print(f"  Sheets: {list(all_sheets.keys())}")
            for sheet_name, df in all_sheets.items():
                print(f"    - {sheet_name}: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"  [ERROR] Error loading: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  [ERROR] File not found")

print("\n" + "="*50)
print("Test complete")
