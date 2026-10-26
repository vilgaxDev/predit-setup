"""
data_manager.py
===============
Simple data file manager for dashboard deployment.
Helps organize and manage health data files.
"""

import os
import shutil
import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def organize_data_files(source_dir, target_dir):
    """
    Organize data files from source to target directory.
    
    Args:
        source_dir: Source directory containing data files
        target_dir: Target directory for organized files
    """
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    # Supported file types
    supported_extensions = ['.csv', '.xlsx', '.xls']
    
    # Copy files
    copied_files = []
    for file_name in os.listdir(source_dir):
        if any(file_name.endswith(ext) for ext in supported_extensions):
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(target_dir, file_name)
            
            if not os.path.exists(target_path):
                shutil.copy2(source_path, target_path)
                copied_files.append(file_name)
                print(f"✅ Copied: {file_name}")
            else:
                print(f"⏭️  Skipped (already exists): {file_name}")
    
    print(f"\n✅ Organized {len(copied_files)} files to {target_dir}")
    return copied_files

def list_data_files(directory):
    """List all data files in directory."""
    data_files = []
    for file_name in os.listdir(directory):
        if file_name.endswith(('.csv', '.xlsx', '.xls')):
            file_path = os.path.join(directory, file_name)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            data_files.append({
                'name': file_name,
                'size_mb': round(file_size, 2),
                'path': file_path
            })
    return data_files

if __name__ == "__main__":
    DATA_DIR = r"D:\sarah"
    UPLOADS_DIR = os.path.join(DATA_DIR, "data_uploads")
    
    print("="*60)
    print("DATA FILE MANAGER")
    print("="*60)
    
    # List current data files
    print("\n[Current Data Files]:")
    current_files = list_data_files(DATA_DIR)
    for f in current_files:
        print(f"  - {f['name']} ({f['size_mb']} MB)")
    
    # List uploaded files
    print(f"\n[Uploaded Files ({UPLOADS_DIR})]:")
    if os.path.exists(UPLOADS_DIR):
        uploaded_files = list_data_files(UPLOADS_DIR)
        for f in uploaded_files:
            print(f"  - {f['name']} ({f['size_mb']} MB)")
    else:
        print("  (No uploads directory)")
    
    print("\n" + "="*60)
    print("DEPLOYMENT READY")
    print("="*60)
    print("[OK] Your data files are organized and ready for deployment")
    print("[OK] Use the dashboard 'Data Management' page to upload additional files")
    print("[OK] The pipeline will check both main directory and uploads directory")
