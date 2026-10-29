"""
check_data_files.py
====================
Quick script to check which data files exist and their locations.
"""

from pathlib import Path

# Define the search locations
locations = [
    Path(__file__).parent,  # python_pipeline
    Path(__file__).parent.parent,  # dashboard
    Path(__file__).parent.parent.parent,  # sarah
]

# Files to check
files_to_check = [
    "COMBINE_MOH_333_3replaced_KCHR.xlsx",
    "MOH_405_ANC final JAN 2026.xlsx",
    "MOH_405_ANC_20 version 2.xlsx",
    "MOH_204A_UNDER5 revised version on jan 2026.xlsx",
    "anonymized_data_kg_export.xlsx"
]

print("=" * 80)
print("DATA FILE CHECK")
print("=" * 80)

for filename in files_to_check:
    print(f"\nFILE: {filename}")
    found = False
    for loc in locations:
        filepath = loc / filename
        if filepath.exists():
            size = filepath.stat().st_size / 1024  # KB
            print(f"  [OK] Found at: {filepath}")
            print(f"       Size: {size:.2f} KB")
            found = True
            break
    if not found:
        print(f"  [X] Not found in any location")

print("\n" + "=" * 80)
print("SEARCHED LOCATIONS:")
print("=" * 80)
for loc in locations:
    print(f"  - {loc}")
