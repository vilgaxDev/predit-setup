import pandas as pd

df = pd.read_excel('anonymized_data_kg_export.xlsx', engine='openpyxl')
print(f'Records: {len(df)}')
print(f'Columns: {len(df.columns)}')
print(f'Column names: {list(df.columns)}')
print(f'\nData types:')
print(df.dtypes)
print(f'\nFirst few rows:')
print(df.head())
