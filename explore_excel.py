"""
Script per esplorare il file Excel e trovare i dati veri
"""
import pandas as pd

print("="*60)
print("ESPLORAZIONE FILE EXCEL")
print("="*60)

file_path = r'C:\Users\moonw\Desktop\Scrivania\Anno 3\Principi e modelli della percezione\mine-detection-project\data\raw\Mine_Dataset.xls'

# Leggi tutti i nomi dei fogli
excel_file = pd.ExcelFile(file_path)
sheet_names = excel_file.sheet_names

print(f"\n📋 Fogli trovati nel file Excel: {len(sheet_names)}")
print("-" * 60)

for i, sheet_name in enumerate(sheet_names):
    print(f"\n{i+1}. FOGLIO: '{sheet_name}'")
    print("-" * 60)
    
    # Leggi il foglio
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    print(f"   Dimensioni: {df.shape[0]} righe x {df.shape[1]} colonne")
    print(f"   Colonne: {df.columns.tolist()}")
    
    print(f"\n   Prime 3 righe:")
    print(df.head(3).to_string())
    
    print("\n" + "="*60)

# Suggerimento
print("\n💡 SUGGERIMENTO:")
print("Cerca il foglio che ha:")
print("  - Circa 338 righe")
print("  - Colonne tipo: V, H, S, M (o simili)")
print("  - Dati numerici nelle prime righe")