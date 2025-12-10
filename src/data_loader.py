"""
Data Loader per Mine Detection Dataset
VERSIONE SEMPLIFICATA
"""
import pandas as pd
import numpy as np


def load_mine_data(filepath='data/raw/Mine_Dataset.xls'):
    """
    Carica il dataset delle mine
    
    Args:
        filepath: percorso al file Excel
        
    Returns:
        DataFrame con i dati
    """
    # Carica file Excel (foglio 'Normalized_Data')
    df = pd.read_excel(filepath, sheet_name='Normalized_Data')
    
    print(f"✓ Caricati {len(df)} campioni")
    print(f"✓ Colonne: {list(df.columns)}")
    
    return df


def print_dataset_info(df):
    """Stampa informazioni sul dataset"""
    print("\n" + "="*60)
    print("📊 INFORMAZIONI DATASET")
    print("="*60)
    print(f"Numero totale campioni: {len(df)}")
    print(f"Numero features: {len(df.columns)}")
    print(f"\nPrime 5 righe:")
    print(df.head())
    print(f"\nStatistiche:")
    print(df.describe())
    print(f"\nValori mancanti:")
    print(df.isnull().sum())
    print("="*60)


def split_features_target(df, target_column='M'):
    """
    Separa features e target
    
    Args:
        df: DataFrame completo
        target_column: nome colonna target (default 'M')
        
    Returns:
        X (features), y (target)
    """
    # Trova colonna target
    if target_column not in df.columns:
        print(f"⚠️  Colonna '{target_column}' non trovata!")
        print(f"Colonne disponibili: {list(df.columns)}")
        return None, None
    
    # Separa features e target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    print(f"✓ Features: {list(X.columns)}")
    print(f"✓ Target: {target_column}")
    print(f"✓ Shape X: {X.shape}, Shape y: {y.shape}")
    
    return X, y


if __name__ == "__main__":
    # Test del caricamento
    print("Test caricamento dati...")
    
    # Carica dati
    df = load_mine_data('data/raw/Mine_Dataset.xls')
    
    # Mostra info
    print_dataset_info(df)
    
    # Separa features e target
    X, y = split_features_target(df)