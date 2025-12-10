"""
TEST COMPLETO - Mine Detection Project
Carica i dati e genera tutti i grafici
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Importa moduli personalizzati
sys.path.append('src')
from visualization import *

print("="*70)
print("🚀 MINE DETECTION PROJECT - TEST COMPLETO")
print("="*70)

# ========================================
# 1. CARICAMENTO DATI
# ========================================
print("\n📊 STEP 1: Caricamento dati...")
print("-"*70)

try:
    # Carica dal foglio corretto!
    df = pd.read_excel('data/raw/Mine_Dataset.xls', sheet_name='Normalized_Data')
    print(f"✓ Dati caricati con successo!")
    print(f"✓ Dimensioni: {df.shape[0]} righe × {df.shape[1]} colonne")
    print(f"✓ Colonne: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ ERRORE nel caricamento: {e}")
    sys.exit(1)

# ========================================
# 2. ESPLORAZIONE DATI
# ========================================
print("\n📊 STEP 2: Esplorazione dati...")
print("-"*70)

print("\n📋 Prime 10 righe:")
print(df.head(10))

print("\n📊 Statistiche descrittive:")
print(df.describe())

print("\n🔍 Informazioni sui dati:")
print(df.info())

print("\n❓ Valori mancanti:")
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("✓ Nessun valore mancante!")

print("\n📈 Distribuzione classi (Mine Type):")
class_dist = df['M'].value_counts().sort_index()
print(class_dist)
print("\nPercentuali:")
print((class_dist / len(df) * 100).round(2))

# ========================================
# 3. CREAZIONE CARTELLE
# ========================================
print("\n📊 STEP 3: Preparazione cartelle...")
print("-"*70)

os.makedirs('results/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
print("✓ Cartelle create/verificate")

# Salva anche una copia CSV
df.to_csv('data/processed/mine_data_clean.csv', index=False)
print("✓ Dati salvati in: data/processed/mine_data_clean.csv")

# ========================================
# 4. GENERAZIONE GRAFICI
# ========================================
print("\n📊 STEP 4: Generazione grafici...")
print("-"*70)

try:
    # Grafico 1: Panoramica generale
    print("\n📈 1/6 - Panoramica dataset...")
    plot_data_overview(df, save_path='results/figures/01_data_overview.png')
    print("    ✓ Salvato: 01_data_overview.png")
    
    # Grafico 2: Voltaggio per tipo mina
    print("\n📈 2/6 - Voltaggio per tipo mina...")
    plot_voltage_by_mine_type(df, save_path='results/figures/02_voltage_by_mine.png')
    print("    ✓ Salvato: 02_voltage_by_mine.png")
    
    # Grafico 3: Correlazione
    print("\n📈 3/6 - Matrice correlazione...")
    plot_correlation_matrix(df, save_path='results/figures/03_correlation_matrix.png')
    print("    ✓ Salvato: 03_correlation_matrix.png")
    
    # Grafico 4: Scatter plot V vs H colorato per Mine Type
    print("\n📈 4/6 - Scatter plot V vs H...")
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df['V'], df['H'], c=df['M'], cmap='viridis', 
                        alpha=0.6, edgecolors='black', s=50)
    ax.set_xlabel('Voltage (V)', fontsize=12)
    ax.set_ylabel('Height (H)', fontsize=12)
    ax.set_title('Voltage vs Height (colored by Mine Type)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Mine Type', ax=ax)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/04_scatter_V_H.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Salvato: 04_scatter_V_H.png")
    
    # Grafico 5: Distribuzione per Soil Type
    print("\n📈 5/6 - Distribuzione per tipo suolo...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Conteggio per soil type
    df['S'].value_counts().sort_index().plot(kind='bar', ax=axes[0], 
                                              edgecolor='black', color='lightblue')
    axes[0].set_title('Samples per Soil Type', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Soil Type')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Mine type distribution per soil type
    pd.crosstab(df['S'], df['M']).plot(kind='bar', stacked=True, ax=axes[1], 
                                        edgecolor='black')
    axes[1].set_title('Mine Type Distribution per Soil Type', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Soil Type')
    axes[1].set_ylabel('Count')
    axes[1].legend(title='Mine Type', bbox_to_anchor=(1.05, 1))
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/figures/05_soil_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Salvato: 05_soil_analysis.png")
    
    # Grafico 6: Pairplot (relazioni tra tutte le variabili)
    print("\n📈 6/6 - Pairplot (può richiedere qualche secondo)...")
    import seaborn as sns
    pairplot = sns.pairplot(df, hue='M', palette='viridis', 
                            plot_kws={'alpha': 0.6, 'edgecolor': 'black', 's': 30},
                            diag_kind='hist')
    pairplot.fig.suptitle('Pairplot - All Features', y=1.02, fontsize=14, fontweight='bold')
    plt.savefig('results/figures/06_pairplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    ✓ Salvato: 06_pairplot.png")
    
    print("\n" + "="*70)
    print("✅ TUTTI I GRAFICI GENERATI CON SUCCESSO!")
    print("="*70)
    
except Exception as e:
    print(f"\n❌ Errore nella generazione grafici: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# 5. RIEPILOGO FINALE
# ========================================
print("\n📁 RIEPILOGO FILES GENERATI:")
print("-"*70)
print("\n📊 Grafici (in results/figures/):")
print("  1. 01_data_overview.png - Panoramica generale")
print("  2. 02_voltage_by_mine.png - Voltaggio per tipo mina")
print("  3. 03_correlation_matrix.png - Correlazioni")
print("  4. 04_scatter_V_H.png - Scatter Voltage vs Height")
print("  5. 05_soil_analysis.png - Analisi tipo suolo")
print("  6. 06_pairplot.png - Relazioni tra tutte le variabili")

print("\n💾 Dati processati:")
print("  - data/processed/mine_data_clean.csv")

print("\n" + "="*70)
print("✅ TEST COMPLETATO CON SUCCESSO!")
print("="*70)
print("\n🎯 PROSSIMI PASSI:")
print("  1. Controlla i grafici in results/figures/")
print("  2. Inizia con la classificazione (Random Forest)")
print("  3. Poi Neural Networks")
print("  4. Infine Clustering")
print("\n💡 Usa i notebook in notebooks/ per l'analisi interattiva!")
print("="*70)