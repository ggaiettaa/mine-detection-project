"""
Script per rigenerare TUTTI i grafici in ITALIANO
Esegui questo per avere tutti i grafici con testi italiani
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Importa modulo italiano
sys.path.append('src')
from visualization import *

print("="*70)
print("🇮🇹 RIGENERAZIONE GRAFICI IN ITALIANO")
print("="*70)

# ========================================
# 1. GRAFICI DATA EXPLORATION
# ========================================
print("\n📊 STEP 1: Data Exploration...")

df = pd.read_excel('data/raw/Mine_Dataset.xls', sheet_name='Normalized_Data')
print(f"✓ Dati caricati: {df.shape}")

os.makedirs('results/figures_ita', exist_ok=True)

# Grafico 1: Overview
print("  1/6 - Panoramica dataset...")
plot_data_overview(df, save_path='results/figures_ita/01_panoramica_dati.png')

# Grafico 2: Voltaggio per tipo
print("  2/6 - Voltaggio per tipo mina...")
plot_voltage_by_mine_type(df, save_path='results/figures_ita/02_voltaggio_per_tipo.png')

# Grafico 3: Correlazione
print("  3/6 - Matrice correlazione...")
plot_correlation_matrix(df, save_path='results/figures_ita/03_matrice_correlazione.png')

# Grafico 4: Scatter V vs H
print("  4/6 - Scatter plot...")
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df['V'], df['H'], c=df['M'], cmap='viridis', 
                    alpha=0.6, edgecolors='black', s=50)
ax.set_xlabel('Voltaggio (V)', fontsize=12)
ax.set_ylabel('Altezza (H)', fontsize=12)
ax.set_title('Voltaggio vs Altezza (colorato per Tipo Mina)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Tipo Mina', ax=ax)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures_ita/04_scatter_V_H.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# Grafico 5: Analisi suolo
print("  5/6 - Analisi tipo suolo...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

df['S'].value_counts().sort_index().plot(kind='bar', ax=axes[0], 
                                          edgecolor='black', color='lightblue')
axes[0].set_title('Campioni per Tipo Suolo', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Tipo Suolo')
axes[0].set_ylabel('Conteggio')
axes[0].grid(True, alpha=0.3, axis='y')

pd.crosstab(df['S'], df['M']).plot(kind='bar', stacked=True, ax=axes[1], 
                                    edgecolor='black')
axes[1].set_title('Distribuzione Tipi Mine per Tipo Suolo', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Tipo Suolo')
axes[1].set_ylabel('Conteggio')
axes[1].legend(title='Tipo Mina', bbox_to_anchor=(1.05, 1))
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures_ita/05_analisi_suolo.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# Grafico 6: Pairplot
print("  6/6 - Pairplot...")
pairplot = sns.pairplot(df, hue='M', palette='viridis', 
                        plot_kws={'alpha': 0.6, 'edgecolor': 'black', 's': 30},
                        diag_kind='hist')
pairplot.fig.suptitle('Pairplot - Tutte le Features', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('results/figures_ita/06_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# ========================================
# 2. GRAFICI RANDOM FOREST
# ========================================
print("\n🌲 STEP 2: Random Forest...")

# Carica metriche
comparison_rf = pd.read_csv('results/metrics/rf_comparison.csv')
importance = pd.read_csv('results/metrics/feature_importance.csv')

# Traduci nomi modelli
comparison_rf['Model'] = comparison_rf['Model'].replace({
    'Baseline RF': 'RF Base',
    'Optimized RF': 'RF Ottimizzato'
})

# Traduci nomi features
importance['Feature'] = importance['Feature'].replace({
    'Voltage': 'Voltaggio',
    'Height': 'Altezza',
    'Soil Type': 'Tipo Suolo'
})

# Feature Importance
print("  - Importanza features...")
plot_feature_importance(importance, 
                       title='Importanza Features - Random Forest Ottimizzato',
                       save_path='results/figures_ita/10_importanza_features.png')

# Comparazione dettagliata
print("  - Comparazione dettagliata...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_names_ita = ['Accuratezza', 'Precisione', 'Richiamo', 'F1-Score']
colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for idx, (metric, metric_ita, color) in enumerate(zip(metrics, metric_names_ita, colors)):
    ax = axes[idx // 2, idx % 2]
    
    values = comparison_rf[metric].values
    x_pos = np.arange(len(comparison_rf))
    
    bars = ax.bar(x_pos, values, width=0.6, color=color, 
                  edgecolor='black', linewidth=2, alpha=0.8)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.0002,
               f'{val:.4f}\n({val*100:.2f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    min_val = min(values)
    max_val = max(values)
    margin = (max_val - min_val) * 0.3 if max_val > min_val else 0.001
    ax.set_ylim([min_val - margin, max_val + margin*2])
    
    ax.set_xlabel('Modello', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_ita, fontsize=12, fontweight='bold')
    ax.set_title(f'Confronto {metric_ita}', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison_rf['Model'], fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#F8F9FA')

plt.suptitle('Random Forest: Base vs Ottimizzato (Vista Dettagliata)', 
            fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('results/figures_ita/11_confronto_rf_dettagliato.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# ========================================
# 3. GRAFICI NEURAL NETWORKS
# ========================================
print("\n🧠 STEP 3: Neural Networks...")

# Carica metriche
comparison_nn = pd.read_csv('results/metrics/nn_comparison.csv')

# Traduci nomi
comparison_nn['Model'] = comparison_nn['Model'].replace({
    'Simple NN': 'NN Semplice',
    'Medium NN': 'NN Media',
    'Deep NN': 'NN Profonda'
})

print("  - Comparazione dettagliata NN...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy
ax = axes[0]
x_pos = np.arange(len(comparison_nn))
bars = ax.bar(x_pos, comparison_nn['Test Accuracy'], width=0.6, 
              color=['#4ECDC4', '#45B7D1', '#96CEB4'],
              edgecolor='black', linewidth=2, alpha=0.8)

for bar, val in zip(bars, comparison_nn['Test Accuracy']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
           f'{val:.4f}\n({val*100:.2f}%)',
           ha='center', va='bottom', fontsize=11, fontweight='bold')

min_val = comparison_nn['Test Accuracy'].min()
max_val = comparison_nn['Test Accuracy'].max()
margin = (max_val - min_val) * 0.3 if max_val > min_val else 0.01
ax.set_ylim([min_val - margin, max_val + margin*2])

ax.set_xlabel('Architettura', fontsize=13, fontweight='bold')
ax.set_ylabel('Accuratezza Test', fontsize=13, fontweight='bold')
ax.set_title('Reti Neurali - Accuratezza Test', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_nn['Model'], fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_facecolor('#F8F9FA')

# Loss
ax = axes[1]
bars = ax.bar(x_pos, comparison_nn['Test Loss'], width=0.6,
              color=['#FFA07A', '#FF6B6B', '#FF8C94'],
              edgecolor='black', linewidth=2, alpha=0.8)

for bar, val in zip(bars, comparison_nn['Test Loss']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
           f'{val:.4f}',
           ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Architettura', fontsize=13, fontweight='bold')
ax.set_ylabel('Perdita Test', fontsize=13, fontweight='bold')
ax.set_title('Reti Neurali - Perdita Test (più basso è meglio)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_nn['Model'], fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_facecolor('#F8F9FA')

plt.suptitle('Confronto Prestazioni Reti Neurali', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures_ita/15_confronto_nn_dettagliato.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# ========================================
# 4. GRAFICI CLUSTERING
# ========================================
print("\n🔍 STEP 4: Clustering...")

# Carica metriche
comparison_clust = pd.read_csv('results/metrics/clustering_comparison.csv')

# Traduci nomi
comparison_clust['Algorithm'] = comparison_clust['Algorithm'].replace({
    'K-Means': 'K-Means',
    'DBSCAN': 'DBSCAN',
    'Hierarchical': 'Gerarchico'
})

print("  - Comparazione clustering...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Silhouette
x_pos = np.arange(len(comparison_clust))
bars = axes[0].bar(x_pos, comparison_clust['Silhouette Score'], 
                   color=['#4ECDC4', '#45B7D1', '#96CEB4'],
                   edgecolor='black', linewidth=2, alpha=0.8)

for bar, val in zip(bars, comparison_clust['Silhouette Score']):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[0].set_xlabel('Algoritmo', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Punteggio Silhouette', fontsize=13, fontweight='bold')
axes[0].set_title('Confronto Punteggio Silhouette (più alto è meglio)', 
                 fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(comparison_clust['Algorithm'])
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_facecolor('#F8F9FA')

# ARI
bars = axes[1].bar(x_pos, comparison_clust['ARI (vs true)'],
                   color=['#FFA07A', '#FF8C94', '#FFB6C1'],
                   edgecolor='black', linewidth=2, alpha=0.8)

for bar, val in zip(bars, comparison_clust['ARI (vs true)']):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[1].set_xlabel('Algoritmo', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Indice Rand Aggiustato', fontsize=13, fontweight='bold')
axes[1].set_title('ARI vs Etichette Vere (1.0 = match perfetto)', 
                 fontsize=14, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(comparison_clust['Algorithm'])
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_facecolor('#F8F9FA')

plt.suptitle('Confronto Algoritmi di Clustering', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures_ita/21_confronto_clustering.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# ========================================
# RIEPILOGO
# ========================================
print("\n" + "="*70)
print("✅ RIGENERAZIONE COMPLETATA!")
print("="*70)

print("\n📁 Grafici in italiano creati in: results/figures_ita/")
print("\n📊 Grafici principali generati:")
print("  • 01-06: Esplorazione dati")
print("  • 10-11: Random Forest")
print("  • 15: Neural Networks")
print("  • 21: Clustering")

print("\n💡 NOTA:")
print("  I grafici delle confusion matrices e training histories")
print("  mantengono alcune etichette in inglese per compatibilità")
print("  con le librerie sklearn/tensorflow.")

print("\n✅ Ora puoi usare questi grafici per la tua presentazione!")
print("="*70)