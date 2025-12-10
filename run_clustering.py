"""
CLUSTERING - Mine Detection
K-Means, DBSCAN, Hierarchical
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, 
    calinski_harabasz_score, adjusted_rand_score
)
from scipy.cluster.hierarchy import dendrogram, linkage

print("="*70)
print("🔍 CLUSTERING ANALYSIS - MINE DETECTION")
print("="*70)

# ========================================
# 1. CARICAMENTO E PREPROCESSING
# ========================================
print("\n📊 STEP 1: Caricamento dati...")
print("-"*70)

df = pd.read_excel('data/raw/Mine_Dataset.xls', sheet_name='Normalized_Data')
print(f"✓ Dati caricati: {df.shape}")

X = df[['V', 'H', 'S']].values
y_true = df['M'].values

# Standardizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Features standardizzate")

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

# ========================================
# 2. OPTIMAL K (ELBOW METHOD)
# ========================================
print("\n" + "="*70)
print("🎯 STEP 2: TROVA K OTTIMALE")
print("="*70)

print("\n🔍 Testing K da 2 a 10...")
k_range = range(2, 11)
inertias = []
silhouettes = []
davies_bouldins = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    davies_bouldins.append(davies_bouldin_score(X_scaled, labels))
    
    print(f"  K={k}: Silhouette={silhouettes[-1]:.4f}, DB={davies_bouldins[-1]:.4f}")

# Plot elbow analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Elbow
axes[0].plot(k_range, inertias, marker='o', linewidth=2.5, markersize=8, color='#4ECDC4')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Inertia', fontsize=12, fontweight='bold')
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_facecolor('#F8F9FA')

# Silhouette
axes[1].plot(k_range, silhouettes, marker='s', linewidth=2.5, markersize=8, color='#45B7D1')
axes[1].axhline(y=max(silhouettes), color='red', linestyle='--', alpha=0.5, label='Best')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
axes[1].set_title('Silhouette Analysis (higher is better)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_facecolor('#F8F9FA')

# Davies-Bouldin
axes[2].plot(k_range, davies_bouldins, marker='^', linewidth=2.5, markersize=8, color='#96CEB4')
axes[2].axhline(y=min(davies_bouldins), color='red', linestyle='--', alpha=0.5, label='Best')
axes[2].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
axes[2].set_title('Davies-Bouldin Index (lower is better)', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_facecolor('#F8F9FA')

plt.suptitle('Optimal K Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/16_optimal_k_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✓ Salvato: 16_optimal_k_analysis.png")

# Trova K ottimale
best_k_silhouette = k_range[np.argmax(silhouettes)]
print(f"\n🏆 K ottimale (Silhouette): {best_k_silhouette}")

# ========================================
# 3. K-MEANS CLUSTERING
# ========================================
print("\n" + "="*70)
print("🎯 STEP 3: K-MEANS CLUSTERING")
print("="*70)

# Usa K=5 (numero di classi reali)
k_optimal = 5
print(f"\n🌟 Usando K={k_optimal} (matching numero classi reali)")

kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Metriche
sil_kmeans = silhouette_score(X_scaled, labels_kmeans)
db_kmeans = davies_bouldin_score(X_scaled, labels_kmeans)
ch_kmeans = calinski_harabasz_score(X_scaled, labels_kmeans)
ari_kmeans = adjusted_rand_score(y_true, labels_kmeans)

print(f"\n📊 K-Means Performance:")
print(f"  Silhouette Score: {sil_kmeans:.4f}")
print(f"  Davies-Bouldin:   {db_kmeans:.4f}")
print(f"  Calinski-Harabasz: {ch_kmeans:.2f}")
print(f"  ARI (vs true labels): {ari_kmeans:.4f}")

# Visualizzazione 2D con PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Clusters trovati
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, 
                          cmap='viridis', alpha=0.6, edgecolors='black', 
                          linewidth=0.5, s=50)
axes[0].scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
               pca.transform(kmeans.cluster_centers_)[:, 1],
               c='red', marker='X', s=300, edgecolors='black', 
               linewidth=2, label='Centroids')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                  fontsize=12, fontweight='bold')
axes[0].set_title(f'K-Means Clusters (K={k_optimal})', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_facecolor('#F8F9FA')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Plot 2: True labels
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, 
                          cmap='viridis', alpha=0.6, edgecolors='black', 
                          linewidth=0.5, s=50)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                  fontsize=12, fontweight='bold')
axes[1].set_title('True Mine Types', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_facecolor('#F8F9FA')
plt.colorbar(scatter2, ax=axes[1], label='Mine Type')

plt.suptitle('K-Means Clustering vs True Labels', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/17_kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Salvato: 17_kmeans_clusters.png")

# ========================================
# 4. DBSCAN CLUSTERING
# ========================================
print("\n" + "="*70)
print("🎯 STEP 4: DBSCAN CLUSTERING")
print("="*70)

print("\n🔍 Testing DBSCAN con diversi epsilon...")
best_eps = None
best_sil = -1
best_labels_dbscan = None

for eps in [0.3, 0.4, 0.5, 0.6, 0.7]:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if n_clusters > 1 and n_noise < len(labels):
        mask = labels != -1
        if np.sum(mask) > 0:
            sil = silhouette_score(X_scaled[mask], labels[mask])
            print(f"  eps={eps}: {n_clusters} clusters, {n_noise} noise, Silhouette={sil:.4f}")
            
            if sil > best_sil:
                best_sil = sil
                best_eps = eps
                best_labels_dbscan = labels

if best_eps is not None:
    print(f"\n🏆 Best eps: {best_eps}")
    
    # Visualizza
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separa noise points
    mask_noise = best_labels_dbscan == -1
    mask_clusters = best_labels_dbscan != -1
    
    # Plot clusters
    if np.sum(mask_clusters) > 0:
        scatter = ax.scatter(X_pca[mask_clusters, 0], X_pca[mask_clusters, 1], 
                           c=best_labels_dbscan[mask_clusters], cmap='viridis',
                           alpha=0.6, edgecolors='black', linewidth=0.5, s=50)
        plt.colorbar(scatter, ax=ax, label='Cluster')
    
    # Plot noise
    if np.sum(mask_noise) > 0:
        ax.scatter(X_pca[mask_noise, 0], X_pca[mask_noise, 1],
                  c='red', marker='x', s=50, alpha=0.8, label='Noise')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
                 fontsize=12, fontweight='bold')
    ax.set_title(f'DBSCAN Clustering (eps={best_eps})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    plt.savefig('results/figures/18_dbscan_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Salvato: 18_dbscan_clusters.png")
else:
    print("⚠️ DBSCAN non ha trovato cluster validi")
    best_labels_dbscan = np.full(len(X_scaled), -1)

# ========================================
# 5. HIERARCHICAL CLUSTERING
# ========================================
print("\n" + "="*70)
print("🎯 STEP 5: HIERARCHICAL CLUSTERING")
print("="*70)

# Hierarchical con K=5
hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward')
labels_hier = hierarchical.fit_predict(X_scaled)

sil_hier = silhouette_score(X_scaled, labels_hier)
db_hier = davies_bouldin_score(X_scaled, labels_hier)
ari_hier = adjusted_rand_score(y_true, labels_hier)

print(f"\n📊 Hierarchical Performance:")
print(f"  Silhouette Score: {sil_hier:.4f}")
print(f"  Davies-Bouldin:   {db_hier:.4f}")
print(f"  ARI (vs true labels): {ari_hier:.4f}")

# Visualizza
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_hier, 
                    cmap='viridis', alpha=0.6, edgecolors='black', 
                    linewidth=0.5, s=50)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
             fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
             fontsize=12, fontweight='bold')
ax.set_title('Hierarchical Clustering (Ward linkage)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F8F9FA')
plt.colorbar(scatter, ax=ax, label='Cluster')

plt.tight_layout()
plt.savefig('results/figures/19_hierarchical_clusters.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Salvato: 19_hierarchical_clusters.png")

# Dendrogram (sample ridotto per leggibilità)
print("\n📊 Creando dendrogram...")
linkage_matrix = linkage(X_scaled[:100], method='ward')

fig, ax = plt.subplots(figsize=(14, 7))
dendrogram(linkage_matrix, ax=ax, color_threshold=0)
ax.set_title('Hierarchical Clustering Dendrogram (100 samples)', 
            fontsize=14, fontweight='bold')
ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures/20_dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Salvato: 20_dendrogram.png")

# ========================================
# 6. CONFRONTO FINALE
# ========================================
print("\n" + "="*70)
print("📊 STEP 6: CONFRONTO ALGORITMI")
print("="*70)

# Prepara comparazione
comparison_clustering = pd.DataFrame({
    'Algorithm': ['K-Means', 'DBSCAN', 'Hierarchical'],
    'Silhouette Score': [sil_kmeans, best_sil if best_eps else 0, sil_hier],
    'Davies-Bouldin': [db_kmeans, 0, db_hier],
    'ARI (vs true)': [ari_kmeans, 0 if not best_eps else adjusted_rand_score(y_true, best_labels_dbscan), ari_hier]
})

print("\n" + comparison_clustering.to_string(index=False))

# Salva
comparison_clustering.to_csv('results/metrics/clustering_comparison.csv', index=False)
print("\n✓ Tabella salvata: results/metrics/clustering_comparison.csv")

# Grafico comparativo
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Silhouette
x_pos = np.arange(len(comparison_clustering))
bars = axes[0].bar(x_pos, comparison_clustering['Silhouette Score'], 
                   color=['#4ECDC4', '#45B7D1', '#96CEB4'],
                   edgecolor='black', linewidth=2, alpha=0.8)

for bar, val in zip(bars, comparison_clustering['Silhouette Score']):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[0].set_xlabel('Algorithm', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Silhouette Score', fontsize=13, fontweight='bold')
axes[0].set_title('Silhouette Score Comparison (higher is better)', 
                 fontsize=14, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(comparison_clustering['Algorithm'])
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_facecolor('#F8F9FA')

# ARI
bars = axes[1].bar(x_pos, comparison_clustering['ARI (vs true)'],
                   color=['#FFA07A', '#FF8C94', '#FFB6C1'],
                   edgecolor='black', linewidth=2, alpha=0.8)

for bar, val in zip(bars, comparison_clustering['ARI (vs true)']):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[1].set_xlabel('Algorithm', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Adjusted Rand Index', fontsize=13, fontweight='bold')
axes[1].set_title('ARI vs True Labels (1.0 = perfect match)', 
                 fontsize=14, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(comparison_clustering['Algorithm'])
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_facecolor('#F8F9FA')

plt.suptitle('Clustering Algorithms Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/21_clustering_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Salvato: 21_clustering_comparison.png")

# ========================================
# RIEPILOGO FINALE
# ========================================
print("\n" + "="*70)
print("✅ CLUSTERING COMPLETATO!")
print("="*70)

print("\n📊 RISULTATI:")
print(f"  K-Means:      Silhouette={sil_kmeans:.4f}, ARI={ari_kmeans:.4f}")
dbscan_result = f"{best_sil:.4f}" if best_eps else "N/A"
print(f"  DBSCAN:       Silhouette={dbscan_result}")
print(f"  Hierarchical: Silhouette={sil_hier:.4f}, ARI={ari_hier:.4f}")

best_algo = comparison_clustering.loc[comparison_clustering['Silhouette Score'].idxmax(), 'Algorithm']
print(f"\n🏆 Miglior algoritmo (Silhouette): {best_algo}")

print("\n📁 FILES GENERATI:")
print("  16. optimal_k_analysis.png - Analisi K ottimale")
print("  17. kmeans_clusters.png - K-Means vs True Labels")
print("  18. dbscan_clusters.png - DBSCAN")
print("  19. hierarchical_clusters.png - Hierarchical")
print("  20. dendrogram.png - Dendrogramma")
print("  21. clustering_comparison.png - Confronto finale")

print("\n🎯 PROGETTO COMPLETATO!")
print("  ✅ Random Forest")
print("  ✅ Neural Networks")
print("  ✅ Clustering")
print("\nOra puoi creare la relazione finale e preparare la presentazione!")

print("\n" + "="*70)