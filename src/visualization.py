"""
Visualizzazione dati per Mine Detection
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA


# Imposta stile globale e caratteri italiani
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'


def plot_data_overview(df, save_path=None):
    """Panoramica generale del dataset"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    columns = df.columns.tolist()
    
    # 1. Distribuzione per tipo di mina
    if 'M' in columns:
        df['M'].value_counts().sort_index().plot(
            kind='bar', ax=axes[0, 0], 
            color=['red', 'blue', 'green', 'orange', 'purple'],
            edgecolor='black'
        )
        axes[0, 0].set_title('Distribuzione Tipi di Mine', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Tipo Mina')
        axes[0, 0].set_ylabel('Numero Campioni')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribuzione Voltaggio
    if 'V' in columns:
        axes[0, 1].hist(df['V'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].set_title('Distribuzione Voltaggio (V)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Voltaggio')
        axes[0, 1].set_ylabel('Frequenza')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribuzione Altezza
    if 'H' in columns:
        axes[1, 0].hist(df['H'], bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[1, 0].set_title('Distribuzione Altezza (H)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Altezza (cm)')
        axes[1, 0].set_ylabel('Frequenza')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Distribuzione Tipo Suolo
    if 'S' in columns:
        df['S'].value_counts().plot(
            kind='bar', ax=axes[1, 1], 
            edgecolor='black', alpha=0.7, color='lightgreen'
        )
        axes[1, 1].set_title('Distribuzione Tipo Suolo (S)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Tipo Suolo')
        axes[1, 1].set_ylabel('Frequenza')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Grafico salvato: {save_path}")
    
    plt.close()
    return fig


def plot_voltage_by_mine_type(df, save_path=None):
    """Boxplot: Voltaggio per tipo di mina"""
    if 'V' not in df.columns or 'M' not in df.columns:
        print("⚠️ Colonne V o M non trovate!")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    mine_types = sorted(df['M'].unique())
    data_by_type = [df[df['M'] == m]['V'].values for m in mine_types]
    
    bp = ax.boxplot(data_by_type, 
                    labels=mine_types,
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', 
                                  markersize=8, markeredgecolor='darkred'),
                    medianprops=dict(color='darkblue', linewidth=2.5),
                    boxprops=dict(linewidth=1.5, edgecolor='black'),
                    whiskerprops=dict(linewidth=1.5, color='black'),
                    capprops=dict(linewidth=1.5, color='black'),
                    flierprops=dict(marker='o', markerfacecolor='red', 
                                   markersize=6, alpha=0.6, markeredgecolor='darkred'))
    
    for patch, color in zip(bp['boxes'], colors[:len(mine_types)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for i, mine_type in enumerate(mine_types, 1):
        mine_data = df[df['M'] == mine_type]['V']
        n = len(mine_data)
        mean_val = mine_data.mean()
        
        stats_text = f'n={n}\nμ={mean_val:.3f}'
        ax.text(i, mine_data.max() + 0.05, stats_text, 
               ha='center', va='bottom', fontsize=9, 
               fontweight='bold', color='darkblue',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='gray', alpha=0.8))
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    ax.set_axisbelow(True)
    
    ax.set_title('Distribuzione Voltaggio per Tipo Mina', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Tipo Mina', fontsize=13, fontweight='bold')
    ax.set_ylabel('Voltaggio (V)', fontsize=13, fontweight='bold')
    
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='w', label='Media',
                  markerfacecolor='red', markersize=8, markeredgecolor='darkred'),
        plt.Line2D([0], [0], color='darkblue', linewidth=2.5, label='Mediana'),
        plt.Line2D([0], [0], marker='o', color='w', label='Outlier',
                  markerfacecolor='red', markersize=6, alpha=0.6, markeredgecolor='darkred')
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             frameon=True, shadow=True, fontsize=10)
    
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print(f"✓ Grafico salvato: {save_path}")
    
    plt.close()
    return fig


def plot_correlation_matrix(df, save_path=None):
    """Matrice di correlazione tra features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title('Matrice di Correlazione tra Features', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Grafico salvato: {save_path}")
    
    plt.close()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None, title='Matrice di Confusione', save_path=None):
    """Matrice di Confusione con annotazioni"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                cbar_kws={"shrink": 0.8}, linewidths=1, linecolor='gray')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Etichetta Vera', fontsize=12)
    ax.set_xlabel('Etichetta Predetta', fontsize=12)
    
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Grafico salvato: {save_path}")
    
    plt.close()
    return fig


def plot_feature_importance(importance_df, title='Importanza Features', save_path=None):
    """Grafico importanza features"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    importance_df.plot(x='Feature', y='Importance', kind='barh', ax=ax, 
                       legend=False, color='steelblue', edgecolor='black')
    ax.set_xlabel('Importanza', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Grafico salvato: {save_path}")
    
    plt.close()
    return fig


def plot_model_comparison(comparison_df, save_path=None):
    """Confronto tra modelli"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_names_ita = ['Accuratezza', 'Precisione', 'Richiamo', 'F1-Score']
    x = np.arange(len(comparison_df))
    width = 0.2
    
    for i, (metric, metric_ita) in enumerate(zip(metrics, metric_names_ita)):
        if metric in comparison_df.columns:
            ax.bar(x + i*width, comparison_df[metric], width, 
                   label=metric_ita, edgecolor='black')
    
    ax.set_xlabel('Modello', fontsize=12)
    ax.set_ylabel('Punteggio', fontsize=12)
    ax.set_title('Confronto Prestazioni Modelli', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df['Model'])
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Grafico salvato: {save_path}")
    
    plt.close()
    return fig


def plot_training_history(history, title='Storia Addestramento', save_path=None):
    """Grafico training/validation per Neural Network"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validazione', 
                linewidth=2, linestyle='--')
    axes[0].set_xlabel('Epoca', fontsize=12)
    axes[0].set_ylabel('Accuratezza', fontsize=12)
    axes[0].set_title('Accuratezza Modello', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validazione', 
                linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoca', fontsize=12)
    axes[1].set_ylabel('Perdita', fontsize=12)
    axes[1].set_title('Perdita Modello', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Grafico salvato: {save_path}")
    
    plt.close()
    return fig


def plot_clusters_2d(X, labels, title='Visualizzazione Cluster', save_path=None):
    """Visualizzazione clustering in 2D con PCA"""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, 
                        cmap='viridis', alpha=0.6, 
                        edgecolors='black', linewidth=0.5, s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, label='Cluster', ax=ax)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Grafico salvato: {save_path}")
    
    plt.close()
    return fig


def plot_elbow_curve(k_range, inertias, save_path=None):
    """Curva Elbow per K-Means"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Numero di Cluster (K)', fontsize=12)
    ax.set_ylabel('Inerzia', fontsize=12)
    ax.set_title('Metodo Elbow per K Ottimale', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Grafico salvato: {save_path}")
    
    plt.close()
    return fig


if __name__ == "__main__":
    print("Modulo visualization_ita.py caricato correttamente! ✓")
    print("\nFunzioni disponibili (ITALIANO):")
    print("- plot_data_overview()")
    print("- plot_voltage_by_mine_type()")
    print("- plot_correlation_matrix()")
    print("- plot_confusion_matrix()")
    print("- plot_feature_importance()")
    print("- plot_model_comparison()")
    print("- plot_training_history()")
    print("- plot_clusters_2d()")
    print("- plot_elbow_curve()")