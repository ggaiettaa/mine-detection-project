"""
RANDOM FOREST - Classificazione Mine Detection
Script completo: Baseline → Tuning → Comparison
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import sys
import os

import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo
import matplotlib.pyplot as plt

# Importa moduli custom
sys.path.append('src')
from visualization import plot_confusion_matrix, plot_feature_importance

print("="*70)
print("🌲 RANDOM FOREST - MINE DETECTION")
print("="*70)

# ========================================
# 1. CARICAMENTO DATI
# ========================================
print("\n📊 STEP 1: Caricamento dati...")
print("-"*70)

df = pd.read_excel('data/raw/Mine_Dataset.xls', sheet_name='Normalized_Data')
print(f"✓ Dati caricati: {df.shape}")

# Separa features e target
X = df[['V', 'H', 'S']]
y = df['M']

print(f"✓ Features (X): {X.shape}")
print(f"✓ Target (y): {y.shape}")
print(f"\nDistribuzione classi:")
print(y.value_counts().sort_index())

# ========================================
# 2. SPLIT TRAIN/TEST
# ========================================
print("\n📊 STEP 2: Split train/test...")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"\nDistribuzione train:")
print(y_train.value_counts().sort_index())
print(f"\nDistribuzione test:")
print(y_test.value_counts().sort_index())

# Crea cartelle
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

# ========================================
# 3. BASELINE RANDOM FOREST
# ========================================
print("\n" + "="*70)
print("🎯 STEP 3: BASELINE RANDOM FOREST")
print("="*70)

print("\n🌲 Training baseline model (default parameters)...")
rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
print("✓ Training completato!")

# Predizioni
y_pred_baseline = rf_baseline.predict(X_test)

# Metriche
acc_baseline = accuracy_score(y_test, y_pred_baseline)
prec_baseline, rec_baseline, f1_baseline, _ = precision_recall_fscore_support(
    y_test, y_pred_baseline, average='weighted'
)

print("\n📊 BASELINE PERFORMANCE:")
print("-"*70)
print(f"Accuracy:  {acc_baseline:.4f} ({acc_baseline*100:.2f}%)")
print(f"Precision: {prec_baseline:.4f}")
print(f"Recall:    {rec_baseline:.4f}")
print(f"F1-Score:  {f1_baseline:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_baseline))

# Confusion Matrix
print("\n📊 Salvando Confusion Matrix baseline...")
plot_confusion_matrix(
    y_test, y_pred_baseline,
    labels=[1, 2, 3, 4, 5],
    title='Confusion Matrix - Baseline Random Forest',
    save_path='results/figures/07_cm_baseline_rf.png'
)

# Feature Importance
print("\n📊 Feature Importance baseline:")
importance_baseline = pd.DataFrame({
    'Feature': ['Voltage', 'Height', 'Soil Type'],
    'Importance': rf_baseline.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance_baseline)

plot_feature_importance(
    importance_baseline,
    title='Feature Importance - Baseline RF',
    save_path='results/figures/08_feature_importance_baseline.png'
)

# ========================================
# 4. HYPERPARAMETER TUNING
# ========================================
print("\n" + "="*70)
print("🔧 STEP 4: HYPERPARAMETER TUNING")
print("="*70)

# Grid di parametri
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print("\n🔍 Grid Search con 5-fold Cross-Validation...")
print(f"Testando {np.prod([len(v) for v in param_grid.values()])} combinazioni...")
print("⏳ Questo potrebbe richiedere qualche minuto...\n")

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n✓ Grid Search completato!")
print(f"\n🏆 Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n📊 Best CV Score: {grid_search.best_score_:.4f}")

# Modello ottimizzato
rf_optimized = grid_search.best_estimator_

# ========================================
# 5. PERFORMANCE MODELLO OTTIMIZZATO
# ========================================
print("\n" + "="*70)
print("🎯 STEP 5: PERFORMANCE MODELLO OTTIMIZZATO")
print("="*70)

# Predizioni
y_pred_optimized = rf_optimized.predict(X_test)

# Metriche
acc_optimized = accuracy_score(y_test, y_pred_optimized)
prec_optimized, rec_optimized, f1_optimized, _ = precision_recall_fscore_support(
    y_test, y_pred_optimized, average='weighted'
)

print("\n📊 OPTIMIZED PERFORMANCE:")
print("-"*70)
print(f"Accuracy:  {acc_optimized:.4f} ({acc_optimized*100:.2f}%)")
print(f"Precision: {prec_optimized:.4f}")
print(f"Recall:    {rec_optimized:.4f}")
print(f"F1-Score:  {f1_optimized:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_optimized))

# Confusion Matrix
print("\n📊 Salvando Confusion Matrix ottimizzata...")
plot_confusion_matrix(
    y_test, y_pred_optimized,
    labels=[1, 2, 3, 4, 5],
    title='Confusion Matrix - Optimized Random Forest',
    save_path='results/figures/09_cm_optimized_rf.png'
)

# Feature Importance
importance_optimized = pd.DataFrame({
    'Feature': ['Voltage', 'Height', 'Soil Type'],
    'Importance': rf_optimized.feature_importances_
}).sort_values('Importance', ascending=False)

plot_feature_importance(
    importance_optimized,
    title='Feature Importance - Optimized RF',
    save_path='results/figures/10_feature_importance_optimized.png'
)

# ========================================
# 6. CONFRONTO BASELINE vs OPTIMIZED
# ========================================
print("\n" + "="*70)
print("📊 STEP 6: CONFRONTO FINALE")
print("="*70)

comparison = pd.DataFrame({
    'Model': ['Baseline RF', 'Optimized RF'],
    'Accuracy': [acc_baseline, acc_optimized],
    'Precision': [prec_baseline, prec_optimized],
    'Recall': [rec_baseline, rec_optimized],
    'F1-Score': [f1_baseline, f1_optimized]
})

print("\n" + comparison.to_string(index=False))

# Calcola miglioramento
improvement = ((acc_optimized - acc_baseline) / acc_baseline) * 100
print(f"\n🚀 Miglioramento: {improvement:+.2f}%")

# ========================================
# GRAFICI COMPARATIVI MIGLIORATI
# ========================================

# GRAFICO 1: Vista dettagliata per ogni metrica
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for idx, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axes[idx // 2, idx % 2]
    
    values = comparison[metric].values
    x_pos = np.arange(len(comparison))
    
    # Barre
    bars = ax.bar(x_pos, values, width=0.6, color=color, 
                  edgecolor='black', linewidth=2, alpha=0.8)
    
    # Valori sopra le barre con percentuale
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.0002,
               f'{val:.4f}\n({val*100:.2f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Zoom intelligente
    min_val = min(values)
    max_val = max(values)
    margin = (max_val - min_val) * 0.3 if max_val > min_val else 0.001
    ax.set_ylim([min_val - margin, max_val + margin*2])
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison['Model'], fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_facecolor('#F8F9FA')

plt.suptitle('Random Forest: Baseline vs Optimized (Detailed View)', 
            fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('results/figures/11_comparison_rf_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Grafico salvato: 11_comparison_rf_detailed.png")

# GRAFICO 2: Miglioramenti percentuali
fig, ax = plt.subplots(figsize=(12, 7))

differences = {}
for metric in metrics:
    baseline_val = comparison.loc[0, metric]
    optimized_val = comparison.loc[1, metric]
    diff_pct = ((optimized_val - baseline_val) / baseline_val) * 100
    differences[metric] = diff_pct

x_pos = np.arange(len(metrics))
values = [differences[m] for m in metrics]
colors_bar = ['green' if v >= 0 else 'red' for v in values]

bars = ax.bar(x_pos, values, color=colors_bar, edgecolor='black', 
              linewidth=2, alpha=0.7, width=0.6)

for i, (bar, val) in enumerate(zip(bars, values)):
    y_pos = val + 0.002 if val >= 0 else val - 0.002
    va = 'bottom' if val >= 0 else 'top'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
           f'{val:+.3f}%',
           ha='center', va=va, fontsize=12, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
ax.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
ax.set_title('Performance Improvement: Optimized vs Baseline RF', 
            fontsize=15, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('results/figures/11_comparison_rf_improvement.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Grafico salvato: 11_comparison_rf_improvement.png")

# GRAFICO 3: Tabella riassuntiva
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table_data = []
for _, row in comparison.iterrows():
    table_data.append([
        row['Model'],
        f"{row['Accuracy']:.6f} ({row['Accuracy']*100:.3f}%)",
        f"{row['Precision']:.6f}",
        f"{row['Recall']:.6f}",
        f"{row['F1-Score']:.6f}"
    ])

# Aggiungi riga differenza
diff_acc = comparison.loc[1, 'Accuracy'] - comparison.loc[0, 'Accuracy']
diff_pct_acc = (diff_acc / comparison.loc[0, 'Accuracy']) * 100
table_data.append([
    'Difference',
    f"{diff_acc:+.6f} ({diff_pct_acc:+.4f}%)",
    f"{comparison.loc[1, 'Precision'] - comparison.loc[0, 'Precision']:+.6f}",
    f"{comparison.loc[1, 'Recall'] - comparison.loc[0, 'Recall']:+.6f}",
    f"{comparison.loc[1, 'F1-Score'] - comparison.loc[0, 'F1-Score']:+.6f}"
])

table = ax.table(cellText=table_data,
                colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                cellLoc='center',
                loc='center',
                colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

for i in range(5):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(3, i)].set_facecolor('#FFA07A')
    table[(3, i)].set_text_props(weight='bold')

plt.title('Random Forest Comparison - Detailed Metrics', 
         fontsize=15, fontweight='bold', pad=20)
plt.savefig('results/figures/11_comparison_rf_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Grafico salvato: 11_comparison_rf_table.png")
print("\n✓ Grafico salvato: results/figures/11_comparison_rf.png")

# ========================================
# 7. SALVATAGGIO RISULTATI
# ========================================
print("\n" + "="*70)
print("💾 STEP 7: SALVATAGGIO RISULTATI")
print("="*70)

# Salva modelli
import joblib
joblib.dump(rf_baseline, 'results/models/rf_baseline.pkl')
joblib.dump(rf_optimized, 'results/models/rf_optimized.pkl')
print("✓ Modelli salvati in results/models/")

# Salva metriche
comparison.to_csv('results/metrics/rf_comparison.csv', index=False)
importance_optimized.to_csv('results/metrics/feature_importance.csv', index=False)
print("✓ Metriche salvate in results/metrics/")

# Salva best params
with open('results/metrics/best_params_rf.txt', 'w') as f:
    f.write("BEST PARAMETERS - RANDOM FOREST\n")
    f.write("="*50 + "\n\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"{param}: {value}\n")
    f.write(f"\nBest CV Score: {grid_search.best_score_:.4f}\n")
    f.write(f"Test Accuracy: {acc_optimized:.4f}\n")
print("✓ Parametri salvati in results/metrics/best_params_rf.txt")

# ========================================
# RIEPILOGO FINALE
# ========================================
print("\n" + "="*70)
print("✅ RANDOM FOREST COMPLETATO!")
print("="*70)

print("\n📊 GRAFICI GENERATI:")
print("  7. cm_baseline_rf.png - Confusion Matrix Baseline")
print("  8. feature_importance_baseline.png")
print("  9. cm_optimized_rf.png - Confusion Matrix Optimized")
print(" 10. feature_importance_optimized.png")
print(" 11. comparison_rf_detailed.png - Vista dettagliata")
print(" 11. comparison_rf_improvement.png - Miglioramenti %")
print(" 11. comparison_rf_table.png - Tabella completa")

print("\n💾 MODELLI SALVATI:")
print("  - rf_baseline.pkl")
print("  - rf_optimized.pkl")

print("\n📈 RISULTATI FINALI:")
print(f"  Baseline Accuracy:  {acc_baseline:.4f} ({acc_baseline*100:.2f}%)")
print(f"  Optimized Accuracy: {acc_optimized:.4f} ({acc_optimized*100:.2f}%)")
print(f"  Miglioramento:      {improvement:+.2f}%")

print("\n🎯 PROSSIMI PASSI:")
print("  1. Esegui: python run_neural_networks.py")
print("  2. Poi: python run_clustering.py")
print("  3. Crea relazione finale")

print("\n" + "="*70)