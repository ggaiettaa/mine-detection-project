"""
NEURAL NETWORKS - Mine Detection
Tre architetture diverse con confronto
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Fix per warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# Import custom
sys.path.append('src')
from visualization import plot_confusion_matrix, plot_training_history

# Set seeds per riproducibilità
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("🧠 NEURAL NETWORKS - MINE DETECTION")
print("="*70)

# ========================================
# 1. CARICAMENTO E PREPROCESSING
# ========================================
print("\n📊 STEP 1: Caricamento e preprocessing...")
print("-"*70)

df = pd.read_excel('data/raw/Mine_Dataset.xls', sheet_name='Normalized_Data')
print(f"✓ Dati caricati: {df.shape}")

# Features e target
X = df[['V', 'H', 'S']].values
y = df['M'].values - 1  # Converti 1-5 in 0-4 per Keras

print(f"✓ Features shape: {X.shape}")
print(f"✓ Target shape: {y.shape}")
print(f"✓ Classi: {np.unique(y)} (0=No mine, 1-4=Mine types)")

# Standardizzazione (importante per NN!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Features standardizzate")

# Split: train/val/test (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"\n✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Validation set: {X_val.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Crea cartelle
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/models', exist_ok=True)

# ========================================
# 2. ARCHITETTURA SEMPLICE (2 layers)
# ========================================
print("\n" + "="*70)
print("🎯 STEP 2: ARCHITETTURA SEMPLICE")
print("="*70)

print("\n🏗️ Costruzione modello...")
model_simple = Sequential([
    Dense(32, activation='relu', input_dim=3, name='hidden1'),
    Dense(16, activation='relu', name='hidden2'),
    Dense(5, activation='softmax', name='output')
], name='SimpleNN')

model_simple.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Architettura:")
model_simple.summary()

print("\n🏋️ Training...")
history_simple = model_simple.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ],
    verbose=1
)

# Valutazione
print("\n📊 Valutazione su test set...")
test_loss_simple, test_acc_simple = model_simple.evaluate(X_test, y_test, verbose=0)
y_pred_simple = np.argmax(model_simple.predict(X_test, verbose=0), axis=1)

print(f"✓ Test Accuracy: {test_acc_simple:.4f} ({test_acc_simple*100:.2f}%)")
print(f"✓ Test Loss: {test_loss_simple:.4f}")

# ========================================
# 3. ARCHITETTURA MEDIA (3 layers + Dropout)
# ========================================
print("\n" + "="*70)
print("🎯 STEP 3: ARCHITETTURA MEDIA (+ Dropout)")
print("="*70)

print("\n🏗️ Costruzione modello...")
model_medium = Sequential([
    Dense(64, activation='relu', input_dim=3, name='hidden1'),
    Dropout(0.3, name='dropout1'),
    Dense(32, activation='relu', name='hidden2'),
    Dropout(0.3, name='dropout2'),
    Dense(16, activation='relu', name='hidden3'),
    Dense(5, activation='softmax', name='output')
], name='MediumNN')

model_medium.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Architettura:")
model_medium.summary()

print("\n🏋️ Training...")
history_medium = model_medium.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ],
    verbose=1
)

# Valutazione
print("\n📊 Valutazione su test set...")
test_loss_medium, test_acc_medium = model_medium.evaluate(X_test, y_test, verbose=0)
y_pred_medium = np.argmax(model_medium.predict(X_test, verbose=0), axis=1)

print(f"✓ Test Accuracy: {test_acc_medium:.4f} ({test_acc_medium*100:.2f}%)")
print(f"✓ Test Loss: {test_loss_medium:.4f}")

# ========================================
# 4. ARCHITETTURA DEEP (4 layers + BatchNorm)
# ========================================
print("\n" + "="*70)
print("🎯 STEP 4: ARCHITETTURA PROFONDA (+ Batch Normalization)")
print("="*70)

print("\n🏗️ Costruzione modello...")
model_deep = Sequential([
    Dense(128, activation='relu', input_dim=3, name='hidden1'),
    BatchNormalization(name='bn1'),
    Dropout(0.3, name='dropout1'),
    Dense(64, activation='relu', name='hidden2'),
    BatchNormalization(name='bn2'),
    Dropout(0.3, name='dropout2'),
    Dense(32, activation='relu', name='hidden3'),
    Dropout(0.2, name='dropout3'),
    Dense(16, activation='relu', name='hidden4'),
    Dense(5, activation='softmax', name='output')
], name='DeepNN')

model_deep.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📋 Architettura:")
model_deep.summary()

print("\n🏋️ Training...")
history_deep = model_deep.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ],
    verbose=1
)

# Valutazione
print("\n📊 Valutazione su test set...")
test_loss_deep, test_acc_deep = model_deep.evaluate(X_test, y_test, verbose=0)
y_pred_deep = np.argmax(model_deep.predict(X_test, verbose=0), axis=1)

print(f"✓ Test Accuracy: {test_acc_deep:.4f} ({test_acc_deep*100:.2f}%)")
print(f"✓ Test Loss: {test_loss_deep:.4f}")

# ========================================
# 5. VISUALIZZAZIONE TRAINING HISTORIES
# ========================================
print("\n" + "="*70)
print("📊 STEP 5: VISUALIZZAZIONE TRAINING")
print("="*70)

# Plot per ogni modello
models_info = [
    ('Simple', history_simple, 'simple'),
    ('Medium', history_medium, 'medium'),
    ('Deep', history_deep, 'deep')
]

for name, history, suffix in models_info:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title(f'{name} NN - Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title(f'{name} NN - Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/figures/12_training_{suffix}_nn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvato: 12_training_{suffix}_nn.png")

# Confronto tutte le histories insieme
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = ['#4ECDC4', '#45B7D1', '#96CEB4']
for (name, history, _), color in zip(models_info, colors):
    axes[0].plot(history.history['val_accuracy'], label=f'{name} NN', 
                linewidth=2.5, color=color)
    axes[1].plot(history.history['val_loss'], label=f'{name} NN', 
                linewidth=2.5, color=color)

axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Validation Accuracy', fontsize=13, fontweight='bold')
axes[0].set_title('Comparison: Validation Accuracy', fontsize=15, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Validation Loss', fontsize=13, fontweight='bold')
axes[1].set_title('Comparison: Validation Loss', fontsize=15, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/13_comparison_all_nn.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Salvato: 13_comparison_all_nn.png")

# ========================================
# 6. CONFUSION MATRICES
# ========================================
print("\n" + "="*70)
print("📊 STEP 6: CONFUSION MATRICES")
print("="*70)

predictions = [
    ('Simple NN', y_pred_simple),
    ('Medium NN', y_pred_medium),
    ('Deep NN', y_pred_deep)
]

for i, (name, y_pred) in enumerate(predictions):
    plot_confusion_matrix(
        y_test, y_pred,
        labels=[0, 1, 2, 3, 4],
        title=f'Confusion Matrix - {name}',
        save_path=f'results/figures/14_cm_{name.lower().replace(" ", "_")}.png'
    )
    print(f"✓ Salvato: 14_cm_{name.lower().replace(' ', '_')}.png")

# ========================================
# 7. CONFRONTO FINALE
# ========================================
print("\n" + "="*70)
print("📊 STEP 7: CONFRONTO FINALE")
print("="*70)

# Tabella comparativa
comparison_nn = pd.DataFrame({
    'Model': ['Simple NN', 'Medium NN', 'Deep NN'],
    'Test Accuracy': [test_acc_simple, test_acc_medium, test_acc_deep],
    'Test Loss': [test_loss_simple, test_loss_medium, test_loss_deep],
    'Epochs Trained': [
        len(history_simple.history['loss']),
        len(history_medium.history['loss']),
        len(history_deep.history['loss'])
    ]
})

print("\n" + comparison_nn.to_string(index=False))

# Salva
comparison_nn.to_csv('results/metrics/nn_comparison.csv', index=False)
print("\n✓ Tabella salvata: results/metrics/nn_comparison.csv")

# ========================================
# GRAFICI COMPARATIVI MIGLIORATI
# ========================================

# GRAFICO 1: Vista dettagliata metriche
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Test Accuracy
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

ax.set_xlabel('Architecture', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Neural Networks - Test Accuracy', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_nn['Model'], fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_facecolor('#F8F9FA')

# Subplot 2: Test Loss
ax = axes[1]
bars = ax.bar(x_pos, comparison_nn['Test Loss'], width=0.6,
              color=['#FFA07A', '#FF6B6B', '#FF8C94'],
              edgecolor='black', linewidth=2, alpha=0.8)

for bar, val in zip(bars, comparison_nn['Test Loss']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.002,
           f'{val:.4f}',
           ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Architecture', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Loss', fontsize=13, fontweight='bold')
ax.set_title('Neural Networks - Test Loss (lower is better)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_nn['Model'], fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_facecolor('#F8F9FA')

plt.suptitle('Neural Networks Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/15_comparison_nn_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Salvato: 15_comparison_nn_detailed.png")

# GRAFICO 2: Tabella comparativa
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('tight')
ax.axis('off')

table_data = []
for _, row in comparison_nn.iterrows():
    table_data.append([
        row['Model'],
        f"{row['Test Accuracy']:.6f} ({row['Test Accuracy']*100:.3f}%)",
        f"{row['Test Loss']:.6f}",
        f"{row['Epochs Trained']}"
    ])

# Trova il migliore
best_idx = comparison_nn['Test Accuracy'].idxmax()
best_model = comparison_nn.loc[best_idx, 'Model']

table_data.append([
    f'🏆 Best: {best_model}',
    f"{comparison_nn.loc[best_idx, 'Test Accuracy']:.6f}",
    f"{comparison_nn.loc[best_idx, 'Test Loss']:.6f}",
    '-'
])

table = ax.table(cellText=table_data,
                colLabels=['Model', 'Test Accuracy', 'Test Loss', 'Epochs'],
                cellLoc='center',
                loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.8)

# Colora header
for i in range(4):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white', size=13)

# Colora best model
for i in range(4):
    table[(4, i)].set_facecolor('#FFD700')
    table[(4, i)].set_text_props(weight='bold', size=12)

# Evidenzia la riga del best model
for i in range(4):
    table[(best_idx + 1, i)].set_facecolor('#E8F8F5')
    table[(best_idx + 1, i)].set_text_props(weight='bold')

plt.title('Neural Networks - Complete Comparison Table', 
         fontsize=16, fontweight='bold', pad=20)
plt.savefig('results/figures/15_comparison_nn_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Salvato: 15_comparison_nn_table.png")

# GRAFICO 3: Confronto grafico accuratezza vs complessità
fig, ax = plt.subplots(figsize=(12, 7))

# Parametri (numero approssimativo)
params = [
    (32+16)*3 + 32+16+5,  # Simple
    (64+32+16)*3 + 64+32+16+5,  # Medium
    (128+64+32+16)*3 + 128+64+32+16+5  # Deep
]

colors = ['#4ECDC4', '#45B7D1', '#96CEB4']
sizes = [comparison_nn.loc[i, 'Test Accuracy']*3000 for i in range(3)]

for i, (model, acc, loss, param, color, size) in enumerate(zip(
    comparison_nn['Model'], 
    comparison_nn['Test Accuracy'],
    comparison_nn['Test Loss'],
    params,
    colors,
    sizes
)):
    ax.scatter(param, acc, s=size, alpha=0.6, color=color, 
              edgecolors='black', linewidth=2, label=model)
    ax.text(param, acc + 0.001, f'{model}\n{acc:.4f}', 
           ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Model Complexity (approx. parameters)', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Neural Networks: Accuracy vs Complexity\n(bubble size = accuracy)', 
            fontsize=15, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('results/figures/15_comparison_nn_complexity.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Salvato: 15_comparison_nn_complexity.png")

# ========================================
# 8. SALVATAGGIO MODELLI
# ========================================
print("\n" + "="*70)
print("💾 STEP 8: SALVATAGGIO MODELLI")
print("="*70)

model_simple.save('results/models/nn_simple.h5')
model_medium.save('results/models/nn_medium.h5')
model_deep.save('results/models/nn_deep.h5')
print("✓ Modelli salvati in results/models/")

# ========================================
# RIEPILOGO FINALE
# ========================================
print("\n" + "="*70)
print("✅ NEURAL NETWORKS COMPLETATO!")
print("="*70)

print("\n📊 RISULTATI FINALI:")
print(f"  Simple NN:  {test_acc_simple:.4f} ({test_acc_simple*100:.2f}%)")
print(f"  Medium NN:  {test_acc_medium:.4f} ({test_acc_medium*100:.2f}%)")
print(f"  Deep NN:    {test_acc_deep:.4f} ({test_acc_deep*100:.2f}%)")

best_model = comparison_nn.loc[comparison_nn['Test Accuracy'].idxmax(), 'Model']
print(f"\n🏆 Miglior modello: {best_model}")

print("\n📁 FILES GENERATI:")
print("  Grafici training: 12_training_*.png (3 files)")
print("  Confronto histories: 13_comparison_all_nn.png")
print("  Confusion matrices: 14_cm_*.png (3 files)")
print("  Confronto dettagliato: 15_comparison_nn_detailed.png")
print("  Tabella completa: 15_comparison_nn_table.png")
print("  Complessità: 15_comparison_nn_complexity.png")

print("\n🎯 PROSSIMO PASSO:")
print("  Esegui: python run_clustering.py")

print("\n" + "="*70)