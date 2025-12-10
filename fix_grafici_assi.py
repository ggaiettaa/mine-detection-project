import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os

print("="*70)
print("🔧 FIX ETICHETTE ASSI - Valori Reali")
print("="*70)

# Carica dati
df = pd.read_excel('data/raw/Mine_Dataset.xls', sheet_name='Normalized_Data')
print(f"✓ Dati caricati: {df.shape}")

# VERIFICA VALORI TIPO SUOLO
print(f"\n🔍 Verifica valori Tipo Suolo (S):")
print(f"   Range S: {df['S'].min()} - {df['S'].max()}")
print(f"   Valori unici S: {sorted(df['S'].unique())}")
print(f"   Distribuzione S:\n{df['S'].value_counts().sort_index()}")

# VALORI REALI (dalle specifiche)
V_MIN, V_MAX = 0, 5      # Voltaggio: 0-5V
H_MIN, H_MAX = 0, 30     # Altezza: 0-30cm
S_MIN, S_MAX = 0, 5      # Tipo Suolo: 0-5 (6 categorie)
S_LABELS = ['Sabbioso\\nSecco', 'Humus\\nSecco', 'Calcareo\\nSecco',
            'Sabbioso\\nUmido', 'Humus\\nUmido', 'Calcareo\\nUmido']

# Funzioni per denormalizzare
def denorm_V(v_norm):
    """Converte voltaggio normalizzato (0-1) in voltaggio reale (0-5V)"""
    return v_norm * (V_MAX - V_MIN) + V_MIN

def denorm_H(h_norm):
    """Converte altezza normalizzata (0-1) in altezza reale (0-30cm)"""
    return h_norm * (H_MAX - H_MIN) + H_MIN

def denorm_S(s_norm):
    """Converte tipo suolo normalizzato (0-1) in categorie discrete (1-6)"""
    # S normalizzato va da 0 a 1, lo mappiamo su 0-5 e poi aggiungiamo 1 per avere 1-6
    s_denorm = np.round(s_norm * (S_MAX - S_MIN) + S_MIN).astype(int)
    return s_denorm + 1  # Converti 0-5 in 1-6 per presentazione

# Crea un DataFrame con i valori reali - INCLUDE ANCHE 'M' per il pairplot
df_real = pd.DataFrame()
df_real['Voltaggio (V)'] = denorm_V(df['V'])
df_real['Altezza (cm)'] = denorm_H(df['H'])
df_real['Tipo Suolo'] = denorm_S(df['S'])  # Denormalizza S: da (0-1) a (1-6)
df_real['M'] = df['M'].astype(int)  # IMPORTANTE: necessario per hue nel pairplot

print(f"\n🔍 Verifica df_real Tipo Suolo:")
print(f"   Range Tipo Suolo in df_real: {df_real['Tipo Suolo'].min()} - {df_real['Tipo Suolo'].max()}")
print(f"   Valori unici: {sorted(df_real['Tipo Suolo'].unique())}")
print(f"   Distribuzione:\n{df_real['Tipo Suolo'].value_counts().sort_index()}")

df_norm = df.copy() # Dati normalizzati originali

os.makedirs('results/figures_corretti', exist_ok=True)

print("\n📊 Rigenerazione grafici con assi corretti...")

# ========================================
# GRAFICO 1: OVERVIEW CON ASSI CORRETTI
# ========================================
print("  1/6 - Panoramica dati...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Distribuzione Mine Types (OK, non ha assi da fixare)
df['M'].value_counts().sort_index().plot(
    kind='bar', ax=axes[0, 0], 
    color=['red', 'blue', 'green', 'orange', 'purple'],
    edgecolor='black'
)
axes[0, 0].set_title('Distribuzione Tipi di Mine', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Tipo Mina\n(1=Nessuna, 2=Anti-carro, 3-5=Anti-uomo)', fontsize=10)
axes[0, 0].set_ylabel('Numero Campioni')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribuzione Voltaggio - CON ASSI CORRETTI
V_real = denorm_V(df['V'])
axes[0, 1].hist(V_real, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 1].set_title('Distribuzione Voltaggio', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Voltaggio (V) - Range: 0-5V', fontsize=10)
axes[0, 1].set_ylabel('Frequenza')
axes[0, 1].grid(True, alpha=0.3)

# 3. Distribuzione Altezza - CON ASSI CORRETTI
H_real = denorm_H(df['H'])
axes[1, 0].hist(H_real, bins=30, edgecolor='black', alpha=0.7, color='coral')
axes[1, 0].set_title('Distribuzione Altezza Sensore', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Altezza dal Suolo (cm) - Range: 0-30cm', fontsize=10)
axes[1, 0].set_ylabel('Frequenza')
axes[1, 0].grid(True, alpha=0.3)

# 4. Distribuzione Suolo - CON ETICHETTE
# Denormalizza S per ottenere le categorie 1-6
S_denorm = denorm_S(df['S'])
soil_counts = S_denorm.value_counts().sort_index()
x_pos = np.arange(1, 7)  # Posizioni 1-6
axes[1, 1].bar(x_pos, soil_counts.values, 
               edgecolor='black', alpha=0.7, color='lightgreen')
axes[1, 1].set_title('Distribuzione Tipo Suolo', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Tipo Suolo', fontsize=10)
axes[1, 1].set_ylabel('Frequenza')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels([f'{i+1}\n{S_LABELS[i].replace(chr(92)+"n", chr(10))}' for i in range(6)], fontsize=7)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('results/figures_corretti/01_panoramica_corretta.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# ========================================
# GRAFICO 2: VOLTAGGIO PER TIPO MINA
# ========================================
print("  2/6 - Voltaggio per tipo mina...")

fig, ax = plt.subplots(figsize=(14, 7))

mine_types = sorted(df['M'].unique())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

# Prepara dati denormalizzati
data_by_type = [denorm_V(df[df['M'] == m]['V']).values for m in mine_types]

bp = ax.boxplot(data_by_type, 
                labels=[f'Tipo {m}' for m in mine_types],
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=8),
                medianprops=dict(color='darkblue', linewidth=2.5),
                boxprops=dict(linewidth=1.5, edgecolor='black'),
                whiskerprops=dict(linewidth=1.5, color='black'),
                capprops=dict(linewidth=1.5, color='black'),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.6))

for patch, color in zip(bp['boxes'], colors[:len(mine_types)]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Statistiche
for i, mine_type in enumerate(mine_types, 1):
    mine_data = denorm_V(df[df['M'] == mine_type]['V'])
    n = len(mine_data)
    mean_val = mine_data.mean()
    stats_text = f'n={n}\nμ={mean_val:.2f}V'
    ax.text(i, mine_data.max() + 0.2, stats_text, 
           ha='center', va='bottom', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

ax.set_title('Distribuzione Voltaggio per Tipo Mina', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Tipo Mina', fontsize=13, fontweight='bold')
ax.set_ylabel('Voltaggio (V) - Range: 0-5V', fontsize=13, fontweight='bold')
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
ax.set_facecolor('#F8F9FA')

legend_elements = [
    plt.Line2D([0], [0], marker='D', color='w', label='Media',
              markerfacecolor='red', markersize=8),
    plt.Line2D([0], [0], color='darkblue', linewidth=2.5, label='Mediana')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('results/figures_corretti/02_voltaggio_per_tipo_corretto.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# ========================================
# GRAFICO 3: SCATTER V vs H CORRETTO
# ========================================
print("  3/6 - Scatter voltaggio vs altezza...")

fig, ax = plt.subplots(figsize=(12, 9))

V_real = denorm_V(df['V'])
H_real = denorm_H(df['H'])

scatter = ax.scatter(V_real, H_real, c=df['M'], cmap='viridis', 
                    alpha=0.6, edgecolors='black', s=60, linewidth=0.5)

ax.set_xlabel('Voltaggio (V) - Anomalia Magnetica [0-5V]', fontsize=13, fontweight='bold')
ax.set_ylabel('Altezza Sensore (cm) - Dal Suolo [0-30cm]', fontsize=13, fontweight='bold')
ax.set_title('Voltaggio vs Altezza Sensore\n(colorato per Tipo Mina)', 
            fontsize=15, fontweight='bold')

# Limiti assi
ax.set_xlim([-0.2, 5.2])
ax.set_ylim([-1, 31])

cbar = plt.colorbar(scatter, ax=ax, label='Tipo Mina')
cbar.set_ticks([1, 2, 3, 4, 5])
cbar.set_ticklabels(['1: Nessuna', '2: Anti-carro', '3: Anti-uomo 1', 
                     '4: Anti-uomo 2', '5: Anti-uomo 3'])

ax.grid(True, alpha=0.3)
ax.set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('results/figures_corretti/04_scatter_corretto.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# ========================================
# GRAFICO 4: ANALISI SUOLO DETTAGLIATA
# ========================================
print("  4/6 - Analisi tipo suolo...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Campioni per suolo
S_denorm = denorm_S(df['S'])
soil_counts = S_denorm.value_counts().sort_index()
x_pos = np.arange(1, 7)  # Posizioni 1-6
bars = axes[0].bar(x_pos, soil_counts.values,
                   edgecolor='black', linewidth=1.5, 
                   color=['#FFE5B4', '#D2B48C', '#F5DEB3', '#87CEEB', '#4682B4', '#6495ED'])

for bar, val in zip(bars, soil_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 2,
                str(int(val)), ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[0].set_title('Campioni per Tipo Suolo', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tipo Suolo', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Numero Campioni', fontsize=12, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([f'{i+1}\n{S_LABELS[i].replace(chr(92)+"n", chr(10))}' for i in range(6)], fontsize=8)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_facecolor('#F8F9FA')

# Subplot 2: Mine per suolo
S_denorm = denorm_S(df['S'])
crosstab = pd.crosstab(S_denorm, df['M'])
crosstab.plot(kind='bar', stacked=True, ax=axes[1], edgecolor='black', linewidth=1)
axes[1].set_title('Distribuzione Tipi Mine per Tipo Suolo', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tipo Suolo', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Numero Campioni', fontsize=12, fontweight='bold')
axes[1].set_xticklabels([f'{i+1}\n{S_LABELS[i].replace(chr(92)+"n", chr(10))}' for i in range(6)], rotation=45, ha='right', fontsize=8)
axes[1].legend(title='Tipo Mina', labels=['1: Nessuna', '2: Anti-carro', 
                                           '3: Anti-uomo 1', '4: Anti-uomo 2', '5: Anti-uomo 3'],
              bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_facecolor('#F8F9FA')

plt.tight_layout()
plt.savefig('results/figures_corretti/05_analisi_suolo_corretta.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")


# =======================================
# GRAFICO 5: Pairplot (Valori Reali) - CORRETTO
# =======================================
print("  5/6 - Pairplot...")

# Usa nomi colonne con unità di misura per avere assi chiari
cols_for_pairplot = ['Voltaggio (V)', 'Altezza (cm)', 'Tipo Suolo']
num_vars = len(cols_for_pairplot) # È 3 (per una griglia 3x3)

# Pairplot con colorazione 'M'
g = sns.pairplot(
    df_real,                      # DataFrame con valori reali E con M
    vars=cols_for_pairplot,       # Seleziona le 3 colonne degli assi
    hue='M',                      # 'M' è disponibile in df_real
    palette='husl',            
    corner=False,             
    diag_kind='hist',             # Istogrammi sulla diagonale (più comprensibili)
    plot_kws={'alpha': 0.6, 'edgecolor': 'black', 'linewidth': 0.5, 's': 50},
    diag_kws={'alpha': 0.7, 'edgecolor': 'black', 'linewidth': 1}
)

print(f"    ✓ pairplot creato. Dimensioni assi: {g.axes.shape} (Dovrebbe essere ({num_vars}, {num_vars}))") 

# Impostazione tick personalizzati per Tipo Suolo (ultima colonna/riga)
S_TICKS = np.arange(1, 7)  # Tipo Suolo denormalizzato va da 1 a 6
S_TICKLABELS = [f'{s}' for s in S_TICKS]
idx_S = 2  # Tipo Suolo è la terza variabile (indice 2)

print(f"    ✓ Tick Tipo Suolo: {S_TICKS}")

for i in range(num_vars):
    # Colonna Tipo Suolo (Asse X)
    g.axes[i, idx_S].set_xticks(S_TICKS)
    g.axes[i, idx_S].set_xticklabels(S_TICKLABELS, fontsize=9)
    
    # Riga Tipo Suolo (Asse Y)
    g.axes[idx_S, i].set_yticks(S_TICKS)
    g.axes[idx_S, i].set_yticklabels(S_TICKLABELS, fontsize=9)

# Imposta limiti assi per rendere i valori reali più chiari
for i in range(num_vars):
    for j in range(num_vars):
        ax = g.axes[i, j]
        
        # Limiti asse X
        if j == 0:  # Voltaggio
            ax.set_xlim(-0.3, 5.3)
        elif j == 1:  # Altezza
            ax.set_xlim(-1, 31)
        elif j == 2:  # Tipo Suolo
            ax.set_xlim(0.5, 6.5)
        
        # Limiti asse Y
        if i == 0:  # Voltaggio
            ax.set_ylim(-0.3, 5.3)
        elif i == 1:  # Altezza
            ax.set_ylim(-1, 31)
        elif i == 2:  # Tipo Suolo
            ax.set_ylim(0.5, 6.5)

# Titolo principale
plt.suptitle("Pairplot con Valori Reali - Classificato per Tipo Mina (M)", 
             fontsize=16, fontweight='bold', y=1.02)

# Migliora la legenda e spostala fuori dal grafico
g._legend.set_title('Tipo Mina', prop={'weight': 'bold', 'size': 11})
new_labels = ['1: Nessuna', '2: Anti-carro', '3: Anti-uomo 1', 
              '4: Anti-uomo 2', '5: Anti-uomo 3']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)
    t.set_fontsize(10)

# Sposta la legenda in alto a destra, fuori dalla griglia
g._legend.set_bbox_to_anchor((1.02, 0.98))
g._legend.set_loc('upper left')

g.fig.tight_layout()

# Salvataggio
plt.savefig('results/figures_corretti/06_pairplot_valori_reali.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# ========================================
# GRAFICO 6: STATISTICHE DESCRITTIVE
# ========================================
print("  6/6 - Tabella statistiche...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Calcola statistiche con valori reali
V_real = denorm_V(df['V'])
H_real = denorm_H(df['H'])
S_real = denorm_S(df['S'])  # Denormalizza anche S

stats_data = [
    ['Voltaggio (V)', f'{V_real.min():.2f}', f'{V_real.max():.2f}', 
     f'{V_real.mean():.2f}', f'{V_real.std():.2f}', f'{V_real.median():.2f}'],
    ['Altezza (cm)', f'{H_real.min():.2f}', f'{H_real.max():.2f}', 
     f'{H_real.mean():.2f}', f'{H_real.std():.2f}', f'{H_real.median():.2f}'],
    ['Tipo Suolo', f'{int(S_real.min())}', f'{int(S_real.max())}', 
     f'{S_real.mean():.2f}', f'{S_real.std():.2f}', f'{S_real.median():.0f}'],
    ['Tipo Mina', '1', '5', f'{df["M"].mean():.2f}', 
     f'{df["M"].std():.2f}', f'{df["M"].median():.0f}']
]

table = ax.table(cellText=stats_data,
                colLabels=['Feature', 'Min', 'Max', 'Media', 'Dev. Std.', 'Mediana'],
                cellLoc='center',
                loc='center',
                colWidths=[0.2, 0.13, 0.13, 0.13, 0.13, 0.13])

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# Colora header
for i in range(6):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white', size=13)

# Colora righe alternate
for i in range(1, 5):
    color = '#F0F0F0' if i % 2 == 0 else '#FFFFFF'
    for j in range(6):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_text_props(size=11)

plt.title('Statistiche Descrittive - Valori Reali\n(Dati Denormalizzati)', 
         fontsize=16, fontweight='bold', pad=20)
plt.savefig('results/figures_corretti/00_statistiche_reali.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✓ Salvato")

# ========================================
# INFORMAZIONI FINALI
# ========================================
print("\n" + "="*70)
print("✅ GRAFICI CON ASSI CORRETTI CREATI!")
print("="*70)

print("\n📊 Grafici corretti salvati in: results/figures_corretti/")
print("\n📏 VALORI REALI:")
print(f"  • Voltaggio (V): {V_MIN}-{V_MAX}V")
print(f"    Range nel dataset: {V_real.min():.2f}-{V_real.max():.2f}V")
print(f"  • Altezza (H): {H_MIN}-{H_MAX}cm")
print(f"    Range nel dataset: {H_real.min():.2f}-{H_real.max():.2f}cm")
print(f"  • Tipo Suolo (S): 1-6 (6 categorie)")
print(f"  • Tipo Mina (M): 1-5")

print("\n🏷️ LEGENDA TIPI SUOLO:")
for i in range(6):
    print(f"  {i+1}: {S_LABELS[i].replace(chr(92)+'n', ' ')}")

print("\n🏷️ LEGENDA TIPI MINA:")
print("  1: Nessuna mina")
print("  2: Anti-carro")
print("  3: Anti-uomo (Tipo 1)")
print("  4: Anti-uomo (Tipo 2)")
print("  5: Anti-uomo (Tipo 3)")

print("\n💡 NOTA IMPORTANTE:")
print("  I dati nel dataset sono NORMALIZZATI (0-1)")
print("  Questi grafici mostrano i VALORI REALI denormalizzati")
print("  per una migliore comprensione e presentazione!")
print("  Il PAIRPLOT ora mostra VALORI REALI sugli assi:")
print("    - Voltaggio in Volt (0-5V)")
print("    - Altezza in centimetri (0-30cm)")
print("    - Tipo Suolo come categorie (1-6)")

print("\n✅ Usa questi grafici per la tua presentazione!")
print("="*70)