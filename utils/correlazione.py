import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
# Correggiamo l'import per la funzione preprocessing
from utils.preprocessing import preprocessing

def analyze_grades(df):
    """Analizza e visualizza statistiche e distribuzione dei voti G1, G2, G3"""
    # Analisi dei voti G1, G2, G3
    target_vars = ['G1', 'G2', 'G3']
    voti_stats = df[target_vars].describe()
    print(voti_stats)
    
    # Visualizzazione distribuzione voti
    plt.figure(figsize=(15, 5))
    for i, var in enumerate(target_vars):
        plt.subplot(1, 3, i+1)
        plt.hist(df[var], bins=20, alpha=0.7, color=['blue', 'green', 'red'][i], edgecolor='black')
        plt.title(f'Distribuzione {var}')
        plt.xlabel(f'{var} (normalizzato)')
        plt.ylabel('Frequenza')
    
    plt.suptitle('Distribuzione dei Voti (G1, G2, G3)', fontsize=16)
    plt.tight_layout()
    plt.show()

def analyze_demographics(df):
    """Analizza e visualizza le variabili demografiche chiave"""
    # Variabili demografiche chiave
    demo_vars = ['age', 'sex', 'Medu', 'Fedu', 'famsize']
    demo_available = [var for var in demo_vars if var in df.columns]
    
    if len(demo_available) >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
       
        for i, var in enumerate(demo_available[:4]):
            if df[var].nunique() <= 10:  # Variabile categorica o discreta
                df[var].value_counts().plot(kind='bar', ax=axes[i], color='skyblue', alpha=0.7)
                axes[i].set_title(f'Distribuzione {var}')
            else:  # Variabile continua
                axes[i].hist(df[var], bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
                axes[i].set_title(f'Distribuzione {var}')
           
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
       
        plt.suptitle('Distribuzione Variabili Demografiche', fontsize=16)
        plt.tight_layout()
        plt.show()

def analyze_correlations(df):
    """Calcola e visualizza la matrice di correlazione"""
    # Calcola la matrice di correlazione
    correlation_matrix = df.corr()
    print(f"Matrice di correlazione calcolata: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
    
    # Heatmap della matrice di correlazione
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix,
                mask=mask,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8},
                annot_kws={'size': 8})
    plt.title('Matrice di Correlazione', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

def analyze_g3_correlations(df, correlation_matrix):
    """Analizza le correlazioni con G3 (voto finale)"""
    # Analisi delle correlazioni con G3 (voto finale)
    g3_correlations = correlation_matrix['G3'].abs().sort_values(ascending=False)
    for i, (var, corr_abs) in enumerate(g3_correlations.head(15).items(), 1):
        actual_corr = correlation_matrix['G3'][var]
        direction = "Positiva" if actual_corr > 0 else "Negativa"
        print(f"{i:2d}. {var:12s}: {actual_corr:6.3f} ({direction}) - |r|={corr_abs:.3f}")
    
    # Classifica le correlazioni per intensità
    strong_corrs = []
    moderate_corrs = []
    weak_corrs = []
    
    for var in correlation_matrix.columns:
        if var != 'G3':
            corr_val = abs(correlation_matrix['G3'][var])
            if corr_val >= 0.7:
                strong_corrs.append((var, correlation_matrix['G3'][var]))
            elif corr_val >= 0.3:
                moderate_corrs.append((var, correlation_matrix['G3'][var]))
            else:
                weak_corrs.append((var, correlation_matrix['G3'][var]))
    
    print("\nCorrelazioni forti con G3:")
    if strong_corrs:
        for var, corr in strong_corrs:
            print(f"   {var}: {corr:.3f}")
    else:
        print("   Nessuna")
    
    print("\nCorrelazioni moderate con G3 (top 10):")
    if moderate_corrs:
        for var, corr in sorted(moderate_corrs, key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f"   {var}: {corr:.3f}")
    
    # Grafico delle top 10 correlazioni con G3
    top_correlations_g3 = g3_correlations.drop('G3').head(10)
    
    plt.figure(figsize=(14, 8))
    colors = ['crimson' if correlation_matrix['G3'][var] < 0 else 'forestgreen'
              for var in top_correlations_g3.index]
    bars = plt.bar(range(len(top_correlations_g3)), top_correlations_g3.values,
                   color=colors, alpha=0.7, edgecolor='black')
    
    plt.xlabel('Variabili', fontsize=12)
    plt.ylabel('Correlazione Assoluta con G3', fontsize=12)
    plt.title('Top 10 Correlazioni con il Voto Finale (G3)', fontsize=14)
    plt.xticks(range(len(top_correlations_g3)), top_correlations_g3.index, rotation=45, ha='right')
    
    # Aggiungi etichette sui bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        actual_corr = correlation_matrix['G3'][top_correlations_g3.index[i]]
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{actual_corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def analyze_grade_relationships(df):
    """Analizza le relazioni tra i voti G1, G2 e G3"""
    grades_corr = df[['G1', 'G2', 'G3']].corr()
    print("\nCorrelazione tra i voti:")
    print(grades_corr.round(3))
    
    # Scatter plots delle correlazioni tra voti
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    pairs = [('G1', 'G2'), ('G1', 'G3'), ('G2', 'G3')]
    colors_scatter = ['blue', 'green', 'red']
    
    for i, (var1, var2) in enumerate(pairs):
        corr_val = grades_corr.loc[var1, var2]
        axes[i].scatter(df[var1], df[var2], alpha=0.6, color=colors_scatter[i], s=30)
        axes[i].set_xlabel(f'{var1} (normalizzato)')
        axes[i].set_ylabel(f'{var2} (normalizzato)')
        axes[i].set_title(f'{var1} vs {var2}\n(r = {corr_val:.3f})')
        axes[i].grid(True, alpha=0.3)
       
        # Aggiungi linea di trend
        z = np.polyfit(df[var1], df[var2], 1)
        p = np.poly1d(z)
        axes[i].plot(df[var1], p(df[var1]), "r--", alpha=0.8, linewidth=2)
    
    plt.suptitle('Correlazioni tra i Voti con Linee di Trend', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("\nOsservazioni:")
    print("- G1, G2 e G3 hanno una forte correlazione, gli studenti hanno prestazioni costanti")
    print("- C'è una correlazione positiva tra higher e G3: l'aspirazione all'istruzione superiore migliora le performance")
    print("- Con Medu e Fedu vediamo che l'istruzione superiore influenza positivamente i risultati scolastici")
    print("- famrel: buone relazioni familiari correlano con migliori voti")
    print("- Dalc/Walc: consumo di alcol correlato con voti più bassi")

def run_correlation_analysis():
    """Funzione wrapper che esegue l'intera analisi di correlazione"""
    # Carica i dati preprocessati
    print("Caricamento e preprocessing dei dati...")
    df = preprocessing()
    
    print("\n" + "="*80)
    print("ANALISI DEI VOTI (G1, G2, G3)")
    print("="*80)
    analyze_grades(df)
    
    print("\n" + "="*80)
    print("ANALISI DEMOGRAFICA")
    print("="*80)
    analyze_demographics(df)
    
    print("\n" + "="*80)
    print("ANALISI DELLE CORRELAZIONI")
    print("="*80)
    correlation_matrix = analyze_correlations(df)
    
    print("\n" + "="*80)
    print("CORRELAZIONI CON IL VOTO FINALE (G3)")
    print("="*80)
    analyze_g3_correlations(df, correlation_matrix)
    
    print("\n" + "="*80)
    print("RELAZIONI TRA I VOTI")
    print("="*80)
    analyze_grade_relationships(df)

# Modifichiamo la funzione main esistente per usare run_correlation_analysis
def main():
    """Funzione principale che coordina l'analisi dei dati"""
    run_correlation_analysis()

if __name__ == "__main__":
    main()