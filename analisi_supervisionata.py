from preprocessing import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def analisi_supervisionata():
    """
    Esegue un'analisi supervisionata sul dataset degli studenti usando Random Forest.

    Carica e pre-processa i dati, crea una variabile target binaria per il rendimento
    (alto/basso rispetto alla media globale dei voti), addestra un classificatore Random Forest,
    valuta le performance e visualizza l'importanza delle feature.

    Returns
    -------
    dict
        Un dizionario con le metriche di valutazione:
        - 'confusion_matrix': matrice di confusione
        - 'classification_report': report di classificazione (stringa)
        - 'accuracy_score': accuratezza (float)
        - 'feature_importances': importanza delle feature (array)
        - 'feature_names': nomi delle feature (array)
    """
    # Carica e preprocessa il dataset
    df = preprocessing()

    # Calcola la media globale dei voti
    media_globale = df[['G1', 'G2', 'G3']].mean(axis=1).mean()

    # Definisci la soglia per rendimento alto/basso rispetto alla media globale
    df['media_voti'] = df[['G1', 'G2', 'G3']].mean(axis=1)
    df['rendimento'] = np.where(df['media_voti'] > media_globale, 1, 0)  # 1 = alto, 0 = basso

    # Seleziona le variabili socio-demografiche come feature (escludi i voti e la media)
    features = df.drop(columns=['G1', 'G2', 'G3', 'media_voti', 'rendimento'])
    target = df['rendimento']

    # Suddividi il dataset in training e test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Inizializza il classificatore Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)

    # Addestra il modello sui dati di training
    rf.fit(X_train, y_train)

    # Effettua le predizioni sui dati di test
    y_pred = rf.predict(X_test)

    # Valuta le performance del modello
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Visualizza la confusion matrix con matplotlib
    plt.figure()
    plt.title("Confusion Matrix")
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Basso', 'Alto'])
    plt.yticks(tick_marks, ['Basso', 'Alto'])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # Visualizza l'importanza delle feature
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Crea un grafico a barre delle feature più importanti
    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), features.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    return {
        'confusion_matrix': cm,
        'classification_report': cr,
        'accuracy_score': acc,
        'feature_importances': importances,
        'feature_names': features.columns.values
    }

if __name__ == "__main__":
    results = analisi_supervisionata()
    print("Analisi completata con successo!")
    print(f"Accuratezza: {results['accuracy_score']:.2f}")
    print("Matrice di confusione:")
    print(results['confusion_matrix'])
    print("Report di classificazione:")
    print(results['classification_report'])