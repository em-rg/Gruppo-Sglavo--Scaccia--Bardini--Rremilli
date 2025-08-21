from utils.preprocessing import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def prepara_dati():
    """
    Prepara i dati per l'analisi, creando variabili binarie per il rendimento e suddividendo il dataset
    in set di addestramento, validazione e test.

    Returns
    -------
    tuple
        - X_train: features per il training
        - X_val: features per la validazione
        - X_test: features per il test
        - y_train: target per il training
        - y_val: target per la validazione
        - y_test: target per il test
        - features: dataframe delle feature utilizzate
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
    X_temp, X_test, y_temp, y_test = train_test_split(features, target, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.175, random_state=42)  

    return X_train, X_val, X_test, y_train, y_val, y_test, features

def addestra_modelli(X_train, y_train):
    """
    Addestra i modelli di classificazione Random Forest e Logistic Regression sui dati di training.

    Parameters
    ----------
    X_train : array-like
        Features per il training
    y_train : array-like
        Target per il training

    Returns
    -------
    tuple
        - rf: modello Random Forest addestrato
        - lr: modello Logistic Regression addestrato
    """
    # Inizializza e addestra il classificatore Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)

    # Inizializza e addestra il classificatore binario 
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    return rf, lr

def valuta_modello(modello, X_val, y_val, X_test, y_test):
    """
    Valuta le performance di un modello di classificazione su set di validazione e test.

    Parameters
    ----------
    modello : object
        Modello di classificazione addestrato
    X_val : array-like
        Features per la validazione
    y_val : array-like
        Target per la validazione
    X_test : array-like
        Features per il test
    y_test : array-like
        Target per il test

    Returns
    -------
    tuple
        - cm: matrice di confusione
        - cr: report di classificazione (stringa)
        - acc: accuratezza sul test (float)
        - acc_val: accuratezza sulla validazione (float)
    """
    y_pred_val = modello.predict(X_val)
    y_pred = modello.predict(X_test)

    # Valuta le performance
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    acc_val = accuracy_score(y_val, y_pred_val)

    return cm, cr, acc, acc_val

def visualizza_confusion_matrix(cm, title, cmap):
    """
    Visualizza la matrice di confusione con un grafico a colori.

    Parameters
    ----------
    cm : array-like
        Matrice di confusione da visualizzare
    title : str
        Titolo del grafico
    cmap : Colormap
        Mappa dei colori da utilizzare per il grafico
    """
    plt.figure()
    plt.title(title)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
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

def visualizza_importanza_feature(rf, features, X_train):
    """
    Visualizza l'importanza delle feature nel modello Random Forest.

    Parameters
    ----------
    rf : object
        Modello Random Forest addestrato
    features : DataFrame
        DataFrame contenente i nomi delle feature
    X_train : array-like
        Features per il training

    Returns
    -------
    array
        Importanza delle feature
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), features.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    return importances

def analisi_supervisionata():
    """
    Esegue un'analisi supervisionata sul dataset degli studenti usando Random Forest e Logistic Regression.

    Carica e pre-processa i dati, crea una variabile target binaria per il rendimento
    (alto/basso rispetto alla media globale dei voti), addestra un classificatore Random Forest
    e uno Logistic Regression, valuta le performance e visualizza l'importanza delle feature.

    Returns
    -------
    dict
        Un dizionario con le metriche di valutazione per entrambi i modelli:
        - 'rf_confusion_matrix': matrice di confusione Random Forest
        - 'rf_classification_report': report di classificazione Random Forest (stringa)
        - 'rf_accuracy_score': accuratezza Random Forest (float)
        - 'rf_feature_importances': importanza delle feature Random Forest (array)
        - 'rf_feature_names': nomi delle feature (array)
        - 'lr_confusion_matrix': matrice di confusione Logistic Regression
        - 'lr_classification_report': report di classificazione Logistic Regression (stringa)
        - 'lr_accuracy_score': accuratezza Logistic Regression (float)
    """
    # Prepara i dati
    X_train, X_val, X_test, y_train, y_val, y_test, features = prepara_dati()

    # Addestra i modelli
    rf, lr = addestra_modelli(X_train, y_train)

    # Valuta i modelli
    cm_rf, cr_rf, acc_rf, acc_val_rf = valuta_modello(rf, X_val, y_val, X_test, y_test)
    cm_lr, cr_lr, acc_lr, acc_val_lr = valuta_modello(lr, X_val, y_val, X_test, y_test)

    # Visualizza i risultati
    visualizza_confusion_matrix(cm_rf, "Confusion Matrix - Random Forest", plt.cm.Blues)
    visualizza_confusion_matrix(cm_lr, "Confusion Matrix - Logistic Regression", plt.cm.Oranges)
    importances = visualizza_importanza_feature(rf, features, X_train)

    results = {
        'rf_confusion_matrix': cm_rf,
        'rf_classification_report': cr_rf,
        'rf_accuracy_score': acc_rf,
        'rf_val_accuracy_score': acc_val_rf,
        'rf_feature_importances': importances,
        'rf_feature_names': features.columns.values,
        'lr_confusion_matrix': cm_lr,
        'lr_classification_report': cr_lr,
        'lr_accuracy_score': acc_lr,
        'lr_val_accuracy_score': acc_val_lr
    }
    print("Analisi completata con successo!")
    print(f"Accuratezza Random Forest: {results['rf_accuracy_score']:.2f}")
    print(f"Accuratezza Logistic Regression: {results['lr_accuracy_score']:.2f}")
    print("Matrice di confusione Random Forest:")
    print(results['rf_confusion_matrix'])
    print("Report di classificazione Random Forest:")
    print(results['rf_classification_report'])
    print("Matrice di confusione Logistic Regression:")
    print(results['lr_confusion_matrix'])
    print("Report di classificazione Logistic Regression:")
    print(results['lr_classification_report'])

if __name__ == "__main__":
    analisi_supervisionata()
