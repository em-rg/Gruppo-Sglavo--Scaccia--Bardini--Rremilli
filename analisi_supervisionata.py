from preprocessing import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

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

    # Inizializza e addestra il classificatore Random Forest
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Inizializza e addestra il classificatore binario 
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Valuta le performance Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cr_rf = classification_report(y_test, y_pred_rf)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # Valuta le performance Logistic Regression
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cr_lr = classification_report(y_test, y_pred_lr)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    # Visualizza la confusion matrix Random Forest
    plt.figure()
    plt.title("Confusion Matrix - Random Forest")
    plt.imshow(cm_rf, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Basso', 'Alto'])
    plt.yticks(tick_marks, ['Basso', 'Alto'])
    thresh = cm_rf.max() / 2.
    for i in range(cm_rf.shape[0]):
        for j in range(cm_rf.shape[1]):
            plt.text(j, i, format(cm_rf[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_rf[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # Visualizza la confusion matrix Logistic Regression
    plt.figure()
    plt.title("Confusion Matrix - Logistic Regression")
    plt.imshow(cm_lr, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.colorbar()
    plt.xticks(tick_marks, ['Basso', 'Alto'])
    plt.yticks(tick_marks, ['Basso', 'Alto'])
    thresh = cm_lr.max() / 2.
    for i in range(cm_lr.shape[0]):
        for j in range(cm_lr.shape[1]):
            plt.text(j, i, format(cm_lr[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_lr[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

    # Visualizza l'importanza delle feature Random Forest
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), features.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

    return {
        'rf_confusion_matrix': cm_rf,
        'rf_classification_report': cr_rf,
        'rf_accuracy_score': acc_rf,
        'rf_feature_importances': importances,
        'rf_feature_names': features.columns.values,
        'lr_confusion_matrix': cm_lr,
        'lr_classification_report': cr_lr,
        'lr_accuracy_score': acc_lr
    }

if __name__ == "__main__":
    results = analisi_supervisionata()
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