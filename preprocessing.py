import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder



def preprocessing(filepath="student-por.csv"):
    """
    Preprocessa il dataset degli studenti.

    Carica il dataset, mostra informazioni e statistiche, gestisce i valori nulli,
    normalizza le colonne numeriche e codifica le variabili categoriche.

    Parameters
    ----------
    filepath : str, optional
        Percorso del file CSV da caricare (default "student-por.csv").

    Returns
    -------
    pandas.DataFrame
        Il DataFrame preprocessato, pronto per l'analisi.
    """

    # Carica il dataset
    df = pd.read_csv(filepath)

    print(df.head(5))
    print(f"Data Shape: Rows = {df.shape[0]}, Columns = {df.shape[1]}")

    # Informazioni generali
    df.info()

    # Statistiche descrittive
    print(df.describe().T)

    # Valori nulli
    null_counts = df.isnull().sum()
    null_counts[null_counts > 0])

    #print("valori mancanti:")
    #print(null_counts[null_counts > 0])


    # Gestione dei valori nulli 
    df = df.dropna()

    # Seleziona le colonne numeriche e normalizza
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Crea un oggetto LabelEncoder
    label_encoder = LabelEncoder()

    # Lista delle colonne da codificare
    Columns = ["school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian","schoolsup","famsup","paid","activities",
                    "nursery","higher","internet","romantic"]

    for i in range(len(Columns)):
        # Recupera i valori unici nella colonna
        Country_keys = df[Columns[i]]
        Country_keys = Country_keys.tolist()
        
        # Esegui la codifica delle etichette
        Country_values = label_encoder.fit_transform(df[Columns[i]])
        Country_values = Country_values.tolist()
        
        # Aggiorna il DataFrame con i valori codificati
        df[Columns[i]] = label_encoder.fit_transform(df[Columns[i]])
        
        # Crea un dizionario che mappa i valori originali a quelli codificati
        Country_dict = dict(zip(Country_keys, Country_values))
        # Stampa il dizionario



    return df

if __name__ == "__main__":
    preprocessing()

