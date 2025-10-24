import pandas as pd
import os
import random

# Definiert die  Pfade zum Ein- und Ausgabeort
RAW_DATA_PATH = 'data/raw/all_annotated_data.csv'
PROCESSED_LOADED_PATH = 'data/processed/loaded_tweets_for_analysis.csv' 

def load_data():
    """
    Lädt die lokale CSV-Datei, benennt die Textspalte um und speichert das bereinigte Rohformat.
    """
    print("--- 1. Daten laden und vorbereiten ---")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"FEHLER: Eingabedatei '{RAW_DATA_PATH}' nicht gefunden.")
        return None
        
    # Laden mit Semikolon als Trennzeichen 
    df = pd.read_csv(RAW_DATA_PATH, delimiter=';')
    
    # Überprüfen und Umbenennen der Text-Spalte 
    if 'Embedded_text' not in df.columns:
        print("FEHLER: Spalte 'Embedded_text' nicht gefunden. Prüfen Sie die Spaltennamen.")
        return None

    # Umbenennen der Spalte 
    df = df.rename(columns={'Embedded_text': 'tweet_text'})

    # Speichern der Ausgabe für das nächste Skript
    os.makedirs(os.path.dirname(PROCESSED_LOADED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_LOADED_PATH, index=False)
    
    print(f"Dataset mit {len(df)} Einträgen geladen und gespeichert in {PROCESSED_LOADED_PATH}.")
    return df

if __name__ == "__main__":
    load_data()