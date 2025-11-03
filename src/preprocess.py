import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import os

# Definiert die relativen Pfade
PROCESSED_LOADED_PATH = 'data/processed/loaded_tweets_for_analysis.csv'
PROCESSED_FINAL_PATH = 'data/processed/processed_for_analysis.csv'

# Stoppwörter herunterladen, falls nötig
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Kombinierte und manuell erweiterte Stoppwortliste
GERMAN_STOPWORDS = set(stopwords.words('german'))
ENGLISH_STOPWORDS = set(stopwords.words('english'))
COMBINED_STOPWORDS = GERMAN_STOPWORDS.union(ENGLISH_STOPWORDS)

IRRELEVANT_TWITTER_TERMS = {'de', 'com', 'thread', 'anzeigen', 'aug', 'uhr', 'live', 'mal', 'angezeigt', 
                           'heute', 'mehr', 'geht', 'gibt', 'müssen', 'immer', 'schon', 'dafür', 
                           'ja', 'lieber', 'wer', 'innen', 'glückwunsch', 'herzlichen', 'neuen', 
                           'dabei', 'gut', 'gute', 'guten', 'dank', 'danke', 'sagt', 'tun', 
                           'tag', 'woche', 'monate', 'märz', 'juni', 'zeit', 'beim', 'co', 
                           'gerade', 'vielen', 'freue', 'letzten', 'jahren', 'viele', 'jan',
                           'morgen', 'ganz', 'statt', 'rund', 'darf', 'kommt', 'neue', 
                           'leben', 'ab', 'ab', 'neue', 'brauchen', 'zeigt', 'seit', 'unserer'} 
COMBINED_STOPWORDS.update(IRRELEVANT_TWITTER_TERMS) 

# Bereinigt den Text: entfernt URLs, Mentions und Sonderzeichen
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^A-Za-zäöüÄÖÜß#\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

# Tokenisiert den Text und entfernt kombinierte Stoppwörter sowie Hashtags
# Filtert Stoppwörter und alle Tokens, die mit # beginnen
def remove_stopwords_and_tokenize(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in COMBINED_STOPWORDS and not word.startswith('#')]
    return " ".join(filtered_tokens)

# Führt die komplette Vorverarbeitung durch und speichert das Ergebnis
def preprocess_data():
    print("--- 2. Datenvorverarbeitung starten ---")
    
    if not os.path.exists(PROCESSED_LOADED_PATH):
        print(f"FEHLER: Eingabedatei '{PROCESSED_LOADED_PATH}' nicht gefunden.")
        return None
    
    df = pd.read_csv(PROCESSED_LOADED_PATH)

    # 1. Rohdaten bereinigen und Entitäten extrahieren
    df['cleaned_text'] = df['tweet_text'].apply(clean_text)
    df['hashtags'] = df['tweet_text'].apply(lambda x: re.findall(r'#\w+', str(x).lower()))
    
    # 2. Extraktion der User
    df['users_mentioned'] = df['tweet_text'].apply(lambda x: re.findall(r'@([A-Za-z0-9_]+)', str(x).lower()))
    
    # 3. Text für Topic Modeling finalisieren 
    df['processed_text'] = df['cleaned_text'].apply(remove_stopwords_and_tokenize)
    
    # Speichern der Ausgabe 
    os.makedirs(os.path.dirname(PROCESSED_FINAL_PATH), exist_ok=True)
    df.to_csv(PROCESSED_FINAL_PATH, index=False)
    print(f"Vorverarbeitete Daten gespeichert in {PROCESSED_FINAL_PATH}.")
    
    return df

if __name__ == "__main__":
    preprocess_data()