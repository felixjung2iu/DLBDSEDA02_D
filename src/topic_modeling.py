import pandas as pd
import numpy as np
import logging
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# Logging konfigurieren, um Gensim-Meldungen zu unterdrücken
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)

# Definiert die relativen Pfade
PROCESSED_FINAL_PATH = 'data/processed/processed_for_analysis.csv'

# Lädt die bereinigten Daten
def load_preprocessed_data(file_path=PROCESSED_FINAL_PATH):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Fehler: Datei {file_path} nicht gefunden.")
        return None
    
    # Konvertiert den Text-String ('wort1 wort2...') in eine Liste von Wörtern ['wort1', 'wort2', ...]
    df = df.dropna(subset=['processed_text'])
    data_tokens = df['processed_text'].apply(lambda x: str(x).split()).tolist()
    
    # Entferne leere Listen
    data_tokens = [tokens for tokens in data_tokens if tokens]
    
    if not data_tokens:
        print("Fehler: Keine Tokens nach der Vorbereitung")
        return None

    # Erstellen des Gensim Dictionaries
    dictionary = corpora.Dictionary(data_tokens)
    # Filtern extrem häufiger/seltener Wörter zur Verbesserung der Themenqualität
    dictionary.filter_extremes(no_below=5, no_above=0.5) 
    corpus = [dictionary.doc2bow(text) for text in data_tokens]
    
    # Filtern der data_tokens, um nur Tokens zu behalten, die im Dictionary verblieben sind
    data_tokens_filtered = [[token for token in doc if token in dictionary.token2id] for doc in data_tokens]
    
    return data_tokens_filtered, dictionary, corpus

# Berechnet den Topic Coherence Score (c_v) 
def compute_coherence_score(lda_model, data_tokens, dictionary, coherence_measure='c_v'):
    coherence_model = CoherenceModel(
        model=lda_model, 
        texts=data_tokens, 
        dictionary=dictionary, 
        coherence=coherence_measure
    )
    return coherence_model.get_coherence()

# Führt LDA durch, gibt die Themen aus und berechnet den Coherence Score
def run_topic_modeling(data_tokens, dictionary, corpus, num_topics=5):
    print(f"Starte LDA-Modellierung mit K={num_topics} Themen...")
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100, 
        chunksize=1000,
        passes=10,
        alpha='auto',
        per_word_topics=True
    )
    
    #Coherence Score berechnen
    coherence_score = compute_coherence_score(lda_model, data_tokens, dictionary, coherence_measure='c_v')
    print(f"\n--- Ergebnisse (K={num_topics}) ---")
    print(f"Coherence Score (c_v): {coherence_score:.4f}")

    #Ergebnisse in Datei schreiben
    output_file = 'results/topic_modeling.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("--- LDA Topic Modeling Results ---\n")
        f.write(f"Anzahl der Themen (K): {num_topics}\n")
        f.write(f"Coherence Score (c_v): {coherence_score:.4f}\n\n") 
        
        f.write("Fünf dominierende Themen (Top 10 Wörter):\n")
        
        # Ausgabe der Top-Wörter pro Thema
        for idx, topic in lda_model.print_topics(num_words=10):
            # Bereinigung der Ausgabe 
            words = ' '.join([word.split('*')[1].replace('"', '').strip() for word in topic.split('+')])
            f.write(f"Thema {idx+1}:\n")
            f.write(f"  Schlüsselwörter: {words}\n")

    print(f"Ergebnisse gespeichert in: {output_file}")
    
    return lda_model, coherence_score

if __name__ == '__main__':  
    data_tokens, dictionary, corpus = load_preprocessed_data()
    if data_tokens and dictionary and corpus:
        run_topic_modeling(data_tokens, dictionary, corpus, num_topics=5)