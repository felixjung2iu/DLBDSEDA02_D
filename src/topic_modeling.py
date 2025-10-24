import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import os
import re

# Definiert die relativen Pfade
PROCESSED_FINAL_PATH = 'data/processed/processed_for_analysis.csv'
RESULTS_PATH = 'results/topic_modeling.txt'

# Ruft die Word-Thema-Wahrscheinlichkeiten ab
def format_topic_output_simple(lda_model, topic_id, num_words=10):
    """Extrahiert Topic-Keywords und formatiert sie im geforderten Stil."""
    topic_output = lda_model.show_topic(topic_id, topn=num_words)
    formatted_lines = []
    for word, prob in topic_output: 
        # Formatiert die Ausgabe: Keyword : Wahrscheinlichkeit (auf 4 Dezimalstellen)
        formatted_lines.append(f"{word} : {prob:.4f}")
        
    return "\n".join(formatted_lines)

# Führt LDA durch, extrahiert die Top 5 Themen und speichert die formatierte Ausgabe
def run_topic_modeling(num_topics=5):
    print("--- 4. Topic Modeling starten ---")
    
    if not os.path.exists(PROCESSED_FINAL_PATH):
        print(f"FEHLER: Eingabedatei '{PROCESSED_FINAL_PATH}' nicht gefunden.")
        return

    df = pd.read_csv(PROCESSED_FINAL_PATH)
    
    # Vorbereitung des Textes
    texts = [re.split(r'\s+', str(doc)) for doc in df['processed_text'].tolist() if pd.notna(doc) and len(str(doc).strip()) > 0]
    
    if not texts:
        print("FEHLER: Keine sauberen Texte für das Topic Modeling gefunden.")
        return

    # Erstellen von Wörterbuch und Corpus
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5) 
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # LDA-Modell trainieren (basierend auf der Aufgabenstellung: 5 Themen)
    lda_model = LdaModel(corpus, 
                         num_topics=num_topics, 
                         id2word=dictionary, 
                         passes=10, 
                         random_state=42)

    # Extrahieren und Speichern der 5 häufigsten Themen
    results = ["Ergebnisse der Latent Dirichlet Allocation (LDA)"]
    
    # Erklärung der Ausgabe 
    results.append("\nDie Zahlen geben die Wahrscheinlichkeit an.")
    results.append("Format: [Keyword : Wahrscheinlichkeit]")

    for idx in range(num_topics):
        results.append(f"\nThema {idx+1}")
        
        # Formatierte Liste hinzufügen
        formatted_list = format_topic_output_simple(lda_model, idx)
        results.append(formatted_list)

    # Speichern der Ergebnisse
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
        
    print(f"Topic Modeling abgeschlossen. Ergebnisse gespeichert in {RESULTS_PATH}.")

if __name__ == "__main__":
    run_topic_modeling(num_topics=5)