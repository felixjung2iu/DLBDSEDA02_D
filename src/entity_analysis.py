import pandas as pd
from collections import Counter
import os
import ast 

# Definiert die relativen Pfade
PROCESSED_FINAL_PATH = 'data/processed/processed_for_analysis.csv'
RESULTS_PATH = 'results/entity_analysis.txt'

 
# Führt Entity-Analyse durch, identifiziert die Top-Hashtags und die Top-5 der am meisten genannten User 
def run_entity_analysis():
    print("--- 3. Entity-Analyse starten ---")
    
    if not os.path.exists(PROCESSED_FINAL_PATH):
        print(f"FEHLER: Eingabedatei '{PROCESSED_FINAL_PATH}' nicht gefunden.")
        return
        
    df = pd.read_csv(PROCESSED_FINAL_PATH)
    results = []
    
    # Analyse der häufigsten Hashtags 
    all_hashtags = [item for sublist in df['hashtags'].apply(ast.literal_eval).tolist() for item in sublist]
    hashtag_counts = Counter(all_hashtags)
    top_hashtags = hashtag_counts.most_common(10)
    
    results.append("Top 10 Häufigste Hashtags:")
    for hashtag, count in top_hashtags:
        results.append(f"  {hashtag}: {count}")

    # Analyse der aktivsten/meistgenannten User 
    all_users = [item for sublist in df['users_mentioned'].apply(ast.literal_eval).tolist() for item in sublist]
    user_counts = Counter(all_users)
    top_users = user_counts.most_common(5)
    
    results.append("\nTop 5 Aktivste/Meistgenannte User:")
    for user, count in top_users:
        results.append(f"  @{user}: {count}")

    # Speichern der Ausgabe
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
        
    print(f"Entity-Analyse abgeschlossen. Ergebnisse gespeichert in {RESULTS_PATH}.")

if __name__ == "__main__":
    run_entity_analysis()