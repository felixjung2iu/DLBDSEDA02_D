# DLBDSEDA02_D Projekt Data Analysis
Verwendeter **[Datensatz](https://huggingface.co/datasets/mox/german_politicians_twitter_sentiment)**

## Projektstruktur
```text
DLBDSEDA02_D/
├─ data/
│  ├─ raw/                        # Ursprungsdaten 
│  └─ processed/                  # Zwischenstände
├─ results/                       # Ergebnisdateien 
├─ src/                           # Python-Skripte für jede Verarbeitungsstufe
│   ├─ load_raw_data.py
│   ├─ preprocess.py
│   ├─ entity_analysis.py
│   ├─ topic_modeling.py
│   └─ requirements.txt
└─ README.md                   
```


## Ablauf und Ausführung

1. Virtuelle Umgebung aktivieren:
```text
.\.venv\Scripts\Activate.ps1
```

2. Abhängigkeiten installieren:
```text
pip install -r requirements.txt
```
3. Skripte in folgender Reihenfolge ausführen:
```text
python src/load_raw_data.py
python src/preprocess.py
python src/entity_analysis.py
python src/topic_modeling.py
```



