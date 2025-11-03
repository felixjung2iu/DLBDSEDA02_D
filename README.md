# DLBDSEDA02_D Projekt Data Analysis


## Projektstruktur

DLBDSEDA02_D/
├─ data/
│  ├─ raw/                        # Ursprungsdaten (https://huggingface.co/datasets/mox/german_politicians_twitter_sentiment)
│  └─ processed/                  # Zwischenstände
├─ results/                       # Ergebnisdateien 
├─ src/                           # Python-Skripte für jede Verarbeitungsstufe
│   ├─ load_raw_data.py
│   ├─ preprocess.py
│   ├─ entity_analysis.py
│   ├─ topic_modeling.py
│   └─ requirements.txt
└─ README.md                   


## Ablauf und Ausführung

1. Virtuelle Umgebung aktivieren:
.\.venv\Scripts\Activate.ps1

2. Abhängigkeiten installieren:
pip install -r requirements.txt

3. Skripte in der richtigen Reihenfolge ausführen:
python src/load_raw_data.py
python src/preprocess.py
python src/entity_analysis.py
python src/topic_modeling.py



