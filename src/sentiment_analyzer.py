import pandas as pd
import os
from transformers import pipeline


INPUT_FILE = "data/AAPL_news.csv"
OUTPUT_FILE = "data/AAPL_sentiment.csv"

def analyze_sentiment():
    print("--- Inicjalizacja modelu FinBERT ---")
    
    classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Błąd: Nie znaleziono pliku {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Wczytano {len(df)} nagłówków wiadomości.")

        
    results = []
    scores = []
    

    for title in df['title']:
        try:
            prediction = classifier(title)[0]
            
            label = prediction['label'] 
            score = prediction['score'] 
            
            results.append(label)
            scores.append(score)
            
        except Exception as e:
            print(f"Błąd przy analizie tekstu: '{title}'. Błąd: {e}")
            results.append("neutral")
            scores.append(0.0)

    df['sentiment_label'] = results
    df['sentiment_score'] = scores

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSukces! Wyniki zapisano w: {OUTPUT_FILE}")
    
    print("\nPrzykładowe wyniki:")
    print(df[['title', 'sentiment_label', 'sentiment_score']].head())

if __name__ == "__main__":
    analyze_sentiment()