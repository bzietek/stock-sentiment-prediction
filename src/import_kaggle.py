import pandas as pd
import os

INPUT_FILE = "data/kaggle_news.csv" 
OUTPUT_FILE = "data/AAPL_news.csv"  
TICKER = "AAPL" 

def convert_kaggle_data():
    print("--- Konwerter Danych z Kaggle (Analyst Ratings Processed) ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Błąd: Nie znaleziono pliku {INPUT_FILE}.")
        return

    print("Wczytywanie pliku (może to zająć chwilę, bo plik jest duży)...")
    
    try:

        df = pd.read_csv(INPUT_FILE)
        print(f"Wczytano: {len(df)} wierszy.")
        print("Przykładowe dane:")
        print(df.head(2))
        

        col_mapping = {
            'title': 'title',
            'date': 'date',
            'stock': 'ticker'
        }
        
        
        df.rename(columns=str.lower, inplace=True) 
        df.rename(columns=col_mapping, inplace=True)
        
        
        if 'ticker' in df.columns:
            print(f"Filtruję dane tylko dla: {TICKER}...")
            df = df[df['ticker'] == TICKER]
        else:
            print("BŁĄD: Nie znaleziono kolumny 'stock' lub 'ticker'.")
            print("Dostępne kolumny:", df.columns)
            return

        
        print("Konwersja dat...")
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        
        df = df.sort_values('date')
        
        final_df = df[['title', 'date']].copy()
        
        final_df['publisher'] = 'AnalystRating'
        final_df['link'] = ''
        final_df['published'] = final_df['date']

        print(f"Znaleziono {len(final_df)} newsów dla {TICKER}.")
        
        if len(final_df) == 0:
            print("Ostrzeżenie: 0 newsów po przefiltrowaniu! Sprawdź czy TICKER jest poprawny (np. czy w pliku jest 'AAPL' czy 'Apple').")
        else:
            final_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Sukces! Zapisano w: {OUTPUT_FILE}")
            print("Przykładowy wiersz:")
            print(final_df.head(1))
        
    except Exception as e:
        print(f"Wystąpił błąd krytyczny: {e}")

if __name__ == "__main__":
    convert_kaggle_data()