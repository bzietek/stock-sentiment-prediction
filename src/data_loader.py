import yfinance as yf
import pandas as pd
import os
from datetime import datetime

TICKER = "AAPL" #Apple
#(Marzec-Czerwiec 2020)
START_DATE = "2020-03-01"
END_DATE = "2021-06-30"
DATA_DIR = "data"

def ensure_data_dir():
    """Tworzymy folder na dane, jeśli nie istnieje"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Utworzono folder: {DATA_DIR}")

def download_stock_prices(ticker, start, end):
    """Pobiera historyczne ceny akcji (OHLCV)."""
    print(f"--- Pobieranie cen akcji dla {ticker} ---")
    
    df = yf.download(ticker, start=start, end=end)
    
    if len(df) > 0:
        filename = f"{DATA_DIR}/{ticker}_prices.csv"
        df.to_csv(filename)
        print(f"Pobrano {len(df)} wierszy.")
        print(f"Zapisano w: {filename}")
    else:
        print("Błąd: Nie pobrano żadnych danych. Sprawdź ticker lub daty.")

def download_recent_news(ticker):
    """
    Pobiera najnowsze wiadomości.
    """
    print(f"\n--- Pobieranie newsów dla {ticker} ---")
    
    stock = yf.Ticker(ticker)
    news_list = stock.news
    
    if news_list:
        news_data = []
        for item in news_list:
            content = item.get('content', {})
            
            title = content.get('title', 'Brak tytułu')
            
            provider = content.get('provider', {})
            publisher = provider.get('displayName', 'Nieznany')
            
            click_info = content.get('clickThroughUrl', {})
            link = click_info.get('url', '')
            
            pub_date = content.get('pubDate', '')

            news_data.append({
                'title': title,
                'publisher': publisher,
                'link': link,
                'published': pub_date
            })
        
        df = pd.DataFrame(news_data)
        
        df['date'] = pd.to_datetime(df['published'], format='mixed', errors='coerce')
        
        filename = f"{DATA_DIR}/{ticker}_news.csv"
        df.to_csv(filename, index=False)
        print(f"Pobrano {len(df)} nagłówków.")
        print(f"Zapisano w: {filename}")
    else:
        print("Nie znaleziono newsów.")
        
if __name__ == "__main__":
    # główna część programu
    ensure_data_dir()
    download_stock_prices(TICKER, START_DATE, END_DATE)
    #download_recent_news(TICKER)