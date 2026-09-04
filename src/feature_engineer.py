import pandas as pd
import numpy as np
import os

SENTIMENT_FILE = "data/AAPL_sentiment.csv"
PRICES_FILE = "data/AAPL_prices.csv"
OUTPUT_FILE = "data/final_dataset.csv"

def process_data():
    if not os.path.exists(SENTIMENT_FILE) or not os.path.exists(PRICES_FILE):
        print("Błąd: Brakuje plików wejściowych.")
        return

    df_news = pd.read_csv(SENTIMENT_FILE)
    df_prices = pd.read_csv(PRICES_FILE)

    first_col = df_prices.columns[0]
    if first_col == 'Price' or first_col != 'Date': 
        df_prices.rename(columns={first_col: 'Date'}, inplace=True)

    df_prices['Date'] = pd.to_datetime(df_prices['Date'], utc=True, errors='coerce')
    df_prices = df_prices.dropna(subset=['Date'])
    
    df_prices = df_prices.sort_values('Date')

    cols_to_fix = ['Close', 'Open', 'High', 'Low', 'Volume']
    for col in cols_to_fix:
        if col in df_prices.columns:
            df_prices[col] = pd.to_numeric(df_prices[col], errors='coerce')

    if 'date' in df_news.columns:
        df_news.rename(columns={'date': 'Date'}, inplace=True)
    
    
    df_news['Date'] = pd.to_datetime(df_news['Date'], utc=True)
    df_news['Date'] = df_news['Date'].dt.normalize()

    df_news = df_news.sort_values('Date')

    print(f"Dane wczytane. Ceny: {len(df_prices)}, Newsy: {len(df_news)}")
    # ---------------------------------------------

    def calculate_weighted_score(row):
        if row['sentiment_label'] == 'positive':
            return row['sentiment_score']
        elif row['sentiment_label'] == 'negative':
            return -row['sentiment_score']
        else:
            return 0.0

    df_news['final_score'] = df_news.apply(calculate_weighted_score, axis=1)


    
    daily_sentiment = df_news.groupby('Date')['final_score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Avg_Sentiment']
    print(f"Unikalnych dni z newsami: {len(daily_sentiment)}")
    print(f"Zakres newsów: {daily_sentiment['Date'].min()} do {daily_sentiment['Date'].max()}")
    print(daily_sentiment.head(10))


    # Zamiast szukać idealnego dopasowania daty, szukamy newsów z przeszłości (np. z weekendu), które są najbliżej danej sesji giełdowej.
    # direction='backward' oznacza: dla każdej ceny szukamy newsa z tej samej daty LUB wcześniejszego.
    # tolerance=pd.Timedelta('3 days') oznacza: news może być maksymalnie sprzed 3 dni (np. piątek-niedziela dla poniedziałku).
    
    df_final = pd.merge_asof(
        df_prices, 
        daily_sentiment, 
        on='Date', 
        direction='backward', 
        tolerance=pd.Timedelta('3 days')
    )


    df_final['Avg_Sentiment'] = df_final['Avg_Sentiment'].fillna(0.0)

    # Jeśli nadal mamy same zera (bo np. newsy są z przyszłości albo bardzo odległe),
    # włączamy tryb DEMO dla celów inżynierskich, żebyś mógł wytrenować model.
    non_zero_count = len(df_final[df_final['Avg_Sentiment'] != 0])
    

    close_col = 'Close' 
    if 'Adj Close' in df_final.columns:
        close_col = 'Adj Close'
            

    df_final['Tomorrow_Close'] = df_final[close_col].shift(-1)
    df_final['Target'] = (df_final['Tomorrow_Close'] > df_final[close_col]).astype(int)

    df_final = df_final.dropna(subset=['Tomorrow_Close'])


    cols_to_keep = ['Date', close_col, 'Volume', 'Avg_Sentiment', 'Target']
    existing_cols = [c for c in cols_to_keep if c in df_final.columns]
    
    df_final = df_final[existing_cols]

    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Zbiór danych zapisano w: {OUTPUT_FILE}")
    
    non_zeros = len(df_final[df_final['Avg_Sentiment'] != 0])
    print(f"Liczba wierszy z aktywnym sentymentem: {non_zeros}")
    print(df_final.tail())

if __name__ == "__main__":
    process_data()