import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os

DATA_FILE = "data/final_dataset.csv"

def visualize_strategy():
    print("--- Generowanie Wykresu Symulacji Inwestycyjnej ---")
    
    if not os.path.exists(DATA_FILE):
        print("Błąd: Brak pliku danych.")
        return

    df = pd.read_csv(DATA_FILE)
    df = df.sort_values('Date')
    
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:].copy() 
    
    features = ['Avg_Sentiment', 'Volume']
    if 'Close' in df.columns: features.append('Close')
    elif 'Adj Close' in df.columns: features.append('Adj Close')
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(train_df[features], train_df['Target'])
    
    test_df['Prediction'] = model.predict(test_df[features])
    
    initial_capital = 10000.0 
    
    col_close = 'Close' if 'Close' in test_df.columns else 'Adj Close'
    test_df['Pct_Change'] = test_df[col_close].pct_change().fillna(0)
    
    test_df['Buy_Hold_Value'] = initial_capital * (1 + test_df['Pct_Change']).cumprod()
    
    test_df['Signal'] = test_df['Prediction'].shift(1).fillna(0)
    
    test_df['Strategy_Change'] = test_df['Pct_Change'] * test_df['Signal']
    
    test_df['AI_Strategy_Value'] = initial_capital * (1 + test_df['Strategy_Change']).cumprod()
    
    plt.figure(figsize=(12, 6))
    
    dates = pd.to_datetime(test_df['Date'])
    
    plt.plot(dates, test_df['Buy_Hold_Value'], label='Strategia "Kup i Trzymaj"', color='gray', linestyle='--', alpha=0.7)
    plt.plot(dates, test_df['AI_Strategy_Value'], label='Strategia AI (Twój Model)', color='green', linewidth=2)
    
    plt.title(f'Symulacja Portfela: AI vs Rynek (Accuracy: {model.score(test_df[features], test_df["Target"]):.2%})')
    plt.xlabel('Data')
    plt.ylabel('Wartość Portfela (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_file = "data/strategy_chart.png"
    plt.savefig(plot_file)
    print(f"Wykres zapisano w: {plot_file}")
    
    plt.show()

if __name__ == "__main__":
    visualize_strategy()