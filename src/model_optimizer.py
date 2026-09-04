import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score
import os

DATA_FILE = "data/final_dataset.csv"

def optimize_model():
    print("--- Rozpoczynam Optymalizację Modelu (Grid Search) ---")
    
    if not os.path.exists(DATA_FILE):
        print("Błąd: Brak pliku danych.")
        return

    df = pd.read_csv(DATA_FILE)
    df = df.sort_values('Date')
    
    features = ['Avg_Sentiment', 'Volume']
    if 'Close' in df.columns: features.append('Close')
    elif 'Adj Close' in df.columns: features.append('Adj Close')
    
    X = df[features]
    y = df['Target']
    
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    print(f"Dane treningowe: {len(X_train)} wierszy.")

    param_grid = {
        'n_estimators': [50, 100, 200],      
        'max_depth': [3, 5, 7],              
        'learning_rate': [0.01, 0.05, 0.1],  
        'subsample': [0.8, 1.0]              
    }
    
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    print("Testowanie kombinacji parametrów")
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='accuracy',
        cv=tscv,
        verbose=1,
        n_jobs=-1 
    )
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print("\n--- ZNALEZIONO NAJLEPSZE PARAMETRY ---")
    print(best_params)
    
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    
    print(f"\nDokładność na zbiorze testowym po optymalizacji: {acc:.2%}")

if __name__ == "__main__":
    optimize_model()