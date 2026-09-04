import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_FILE = "data/final_dataset.csv"

def train_model():
    print("--- Trenowanie Modelu XGBoost ---")

    if not os.path.exists(DATA_FILE):
        print("Błąd: Brak pliku z danymi. Uruchom najpierw feature_engineer.py!")
        return

    df = pd.read_csv(DATA_FILE)
    
    df = df.sort_values('Date')
    print(f"Wczytano {len(df)} wierszy danych.")


    features = ['Avg_Sentiment', 'Volume'] 
        
    if 'Close' in df.columns:
        features.append('Close')
    elif 'Adj Close' in df.columns:
        features.append('Adj Close')

    X = df[features]
    y = df['Target']

    print(f"Cechy użyte do treningu: {features}")

    split_index = int(len(df) * 0.8) 
    
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    print(f"Trening na {len(X_train)} próbkach, Test na {len(X_test)} próbkach.")

    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=5, 
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("Model wytrenowany.")


    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    print(f"\n--- WYNIKI ---")
    print(f"Dokładność (Accuracy): {acc:.2%}")
    print("\nRaport klasyfikacji:")
    print(classification_report(y_test, predictions))


    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Spadek', 'Wzrost'], yticklabels=['Spadek', 'Wzrost'])
    plt.title('Macierz Pomyłek (Confusion Matrix)')
    plt.ylabel('Prawda')
    plt.xlabel('Predykcja')
    

    cm_path = "data/confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"\nZapisano macierz pomyłek do: {cm_path}")
    
    plt.show()

    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "xgboost_model.json")
    model.save_model(model_path)
    print(f"\nZapisano model do pliku: {model_path}")

    importances = model.feature_importances_
    for name, imp in zip(features, importances):
        print(f"Ważność cechy '{name}': {imp:.4f}")

if __name__ == "__main__":
    train_model()