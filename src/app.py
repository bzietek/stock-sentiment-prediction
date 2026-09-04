import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="StockSentiment AI | Prognozowanie Giełdowe", 
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS dla lepszego wyglądu
st.markdown("""
<style>
    /* Większe nagłówki */
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Karty metryk */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card-red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .metric-card-blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Kafelki informacyjne */
    .info-tile {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    /* Tabela decyzji */
    .decision-buy { color: #28a745; font-weight: bold; }
    .decision-sell { color: #dc3545; font-weight: bold; }
    
    /* Sidebar */
    .css-1d391kg { padding-top: 1rem; }
    
    /* Ukrycie stopki Streamlit */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

DATA_FILE = "data/final_dataset.csv"
MODEL_FILE = "models/xgboost_model.json"
NEWS_FILE = "data/AAPL_sentiment.csv"  

@st.cache_data
def load_data():
    """Wczytuje dane z pliku CSV."""
    if not os.path.exists(DATA_FILE):
        return None
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date')
    return df

@st.cache_data
def load_news():
    """Wczytuje dane o newsach z sentymentem."""
    if not os.path.exists(NEWS_FILE):
        return None
    df = pd.read_csv(NEWS_FILE)
    if 'date' in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    return df

@st.cache_resource
def load_model():
    """Wczytuje wytrenowany model XGBoost."""
    if not os.path.exists(MODEL_FILE):
        return None
    model = XGBClassifier()
    model.load_model(MODEL_FILE)
    return model

def get_features(df):
    """Zwraca listę cech używanych przez model."""
    features = ['Avg_Sentiment', 'Volume']
    if 'Close' in df.columns:
        features.append('Close')
    elif 'Adj Close' in df.columns:
        features.append('Adj Close')
    return features

def calculate_portfolio_value(df, model, features, initial_capital=10000):
    """Oblicza wartość portfela dla strategii AI vs Buy&Hold."""
    df = df.copy()
    col_close = 'Close' if 'Close' in df.columns else 'Adj Close'
    
    df['Prediction'] = model.predict(df[features])
    
    df['Pct_Change'] = df[col_close].pct_change().fillna(0)
    
    df['Buy_Hold'] = initial_capital * (1 + df['Pct_Change']).cumprod()
    
    df['Signal'] = df['Prediction'].shift(1).fillna(0)
    df['Strategy_Change'] = df['Pct_Change'] * df['Signal']
    df['AI_Portfolio'] = initial_capital * (1 + df['Strategy_Change']).cumprod()
    
    return df



st.markdown('<h1 class="main-header">📈 StockSentiment AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Inteligentny system prognozowania ruchów giełdowych oparty na analizie sentymentu wiadomości</p>', unsafe_allow_html=True)

df = load_data()
model = load_model()
news_df = load_news()

if df is None or model is None:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f0f2f6; border-radius: 20px;">
            <h2>🚀 Witaj w StockSentiment AI!</h2>
            <p style="font-size: 1.1rem; color: #666;">
                Aby rozpocząć analizę, potrzebujemy danych historycznych.<br>
                Kliknij poniżej, aby wygenerować przykładowe dane demonstracyjne.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔄 Wygeneruj Dane Demonstracyjne", use_container_width=True, type="primary"):
            with st.spinner("Generowanie danych... Proszę czekać."):
                exec(open("synthetic_test.py").read())
                import sys
                sys.argv = ['model_trainer.py']  # Reset argumentów
                
                train_df = pd.read_csv(DATA_FILE)
                train_df = train_df.sort_values('Date')
                features = ['Avg_Sentiment', 'Volume', 'Close']
                X = train_df[features]
                y = train_df['Target']
                split = int(len(train_df) * 0.8)
                
                model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
                model.fit(X.iloc[:split], y.iloc[:split])
                
                os.makedirs("models", exist_ok=True)
                model.save_model(MODEL_FILE)
                
            st.success("✅ Dane wygenerowane! Odśwież stronę (F5), aby zobaczyć dashboard.")
            st.balloons()
        
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; background: #e8f4f8; border-radius: 10px;">
            <p style="margin: 0; font-size: 0.9rem;">
                💡 <strong>Wskazówka:</strong> Dane demonstracyjne to syntetyczne dane giełdowe 
                z celowo wbudowaną zależnością sentyment → cena. Służą do prezentacji możliwości systemu.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()


st.sidebar.markdown("## 🎛️ Panel Sterowania")
st.sidebar.markdown("---")

# Wybór trybu
mode = st.sidebar.radio(
    "Wybierz widok:",
    ["🏠 Panel Inwestora", "⏱️ Symulacja Historyczna", "📋 Dziennik Decyzji", "🔧 Panel Techniczny"],
    index=0
)

st.sidebar.markdown("---")

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

st.sidebar.markdown("### 📅 Zakres Analizy")
date_range = st.sidebar.date_input(
    "Wybierz okres:",
    value=(max_date - timedelta(days=90), max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
    df_filtered = df[mask].copy()
else:
    df_filtered = df.tail(90).copy()

features = get_features(df)
col_close = 'Close' if 'Close' in df.columns else 'Adj Close'


if mode == "🏠 Panel Inwestora":
    

    df_portfolio = calculate_portfolio_value(df_filtered, model, features)
    

    st.markdown("### 📊 Podsumowanie Okresu")
    
    col1, col2, col3, col4 = st.columns(4)
    
    last_price = df_filtered.iloc[-1][col_close]
    first_price = df_filtered.iloc[0][col_close]
    price_change = ((last_price - first_price) / first_price) * 100
    
    with col1:
        st.metric(
            "💵 Cena AAPL", 
            f"${last_price:.2f}", 
            f"{price_change:+.2f}%"
        )
    
    ai_final = df_portfolio['AI_Portfolio'].iloc[-1]
    ai_return = ((ai_final - 10000) / 10000) * 100
    
    with col2:
        st.metric(
            "🤖 Zysk Strategii AI",
            f"${ai_final:,.0f}",
            f"{ai_return:+.2f}%"
        )
    
    bh_final = df_portfolio['Buy_Hold'].iloc[-1]
    bh_return = ((bh_final - 10000) / 10000) * 100
    
    with col3:
        st.metric(
            "📈 Zysk Buy & Hold",
            f"${bh_final:,.0f}",
            f"{bh_return:+.2f}%"
        )
    

    df_portfolio['Actual'] = (df_portfolio[col_close].shift(-1) > df_portfolio[col_close]).astype(int)
    df_portfolio['Correct'] = (df_portfolio['Prediction'] == df_portfolio['Actual']).astype(int)
    accuracy = df_portfolio['Correct'].mean() * 100
    
    with col4:
        st.metric(
            "🎯 Trafność Prognoz",
            f"{accuracy:.1f}%",
            f"{accuracy - 50:+.1f}% vs losowe"
        )
    
    st.markdown("---")
    
    st.markdown("### 💼 Symulacja Portfela Inwestycyjnego")
    st.markdown("*Kapitał początkowy: $10,000*")
    
    fig = go.Figure()
    

    fig.add_trace(go.Scatter(
        x=df_portfolio['Date'],
        y=df_portfolio['Buy_Hold'],
        name='📊 Kup i Trzymaj',
        line=dict(color='#888888', width=2, dash='dash'),
        hovertemplate='Data: %{x}<br>Wartość: $%{y:,.2f}<extra></extra>'
    ))
    

    fig.add_trace(go.Scatter(
        x=df_portfolio['Date'],
        y=df_portfolio['AI_Portfolio'],
        name='🤖 Strategia AI',
        line=dict(color='#28a745', width=3),
        fill='tonexty',
        fillcolor='rgba(40, 167, 69, 0.1)',
        hovertemplate='Data: %{x}<br>Wartość: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title='Data',
        yaxis_title='Wartość Portfela (USD)',
        hovermode='x unified',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔮 Prognoza na Następną Sesję")
    
    last_row = df_filtered.iloc[[-1]][features]
    prediction = model.predict(last_row)[0]
    last_sentiment = df_filtered.iloc[-1]['Avg_Sentiment']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.success("### 📈 Sygnał: **WZROST**")
            st.markdown("""
            Model przewiduje, że cena akcji Apple **wzrośnie** podczas następnej sesji giełdowej.
            
            **Rekomendacja:** Rozważ utrzymanie lub zwiększenie pozycji.
            """)
        else:
            st.error("### 📉 Sygnał: **SPADEK**")
            st.markdown("""
            Model przewiduje, że cena akcji Apple **spadnie** podczas następnej sesji giełdowej.
            
            **Rekomendacja:** Rozważ zabezpieczenie pozycji lub wstrzymanie zakupów.
            """)
    
    with col2:
        st.info(f"""
        **Dane wejściowe modelu:**
        - Średni sentyment: `{last_sentiment:.4f}` {'😊' if last_sentiment > 0 else '😟' if last_sentiment < 0 else '😐'}
        - Ostatnia cena: `${last_price:.2f}`
        - Wolumen: `{df_filtered.iloc[-1]['Volume']:,.0f}`
        """)


elif mode == "⏱️ Symulacja Historyczna":
    
    st.markdown("### ⏱️ Wehikuł Czasu - Symulacja Dzień po Dniu")
    st.markdown("*Cofnij się w czasie i zobacz, jak system podejmował decyzje*")
    

    available_dates = df_filtered['Date'].dt.date.unique()
    
    if len(available_dates) > 1:
        selected_date_idx = st.slider(
            "Wybierz dzień do analizy:",
            min_value=0,
            max_value=len(available_dates) - 1,
            value=len(available_dates) - 1,
            format=f"Dzień %d z {len(available_dates)}"
        )
        selected_date = available_dates[selected_date_idx]
        
        st.markdown(f"## 📅 Analiza dla dnia: **{selected_date}**")
        
        mask = df_filtered['Date'].dt.date <= selected_date
        df_until_date = df_filtered[mask].copy()
        
        if len(df_until_date) > 0:
            current_row = df_until_date.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💵 Cena zamknięcia", f"${current_row[col_close]:.2f}")
            
            with col2:
                sentiment = current_row['Avg_Sentiment']
                sentiment_emoji = '😊 Pozytywny' if sentiment > 0.05 else '😟 Negatywny' if sentiment < -0.05 else '😐 Neutralny'
                st.metric("📰 Sentyment dnia", sentiment_emoji, f"{sentiment:.4f}")
            
            with col3:
                pred = model.predict(current_row[features].values.reshape(1, -1))[0]
                decision = "📈 KUP / TRZYMAJ" if pred == 1 else "📉 SPRZEDAJ / CZEKAJ"
                st.metric("🤖 Decyzja AI", decision)
            
            st.markdown("---")
            
            if selected_date_idx < len(available_dates) - 1:
                next_date = available_dates[selected_date_idx + 1]
                next_row = df_filtered[df_filtered['Date'].dt.date == next_date].iloc[0]
                
                actual_change = next_row[col_close] - current_row[col_close]
                was_correct = (pred == 1 and actual_change > 0) or (pred == 0 and actual_change <= 0)
                
                if was_correct:
                    st.success(f"""
                    ✅ **Decyzja była TRAFNA!**
                    
                    Następnego dnia ({next_date}) cena {'wzrosła' if actual_change > 0 else 'spadła'} 
                    o **${abs(actual_change):.2f}** ({(actual_change/current_row[col_close])*100:+.2f}%)
                    """)
                else:
                    st.error(f"""
                    ❌ **Decyzja była BŁĘDNA**
                    
                    Następnego dnia ({next_date}) cena {'wzrosła' if actual_change > 0 else 'spadła'} 
                    o **${abs(actual_change):.2f}** ({(actual_change/current_row[col_close])*100:+.2f}%)
                    """)
            
            st.markdown("### 📊 Wykres do wybranego dnia")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_until_date['Date'],
                y=df_until_date[col_close],
                name='Cena AAPL',
                line=dict(color='#1E88E5', width=2),
                hovertemplate='Data: %{x}<br>Cena: $%{y:.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=[df_until_date['Date'].iloc[-1]],
                y=[current_row[col_close]],
                name='Wybrany dzień',
                mode='markers',
                marker=dict(color='red', size=15, symbol='circle'),
                hovertemplate='📍 Wybrany dzień<br>Cena: $%{y:.2f}<extra></extra>'
            ))
            
            fig.add_hline(y=current_row[col_close], line_dash='dash', line_color='red', opacity=0.5)
            
            fig.update_layout(
                xaxis_title='Data',
                yaxis_title='Cena (USD)',
                hovermode='x unified',
                margin=dict(l=0, r=0, t=30, b=0),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if news_df is not None:
                st.markdown("### 📰 Wiadomości z tego dnia")
                day_news = news_df[news_df['Date'].dt.date == selected_date]
                
                if len(day_news) > 0:
                    for _, news in day_news.head(5).iterrows():
                        sentiment_label = news.get('sentiment_label', 'neutral')
                        emoji = '🟢' if sentiment_label == 'positive' else '🔴' if sentiment_label == 'negative' else '⚪'
                        st.markdown(f"{emoji} **{news['title']}**")
                else:
                    st.info("Brak wiadomości z tego dnia w bazie danych.")
    else:
        st.warning("Wybierz szerszy zakres dat, aby korzystać z symulacji.")


elif mode == "📋 Dziennik Decyzji":
    
    st.markdown("### 📋 Dziennik Decyzji Systemu")
    st.markdown("*Historia wszystkich decyzji z oceną trafności*")
    

    with st.expander("ℹ️ Jak system podejmuje decyzje?", expanded=False):
        st.info("""
        **Jak interpretować decyzje systemu?**
        
        System działa w trzech krokach:
        
        1. **Zbieranie wiadomości** – dla każdego dnia zbierane są wszystkie nagłówki artykułów o Apple.
        
        2. **Analiza sentymentu** – model FinBERT ocenia każdy nagłówek osobno, przypisując mu wartość:
           - Pozytywny (+0.5 do +1.0) np. *"Apple notuje rekordowe zyski"*
           - Negatywny (-0.5 do -1.0) np. *"Problemy z dostawami iPhone'a"*
           - Neutralny (około 0) np. *"Apple ogłasza datę konferencji"*
        
        3. **Agregacja** – jeśli danego dnia było np. 3 artykuły (+0.8, -0.3, +0.5), system oblicza średnią:
           **Średni sentyment = (+0.8 - 0.3 + 0.5) / 3 = +0.33** (umiarkowanie pozytywny)
        
        4. **Decyzja** – model XGBoost analizuje średni sentyment, cenę i wolumen, a następnie przewiduje kierunek zmiany ceny na następny dzień.
        
        💡 **Uwaga:** Jeden bardzo negatywny artykuł może "przebić" kilka umiarkowanie pozytywnych – tak jak w prawdziwym życiu, złe wiadomości często mają większy wpływ na rynki.
        """)
    

    df_decisions = df_filtered.copy()
    df_decisions['Prediction'] = model.predict(df_decisions[features])
    df_decisions['Tomorrow_Price'] = df_decisions[col_close].shift(-1)
    df_decisions['Actual_Direction'] = (df_decisions['Tomorrow_Price'] > df_decisions[col_close]).astype(int)
    df_decisions['Correct'] = df_decisions['Prediction'] == df_decisions['Actual_Direction']

    df_display = df_decisions[['Date', col_close, 'Avg_Sentiment', 'Prediction', 'Actual_Direction', 'Correct']].copy()
    df_display.columns = ['Data', 'Cena', 'Sentyment', 'Prognoza', 'Rzeczywistość', 'Trafione?']
    df_display['Data'] = df_display['Data'].dt.strftime('%Y-%m-%d')
    df_display['Cena'] = df_display['Cena'].apply(lambda x: f"${x:.2f}")
    df_display['Sentyment'] = df_display['Sentyment'].apply(lambda x: f"{x:.4f}")
    df_display['Prognoza'] = df_display['Prognoza'].apply(lambda x: '📈 Wzrost' if x == 1 else '📉 Spadek')
    df_display['Rzeczywistość'] = df_display['Rzeczywistość'].apply(lambda x: '📈 Wzrost' if x == 1 else '📉 Spadek')
    df_display['Trafione?'] = df_display['Trafione?'].apply(lambda x: '✅' if x else '❌')
    
    total = len(df_decisions.dropna())
    correct = df_decisions['Correct'].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Liczba decyzji", total)
    with col2:
        st.metric("✅ Trafione", int(correct))
    with col3:
        st.metric("🎯 Skuteczność", f"{accuracy:.1f}%")
    
    st.markdown("---")
    
    filter_option = st.radio("Filtruj:", ["Wszystkie", "Tylko trafione ✅", "Tylko błędne ❌"], horizontal=True)
    
    if filter_option == "Tylko trafione ✅":
        df_display = df_display[df_display['Trafione?'] == '✅']
    elif filter_option == "Tylko błędne ❌":
        df_display = df_display[df_display['Trafione?'] == '❌']
    

    st.dataframe(
        df_display.sort_values('Data', ascending=False).dropna(),
        use_container_width=True,
        height=400
    )
    

    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Pobierz jako CSV",
        csv,
        "decyzje_systemu.csv",
        "text/csv"
    )

elif mode == "🔧 Panel Techniczny":
    
    st.markdown("### 🔧 Panel Techniczny")
    st.markdown("*Szczegółowe dane i analiza dla zaawansowanych użytkowników*")
    
    tab1, tab2, tab3 = st.tabs(["📈 Wykres Cen i Sentymentu", "📊 Statystyki", "🗂️ Surowe Dane"])
    
    with tab1:

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=df_filtered['Date'],
                y=df_filtered[col_close],
                name='Cena AAPL',
                line=dict(color='#1E88E5', width=2),
                hovertemplate='Cena: $%{y:.2f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        colors = ['#4CAF50' if x > 0 else '#F44336' if x < 0 else '#9E9E9E' for x in df_filtered['Avg_Sentiment']]
        fig.add_trace(
            go.Bar(
                x=df_filtered['Date'],
                y=df_filtered['Avg_Sentiment'],
                name='Sentyment',
                marker_color=colors,
                opacity=0.5,
                hovertemplate='Sentyment: %{y:.4f}<extra></extra>'
            ),
            secondary_y=True
        )
        
        fig.add_hline(y=0, line_dash='dash', line_color='grey', opacity=0.5, secondary_y=True)
        
        fig.update_xaxes(title_text='Data')
        fig.update_yaxes(title_text='Cena Akcji (USD)', secondary_y=False, color='#1E88E5')
        fig.update_yaxes(title_text='Sentyment (FinBERT)', secondary_y=True, color='#FF7043')
        
        fig.update_layout(
            hovermode='x unified',
            margin=dict(l=0, r=0, t=30, b=0),
            height=450,
            legend=dict(yanchor='top', y=0.99, xanchor='right', x=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Statystyki opisowe")
        
        stats_df = df_filtered[[col_close, 'Volume', 'Avg_Sentiment']].describe()
        stats_df.columns = ['Cena', 'Wolumen', 'Sentyment']
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
        
        st.markdown("#### Korelacje")
        corr = df_filtered[[col_close, 'Volume', 'Avg_Sentiment']].corr()
        corr.index = ['Cena', 'Wolumen', 'Sentyment']
        corr.columns = ['Cena', 'Wolumen', 'Sentyment']
        
        fig = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale='RdYlGn',
            zmin=-1, zmax=1,
            aspect='auto'
        )
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.dataframe(df_filtered.sort_values('Date', ascending=False), use_container_width=True, height=400)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    <p>StockSentiment AI v1.0</p>
    <p>Praca Inżynierska 2026</p>
    <p>Bartosz Ziętek</p>
</div>
""", unsafe_allow_html=True)