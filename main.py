import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings

# Mengabaikan peringatan yang tidak relevan
warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman Utama ---
st.set_page_config(
    page_title="Finance Modeling Pro",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# BAGIAN 1: FUNGSI-FUNGSI UTAMA (DATA, MODEL, PLOTTING, HELPERS)
# ==============================================================================

# --- Fungsi untuk Pengambilan Data ---

@st.cache_resource(ttl=timedelta(hours=1))
def get_ticker(ticker_symbol):
    """
    Mengambil dan memvalidasi objek Ticker yfinance. Disimpan di cache.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        if stock.history(period="1d").empty:
            st.error(f"Ticker '{ticker_symbol}' tidak valid atau tidak ditemukan. Mohon coba ticker lain (contoh: GOOGL, BBCA.JK).")
            return None
        return stock
    except Exception as e:
        st.error(f"Gagal terhubung ke yfinance untuk ticker '{ticker_symbol}': {e}")
        return None

@st.cache_data(ttl=timedelta(minutes=15))
def fetch_historical_data(ticker_symbol, start_date, end_date, interval='1d'):
    """
    Mengambil data historis (OHLCV) untuk ticker yang sudah divalidasi.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        data = stock.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
        if data.empty:
            st.warning(f"Tidak ada data historis yang ditemukan untuk {ticker_symbol} pada rentang waktu yang dipilih.")
            return None
        return data
    except Exception as e:
        st.error(f"Gagal mengambil data historis: {e}")
        return None

@st.cache_data(ttl=timedelta(hours=1))
def get_company_info(ticker_symbol):
    """
    Mengambil informasi profil perusahaan.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        return stock.info
    except Exception:
        st.warning("Gagal mengambil sebagian informasi detail perusahaan.")
        return {}

@st.cache_data(ttl=timedelta(hours=6))
def get_financial_statements(ticker_symbol):
    """
    Mengambil laporan keuangan (Laba Rugi, Neraca, Arus Kas).
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        return stock.financials, stock.balance_sheet, stock.cashflow
    except Exception as e:
        st.error(f"Gagal mengambil laporan keuangan: {e}")
        return None, None, None

# --- Fungsi untuk Model LSTM ---

def create_lstm_model(input_shape):
    """
    Membuat arsitektur model LSTM yang robust.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data_for_lstm(data, sequence_length=60):
    """
    Mempersiapkan data untuk model LSTM (scaling dan membuat sekuens).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    X_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, scaler

def train_lstm_model(X_train, y_train, epochs):
    """
    Melatih model LSTM dan mengembalikan model serta history pelatihan.
    """
    model = create_lstm_model(input_shape=(X_train.shape[1], 1))
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)
    return model, history

def make_lstm_predictions(model, data, sequence_length, prediction_days, scaler):
    """
    Membuat prediksi harga untuk beberapa hari ke depan secara iteratif.
    """
    last_sequence = data['Close'].values[-sequence_length:]
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    
    future_predictions_scaled = []
    current_batch = last_sequence_scaled.reshape(1, sequence_length, 1)

    for _ in range(prediction_days):
        next_pred_scaled = model.predict(current_batch, verbose=0)[0]
        future_predictions_scaled.append(next_pred_scaled)
        current_batch = np.append(current_batch[:, 1:, :], [[next_pred_scaled]], axis=1)
        
    predicted_prices = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    return predicted_prices

def calculate_performance_metrics(y_true, y_pred):
    """
    Menghitung metrik performa: MAE, RMSE, dan MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# --- Fungsi untuk Plotting ---

def plot_historical_data(df, ticker_symbol):
    """
    Membuat plot data historis (harga dan volume saja).
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'{ticker_symbol} Historical Price', 'Volume'), row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='rgba(100,100,150,0.5)'), row=2, col=1)
    fig.update_layout(title_text=f"{ticker_symbol} Price History", xaxis_rangeslider_visible=False, template='plotly_dark')
    return fig

def plot_training_loss(history):
    """
    Membuat plot training & validation loss.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
    fig.update_layout(title_text='Model Training vs. Validation Loss', xaxis_title='Epochs', yaxis_title='Loss', template='plotly_dark')
    return fig

def plot_price_predictions(historical_data, predicted_prices, prediction_days):
    """
    Membuat plot harga aktual dan hasil prediksi LSTM.
    """
    last_date = historical_data.index[-1]
    prediction_dates = pd.to_datetime([last_date + pd.DateOffset(days=i+1) for i in range(prediction_days)])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_data.index[-200:], y=historical_data['Close'][-200:], mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=prediction_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted Price', line=dict(dash='dash', color='tomato')))
    fig.update_layout(title='Actual vs. Predicted Stock Price', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
    return fig

# --- Fungsi Bantuan (Helpers) ---

def format_large_number(num):
    """
    Format angka besar menjadi format yang lebih mudah dibaca (Miliar, Triliun).
    """
    if not isinstance(num, (int, float)): return num
    if abs(num) >= 1e12: return f"{num / 1e12:.2f} T"
    if abs(num) >= 1e9: return f"{num / 1e9:.2f} M"
    if abs(num) >= 1e6: return f"{num / 1e6:.2f} Jt"
    return f"{num:,.2f}"

def calculate_dcf_industry(fcf, risk_free_rate, shares_outstanding, net_debt, years=5, terminal_growth_rate=0.03):
    """
    DCF industri: proyeksi FCF 5 tahun dengan pertumbuhan terminal growth rate tetap (default 3%),
    discount hanya dengan risk free rate.
    """
    if not all(isinstance(val, (int, float)) for val in [fcf, shares_outstanding, net_debt, risk_free_rate] if val is not None):
        return None, "Input tidak valid."
    projected_fcf = [fcf * ((1 + terminal_growth_rate) ** i) for i in range(1, years+1)]
    terminal_value = (projected_fcf[-1] * (1 + terminal_growth_rate)) / (risk_free_rate - terminal_growth_rate)
    discounted_fcf = [val / ((1 + risk_free_rate) ** (i+1)) for i, val in enumerate(projected_fcf)]
    discounted_terminal = terminal_value / ((1 + risk_free_rate) ** years)
    enterprise_value = sum(discounted_fcf) + discounted_terminal
    equity_value = enterprise_value - net_debt
    intrinsic_value = equity_value / shares_outstanding
    return intrinsic_value, "Sukses"

def dcf_simple(
    fcf,
    discount_rate,
    shares_outstanding,
    net_debt,
    years=5,
    growth_rate=0.05,
    terminal_growth=0.03
):
    # Proyeksi FCF 5 tahun dengan growth rate tetap
    projected_fcf = [fcf * ((1 + growth_rate) ** i) for i in range(1, years+1)]
    terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    discounted_fcf = [fcf_ / ((1 + discount_rate) ** (i+1)) for i, fcf_ in enumerate(projected_fcf)]
    discounted_terminal = terminal_value / ((1 + discount_rate) ** years)
    enterprise_value = sum(discounted_fcf) + discounted_terminal
    equity_value = enterprise_value - net_debt
    intrinsic_value = equity_value / shares_outstanding
    return intrinsic_value

# ==============================================================================
# BAGIAN 2: TAMPILAN (UI) HALAMAN
# ==============================================================================

def show_stock_prediction_page():
    st.title("📈 Prediksi Harga Saham")
    st.markdown("Gunakan model LSTM untuk memprediksi harga saham di masa depan berdasarkan data historis.")
    st.markdown("---")

    st.sidebar.header("Parameter Prediksi")
    with st.sidebar.form("prediksi_form"):
        ticker_symbol = st.text_input("Ticker Saham", "GOOGL").upper()
        # Validasi otomatis untuk saham Indonesia
        if ticker_symbol and not ticker_symbol.endswith('.JK') and len(ticker_symbol) <= 5:
            ticker_symbol += '.JK'
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Tanggal Mulai", datetime.now() - timedelta(days=365*4))
        end_date = col2.date_input("Tanggal Selesai", datetime.now())
        interval = st.selectbox("Interval Data", ['1d', '1wk', '1mo'], index=0)
        prediction_days = st.number_input("Jumlah Hari Prediksi:", 1, 90, 30, key="pred_days")
        epochs = st.number_input("Jumlah Epochs Pelatihan:", 10, 200, 50, key="epochs")
        run_prediction = st.form_submit_button("🚀 Jalankan Prediksi")

    data = fetch_historical_data(ticker_symbol, start_date, end_date, interval)
    if data is not None and not data.empty:
        st.subheader(f"Data Historis untuk {ticker_symbol}")
        st.plotly_chart(plot_historical_data(data, ticker_symbol), use_container_width=True)
        with st.expander("Lihat Data Historis Mentah"):
            st.dataframe(data, use_container_width=True)
        st.markdown("---")
        st.subheader("🔮 Prediksi dengan LSTM")
        if run_prediction:
            if len(data) < 70:
                st.error("Data historis tidak cukup untuk melatih model. Silakan pilih rentang tanggal yang lebih panjang.")
            else:
                with st.spinner("Mempersiapkan data dan melatih model LSTM... Ini mungkin memakan waktu beberapa saat."):
                    try:
                        X_train, y_train, scaler = prepare_data_for_lstm(data)
                        model, history = train_lstm_model(X_train, y_train, epochs)
                        predicted_prices = make_lstm_predictions(model, data, 60, prediction_days, scaler)
                        st.success("✅ Proses prediksi selesai!")
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.plotly_chart(plot_price_predictions(data, predicted_prices, prediction_days), use_container_width=True)
                        with col_res2:
                            st.plotly_chart(plot_training_loss(history), use_container_width=True)
                        train_predictions = scaler.inverse_transform(model.predict(X_train))
                        y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
                        mae, rmse, mape = calculate_performance_metrics(y_train_actual, train_predictions)
                        st.subheader("Metrik Performa Model (Evaluasi pada Data Training)")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("MAE", f"{mae:.2f}")
                        c2.metric("RMSE", f"{rmse:.2f}")
                        c3.metric("MAPE", f"{mape:.2f}%")
                        prediction_df = pd.DataFrame({
                            'Date': pd.date_range(start=data.index[-1] + timedelta(days=1), periods=prediction_days),
                            'Predicted_Close': predicted_prices.flatten()
                        })
                        st.dataframe(prediction_df.set_index('Date'), use_container_width=True)
                        csv = prediction_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 Unduh Hasil Prediksi (CSV)", csv, f'{ticker_symbol}_prediction.csv', 'text/csv')
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat proses prediksi: {e}")

def show_fundamental_analysis_page():
    st.title("📊 Analisis Fundamental Perusahaan")
    st.markdown("Evaluasi kesehatan finansial dan valuasi intrinsik sebuah perusahaan.")
    st.markdown("---")

    st.sidebar.header("Parameter Analisis")
    with st.sidebar.form("analisis_form"):
        ticker_symbol = st.text_input("Ticker Saham", "MSFT").upper()
        if ticker_symbol and not ticker_symbol.endswith('.JK') and len(ticker_symbol) <= 5:
            ticker_symbol += '.JK'
        discount_rate = st.number_input("Discount Rate (%)", 0.0, 20.0, 6.0, 0.1) / 100
        submit_analisis = st.form_submit_button("Tampilkan Analisis")

    if submit_analisis:
        ticker = get_ticker(ticker_symbol)
        if ticker:
            with st.spinner(f"Mengambil data fundamental untuk {ticker_symbol}..."):
                info = get_company_info(ticker_symbol)
                income_stmt, balance_sheet, cash_flow = get_financial_statements(ticker_symbol)

            if info:
                st.header(f"Ringkasan: {info.get('longName', ticker_symbol)}")
                c1, c2 = st.columns((3, 1))
                c1.write(info.get('longBusinessSummary', 'Tidak ada ringkasan bisnis.'))
                if info.get('logo_url'): c2.image(info.get('logo_url'), width=150)
                c2.markdown(f"**Sektor: {info.get('sector', 'N/A')}")
                if info.get('website'): c2.markdown(f"Situs: [{info.get('website')}]({info.get('website')})")

                st.subheader("Metrik Kunci")
                metrics_cols = st.columns(4)
                pe = info.get('trailingPE')
                eps = info.get('trailingEps')
                beta = info.get('beta')
                metrics_cols[0].metric("Kapitalisasi Pasar", format_large_number(info.get('marketCap')))
                metrics_cols[1].metric("P/E Ratio (TTM)", f"{pe:.2f}" if pe else "N/A")
                metrics_cols[2].metric("EPS (TTM)", f"{eps:.2f}" if eps else "N/A")
                metrics_cols[3].metric("Beta", f"{beta:.2f}" if beta else "N/A")

                st.markdown("---")
                st.subheader("Laporan Keuangan Tahunan")
                if income_stmt is not None and not income_stmt.empty:
                    tab1, tab2, tab3 = st.tabs(["Laba Rugi", "Neraca Keuangan", "Arus Kas"])
                    with tab1: st.dataframe(income_stmt.map(format_large_number), use_container_width=True)
                    with tab2: st.dataframe(balance_sheet.map(format_large_number), use_container_width=True)
                    with tab3: st.dataframe(cash_flow.map(format_large_number), use_container_width=True)
                st.markdown("---")
                st.subheader("Kalkulator Valuasi DCF (Sederhana)")
                if cash_flow is not None and 'Free Cash Flow' in cash_flow.index:
                    fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
                    shares = info.get('sharesOutstanding')
                    debt = info.get('netDebt', info.get('totalDebt', 0) - info.get('totalCash', 0))
                    st.info("DCF sederhana: proyeksi FCF 5 tahun (growth rate 5%), terminal value growth 3%, discount rate sesuai input. Semua data otomatis dari Yahoo Finance.", icon="⚠")
                    intrinsic_val = dcf_simple(fcf, discount_rate, shares, debt)
                    if intrinsic_val:
                        price = info.get('currentPrice', info.get('previousClose', 0))
                        st.markdown("#### Hasil Valuasi:")
                        val_cols = st.columns(2)
                        val_cols[0].metric("Nilai Intrinsik per Saham", format_large_number(intrinsic_val))
                        val_cols[1].metric("Harga Pasar Saat Ini", format_large_number(price))
                        if price and price > 0:
                            diff = ((intrinsic_val - price) / price) * 100
                            if diff > 10: st.success(f"Berdasarkan model ini, saham berpotensi Undervalued sebesar {diff:.2f}%.")
                            elif diff < -10: st.error(f"Berdasarkan model ini, saham berpotensi Overvalued sebesar {-diff:.2f}%.")
                            else: st.warning(f"Harga wajar (Fairly Valued), perbedaan {diff:.2f}%.")
                    else:
                        st.error("Gagal menghitung DCF.")
                else:
                    st.warning("Data Arus Kas Bebas (FCF) tidak tersedia untuk menghitung DCF.")

# ==============================================================================
# BAGIAN 3: KONTROL UTAMA APLIKASI
# ==============================================================================

st.sidebar.title("Finance Modeling Pro")
page_selection = st.sidebar.radio("Pilih Halaman", ("Prediksi Saham", "Analisis Fundamental"))
st.sidebar.info("Aplikasi ini menggunakan data real-time dari Yahoo Finance untuk analisis dan prediksi.")
st.sidebar.warning("Disclaimer: Ini bukan nasihat keuangan. Lakukan riset Anda sendiri.")
st.sidebar.markdown("---")


if page_selection == "Prediksi Saham":
    show_stock_prediction_page()
else:
    show_fundamental_analysis_page()
