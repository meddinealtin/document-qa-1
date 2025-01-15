import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# CSS ile Arka Plan ve Kenarları Özelleştirme
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f0; /* Gri arka plan */
    }
    .stApp {
        border: 20px solid #21395E; /* Lacivert kenar */
        border-radius: 20px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Görsel Ekleme
st.image("/workspaces/document-qa-1/goldberg.png", use_container_width=True)

# Streamlit Başlığı
st.title("Zaman Serisi Tahmini")

# CSV Dosyasını Yükleme
uploaded_file = st.file_uploader("CSV dosyanızı yükleyin", type="csv")

if uploaded_file:
    # Veriyi Yükleme
    df = pd.read_csv(uploaded_file)

    # Veri Gözden Geçirme
    st.write("Yüklenen Veri:")
    st.write(df.head())

     # Eksik Değerleri Kontrol Etme
    st.write("Eksik Değerler:")
    st.write(df.isna().sum())  # Her sütunda kaç eksik değer olduğunu gösterir


    # İstenmeyen sütunları kaldırma
    columns_to_keep = ['satis_fiyati', 'kar', 'kar_orani', 'urun_grubu', 'teslim_tarihi', 'siparis_tarihi']
    df = df[columns_to_keep]

    # Tarih ve Değer Sütunlarını Seçme
    date_column = st.selectbox("Tarih sütununu seçin:", df.columns)
    value_column = st.selectbox("Değer sütununu seçin:", [col for col in df.columns if col != date_column])

    # Ek Tarih Kontrolü (Sipariş ve Teslim Tarihi)
    if "siparis_tarihi" in df.columns and "teslim_tarihi" in df.columns:
        # Tarih sütunlarını datetime formatına çevirme
        df["siparis_tarihi"] = pd.to_datetime(df["siparis_tarihi"])
        df["teslim_tarihi"] = pd.to_datetime(df["teslim_tarihi"])
        
        # Sipariş tarihindeki maksimum değeri bulma
        max_siparis_tarihi = df["siparis_tarihi"].max()

        # Teslim tarihinin bu maksimum değerden büyük olmadığı verileri filtreleme
        df = df[df["teslim_tarihi"] <= max_siparis_tarihi]
        
        # Filtrelenmiş veriyi gösterme
        st.write("Filtrelenmiş Veri (Teslim tarihi, sipariş tarihindeki maksimum değerden büyük değil):")
        st.write(df)

    # Tarih sütununu datetime formatına çevirme
    df[date_column] = pd.to_datetime(df[date_column])

    # Prophet için gerekli sütun adlarını ayarlama
    df = df[[date_column, value_column]].rename(columns={date_column: "ds", value_column: "y"})

    # Aylık Bazda Gruplama (Opsiyonel)
    df = df.groupby(pd.Grouper(key="ds", freq="M")).sum().reset_index()

    # Aylık toplam satışların outlier'larını tespit etme (Sadece üst sınır)
    Q1 = df["y"].quantile(0.25)
    Q3 = df["y"].quantile(0.75)
    IQR = Q3 - Q1

    # Yalnızca üst sınırdaki outlier'ları filtreleme
    upper_limit = Q3 + 3 * IQR

    # Outlier olan satışları yarıya indirme
    df_no_outliers = df.copy()
    df_no_outliers.loc[df_no_outliers["y"] > upper_limit, "y"] = df_no_outliers["y"] / 2

    # Regresör Listesi
    regressors = ['AMELİYAT MASASI', 'HİDROJEN PEROKSİT', 'KARTUŞ', 
                  'OKSİJEN SİSTEMİ', 'OTOKLAV', 'REVERSE OSMOS', 'YIKAMA']

    # Kullanıcıdan Tahmin Süresi Almak için Slider
    periods = st.slider("Tahmin süresini (ay olarak) seçin:", min_value=1, max_value=36, value=12, step=1)

    # Prophet Modeli
    model = Prophet()

    # Regresörleri Modele Eklemek
    for regressor in regressors:
        if regressor in df.columns:
            model.add_regressor(regressor)

    # Modeli Eğitmek (Outlier'lar yarıya indirilmiş veriyle)
    model.fit(df_no_outliers)

    # Tahmin İçin Gelecek Tarihler
    future = model.make_future_dataframe(periods=periods, freq="M")

    # Regresörlerin Gelecek Değerlerini Ekleme
    for regressor in regressors:
        if regressor in df.columns:
            future[regressor] = df[regressor].iloc[-1]  # Son bilinen değerle doldurulabilir

    # Tahmin Yapma
    forecast = model.predict(future)

    # Tahmin Sonuçları
    st.write("Tahmin Sonuçları:")
    st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

    # Orijinal Veri ve Tahminleri Görselleştirme
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    # Sezonluk Bileşenlerin Görselleştirilmesi
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)
