import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os

st.set_page_config(page_title="Dashboard Klaster", layout="wide")
st.title("📊 Aplikasi Analisis Klaster")

# ======== File default hasil clustering ========
file_csv_default = "Data_Fixx_Visualisasi.csv"
if os.path.exists(file_csv_default):
    df_default = pd.read_csv(file_csv_default, sep=None, engine='python')

else:
    df_default = None

# ======== Sidebar Menu ========
menu = st.sidebar.selectbox("Pilih Menu", ["Dashboard", "Cluster", "Pie Chart", "Cluster Baru"])

# ======== Menu: Dashboard ========
if menu == "Dashboard":
    st.subheader("📄 Data Lengkap")
    if df_default is not None:
        st.dataframe(df_default)
    else:
        st.warning("⚠️ Belum ada file CSV utama yang dimuat.")

# ======== Menu: Cluster (Bar Chart) ========
elif menu == "Cluster":
    st.subheader("📊 Jumlah Data per Cluster (Bar Chart)")
    if df_default is not None and 'Cluster' in df_default.columns:
        counts = df_default['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots()
        sns.barplot(x=counts.index.astype(str), y=counts.values, palette="pastel", ax=ax)
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.5, str(v), ha='center')
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Jumlah Data")
        ax.set_title("Distribusi Jumlah Data per Cluster")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Kolom 'Cluster' tidak ditemukan di file.")

# ======== Menu: Pie Chart ========
elif menu == "Pie Chart":
    st.subheader("🥧 Pie Chart Distribusi Klaster")
    if df_default is not None and 'Cluster' in df_default.columns:
        fig, ax = plt.subplots()
        df_default['Cluster'].value_counts().sort_index().plot.pie(
            autopct='%1.1f%%',
            labels=[f"Cluster {int(i)}" for i in sorted(df_default['Cluster'].unique())],
            ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Tidak ada kolom 'Cluster' di file.")

# ======== Menu: Cluster Baru (Upload dan Clustering Otomatis) ========
elif menu == "Cluster Baru":
    st.subheader("📥 Upload Data untuk Clustering Baru")

    uploaded_file = st.file_uploader("Upload file CSV (delimiter ;)", type=["csv"])
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file, sep=';')
        st.write("📄 Data Awal", df_new.head())

        # Pilih jumlah cluster
        k = st.slider("Pilih jumlah klaster", 2, 10, 3)

        # Jalankan clustering
        df_numerik = df_new.select_dtypes(include='number')
        model = KMeans(n_clusters=k, random_state=42)
        df_new['Cluster'] = model.fit_predict(df_numerik)

        st.success("✅ Clustering berhasil dijalankan!")
        st.subheader("📄 Data Hasil Clustering Baru")
        st.dataframe(df_new)

        # Simpan hasil
        df_new.to_csv(file_csv_default, index=False, sep=';')
        st.success("💾 Hasil disimpan sebagai 'data_cluster.csv' dan akan muncul di semua menu.")