import pickle
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = "data/02_realisasi_anggaran_klasifikasi.csv"
MODEL_PATH = "model/Best_model.pkcls"
NUMERIC_FEATURES = [
    "jumlah_spm",
    "revisi_dipa",
    "deviasi_rpd_persen",
    "skor_ikpa",
]
TYPE_CATEGORIES = [
    "Dekonsentrasi",
    "Kantor Daerah",
    "Kantor Pusat",
    "Tugas Pembantuan",
]
MODEL_FEATURES = NUMERIC_FEATURES + [f"tipe_satker={t}" for t in TYPE_CATEGORIES]
LABEL_MAP = {0: "Tidak", 1: "Ya"}

st.set_page_config(
    page_title="Dashboard Realisasi Anggaran",
    page_icon="📊",
    layout="wide",
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["realisasi_tercapai_95persen"] = df["realisasi_tercapai_95persen"].astype(str).str.strip()
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

@st.cache_data
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES].copy()
    for tipe in TYPE_CATEGORIES:
        X[f"tipe_satker={tipe}"] = (df["tipe_satker"] == tipe).astype(float)
    return X[MODEL_FEATURES]

def add_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    X = prepare_features(df)
    y_pred = model.predict(X.to_numpy())
    proba = model.predict_proba(X.to_numpy())[:, 1]
    df_result = df.copy()
    df_result["prediksi_model"] = np.where(y_pred == 1, "Ya", "Tidak")
    df_result["probabilitas_tercapai_95persen"] = proba
    return df_result

def predict_single(values: dict, model):
    row = pd.DataFrame([values])
    features = prepare_features(row)
    y_pred = model.predict(features.to_numpy())[0]
    proba = model.predict_proba(features.to_numpy())[0, 1]
    return LABEL_MAP[int(y_pred)], float(proba)


def main():
    st.title("📊 Dashboard Realisasi Anggaran")
    st.markdown(
        "Aplikasi ini menampilkan ringkasan data realisasi anggaran, visualisasi utama, dan prediksi model untuk estimasi apakah realisasi akan mencapai 95% atau lebih."
    )

    df = load_data()
    model = load_model()

    with st.sidebar:
        st.header("Filter Data")
        ministry = st.multiselect("Kementerian", sorted(df["nama_kementerian"].unique()), default=sorted(df["nama_kementerian"].unique()))
        province = st.multiselect("Provinsi", sorted(df["provinsi"].unique()), default=sorted(df["provinsi"].unique()))
        satker_type = st.multiselect("Tipe Satker", sorted(df["tipe_satker"].unique()), default=sorted(df["tipe_satker"].unique()))
        expenditure_type = st.multiselect(
            "Jenis Belanja Utama",
            sorted(df["jenis_belanja_utama"].unique()),
            default=sorted(df["jenis_belanja_utama"].unique()),
        )
        pagu_min, pagu_max = st.slider(
            "Rentang Pagu (miliar)",
            float(df["pagu_miliar"].min()),
            float(df["pagu_miliar"].max()),
            (float(df["pagu_miliar"].min()), float(df["pagu_miliar"].max())),
        )
        show_predictions = st.checkbox("Tampilkan hasil prediksi model", value=True)
        st.markdown("---")
        st.write("Model: Orange Logistic Regression")
        st.write("Output: prediksi realisasi tercapai 95%")

    filtered = df[
        df["nama_kementerian"].isin(ministry)
        & df["provinsi"].isin(province)
        & df["tipe_satker"].isin(satker_type)
        & df["jenis_belanja_utama"].isin(expenditure_type)
        & df["pagu_miliar"].between(pagu_min, pagu_max)
    ]

    st.subheader("Ringkasan Data")
    total_rows = len(filtered)
    reached = (filtered["realisasi_tercapai_95persen"] == "Ya").sum()
    reached_rate = 0 if total_rows == 0 else reached / total_rows * 100
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Satker", total_rows)
    col2.metric("Tercapai 95%", f"{reached} ({reached_rate:.1f}%)")
    col3.metric("Rata-rata Skor IKPA", f"{filtered['skor_ikpa'].mean():.2f}")
    col4.metric("Rata-rata Deviasi RPD", f"{filtered['deviasi_rpd_persen'].mean():.2f}")

    if total_rows == 0:
        st.warning("Tidak ada data pada filter yang dipilih. Silakan sesuaikan filter sidebar.")
        return

    st.markdown("### Visualisasi Utama")
    fig1 = px.bar(
        filtered["realisasi_tercapai_95persen"].value_counts().reset_index().rename(columns={"index": "Status", "realisasi_tercapai_95persen": "Jumlah"}),
        x="Status",
        y="Jumlah",
        color="Status",
        title="Distribusi Realisasi Tercapai 95%",
        color_discrete_map={"Ya": "#2ca02c", "Tidak": "#d62728"},
    )
    fig2 = px.bar(
        filtered.groupby("tipe_satker")["skor_ikpa"].mean().reset_index().sort_values("skor_ikpa", ascending=False),
        x="tipe_satker",
        y="skor_ikpa",
        title="Rata-rata Skor IKPA per Tipe Satker",
        labels={"skor_ikpa": "Rata-rata Skor IKPA", "tipe_satker": "Tipe Satker"},
    )
    fig3 = px.bar(
        filtered.groupby("jenis_belanja_utama")["deviasi_rpd_persen"].mean().reset_index().sort_values("deviasi_rpd_persen", ascending=False),
        x="jenis_belanja_utama",
        y="deviasi_rpd_persen",
        title="Rata-rata Deviasi RPD per Jenis Belanja Utama",
        labels={"deviasi_rpd_persen": "Rata-rata Deviasi RPD", "jenis_belanja_utama": "Jenis Belanja Utama"},
    )
    fig4 = px.scatter(
        filtered,
        x="pagu_miliar",
        y="skor_ikpa",
        color="realisasi_tercapai_95persen",
        hover_data=["nama_kementerian", "provinsi", "tipe_satker"],
        title="Pagu vs Skor IKPA",
        labels={"pagu_miliar": "Pagu (miliar)", "skor_ikpa": "Skor IKPA"},
    )

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    if show_predictions:
        st.markdown("### Prediksi Model")
        predictions = add_predictions(filtered, model)
        pred_counts = predictions["prediksi_model"].value_counts().reindex(["Ya", "Tidak"], fill_value=0).reset_index()
        pred_counts.columns = ["Prediksi", "Jumlah"]
        fig_pred = px.bar(
            pred_counts,
            x="Prediksi",
            y="Jumlah",
            color="Prediksi",
            title="Hasil Prediksi Model Realisasi 95%",
            color_discrete_map={"Ya": "#1f77b4", "Tidak": "#ff7f0e"},
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        prediction_table = pd.crosstab(
            predictions["realisasi_tercapai_95persen"],
            predictions["prediksi_model"],
            rownames=["Aktual"],
            colnames=["Prediksi"],
        )
        st.subheader("Tabel Perbandingan Aktual vs Prediksi")
        st.dataframe(prediction_table, use_container_width=True)

        st.subheader("Top 10 Satker dengan Probabilitas Tertinggi")
        st.dataframe(
            predictions.sort_values("probabilitas_tercapai_95persen", ascending=False)[
                ["kode_satker", "nama_kementerian", "provinsi", "tipe_satker", "jenis_belanja_utama", "skor_ikpa", "probabilitas_tercapai_95persen", "prediksi_model", "realisasi_tercapai_95persen"]
            ].head(10),
            use_container_width=True,
        )

        st.subheader("Form Prediksi Satker Baru")
        with st.form(key="prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                jumlah_spm = st.number_input("Jumlah SPM", min_value=0.0, value=50.0)
                revisi_dipa = st.number_input("Revisi DIPA", min_value=0.0, value=1.0)
                deviasi_rpd = st.number_input("Deviasi RPD (%)", min_value=0.0, value=10.0)
            with c2:
                skor_ikpa = st.number_input("Skor IKPA", min_value=0.0, max_value=100.0, value=85.0)
                tipe_satker_input = st.selectbox("Tipe Satker", TYPE_CATEGORIES)
            predict_button = st.form_submit_button("Hitung Prediksi")
            if predict_button:
                values = {
                    "jumlah_spm": jumlah_spm,
                    "revisi_dipa": revisi_dipa,
                    "deviasi_rpd_persen": deviasi_rpd,
                    "skor_ikpa": skor_ikpa,
                    "tipe_satker": tipe_satker_input,
                }
                label, probability = predict_single(values, model)
                st.metric("Prediksi", label, delta=f"{probability*100:.1f}% kemungkinan Ya")

    st.markdown("---")
    st.subheader("Tabel Data Terpilih")
    cols = [
        "kode_satker",
        "nama_kementerian",
        "provinsi",
        "tipe_satker",
        "jenis_belanja_utama",
        "pagu_miliar",
        "jumlah_pegawai",
        "jumlah_spm",
        "revisi_dipa",
        "realisasi_tw1_persen",
        "realisasi_tw2_persen",
        "realisasi_tw3_persen",
        "deviasi_rpd_persen",
        "skor_ikpa",
        "realisasi_tercapai_95persen",
    ]
    st.dataframe(filtered[cols].reset_index(drop=True), use_container_width=True)

if __name__ == "__main__":
    main()
