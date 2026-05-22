# Dashboard Realisasi Anggaran

Dashboard ini menampilkan visualisasi data realisasi anggaran dan hasil prediksi model untuk memperkirakan apakah realisasi anggaran akan mencapai 95%.

## Cara menjalankan

1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

## Struktur proyek

- `data/02_realisasi_anggaran_klasifikasi.csv` — dataset realisasi anggaran.
- `model/Best_model.pkcls` — model Orange Logistic Regression untuk prediksi `realisasi_tercapai_95persen`.
- `streamlit_app.py` — aplikasi dashboard Streamlit.
- `requirements.txt` — paket Python yang dibutuhkan.
