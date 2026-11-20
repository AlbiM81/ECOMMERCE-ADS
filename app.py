
import json
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# 1. KONFIGURASI APLIKASI
# =========================================================

st.set_page_config(
    page_title="Prediksi Pembelian E-Commerce",
    page_icon="🛒",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best_purchase_predictor.pkl"
META_PATH = BASE_DIR / "model_metadata.json"
LOG_PATH = BASE_DIR / "prediction_logs.jsonl"

MAX_UPLOAD_MB = 10

NUMERIC_COLS = [
    "product_views",
    "cart_additions",
    "ad_clicks",
    "discount_rate",
    "user_rating",
    "time_on_site",
]

REQUIRED_COLS = NUMERIC_COLS + ["device_type"]

DEVICE_OPTIONS = ["Mobile", "Desktop", "Tablet"]


# =========================================================
# 2. LOGGING AKTIVITAS
# =========================================================

def log_event(action, status, **details):
    """Mencatat aktivitas aplikasi tanpa menyimpan data pribadi."""

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "status": status,
        **details,
    }

    try:
        with open(LOG_PATH, "a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")
        return True

    except OSError:
        return False


# =========================================================
# 3. MEMUAT MODEL DAN METADATA
# =========================================================

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_metadata(metadata_path):
    if not Path(metadata_path).exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


if not MODEL_PATH.exists():
    st.error(
        "Model tidak ditemukan. "
        "Pastikan best_purchase_predictor.pkl berada "
        "dalam direktori aplikasi."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    log_event(
        "model_loading",
        "error",
        error_type=type(error).__name__,
    )
    st.error("Model gagal dimuat. Periksa file model dan dependensinya.")
    st.stop()

try:
    metadata = load_metadata(str(META_PATH))
except Exception as error:
    metadata = {}
    log_event(
        "metadata_loading",
        "error",
        error_type=type(error).__name__,
    )
    st.warning("Metadata model tidak dapat dibaca.")


# =========================================================
# 4. VALIDASI DATA
# =========================================================

def validate_data(data):
    """Memeriksa kelengkapan, tipe, dan rentang nilai input."""

    errors = []
    data = data.copy()

    missing_cols = [
        col for col in REQUIRED_COLS
        if col not in data.columns
    ]

    if missing_cols:
        errors.append(
            f"Kolom yang belum tersedia: {missing_cols}"
        )
        return data, errors

    # Konversi fitur numerik
    for col in NUMERIC_COLS:
        original = data[col]
        converted = pd.to_numeric(original, errors="coerce")

        invalid_conversion = (
            original.notna() & converted.isna()
        )

        if invalid_conversion.any():
            rows = (data.index[invalid_conversion] + 1).tolist()
            errors.append(
                f"{col}: terdapat nilai bukan angka pada baris {rows[:10]}"
            )

        data[col] = converted

        if data[col].isna().any():
            errors.append(
                f"{col}: terdapat nilai kosong."
            )

        if np.isinf(data[col].dropna()).any():
            errors.append(
                f"{col}: terdapat nilai tak terhingga."
            )

    # Pemeriksaan nilai nonnegatif
    nonnegative_cols = [
        "product_views",
        "cart_additions",
        "ad_clicks",
        "time_on_site",
    ]

    for col in nonnegative_cols:
        if (data[col].dropna() < 0).any():
            errors.append(
                f"{col}: nilai tidak boleh negatif."
            )

    # Batas diskon
    if (
        (data["discount_rate"].dropna() < 0).any()
        or (data["discount_rate"].dropna() > 0.5).any()
    ):
        errors.append(
            "discount_rate harus berada pada rentang 0 sampai 0.5."
        )

    # Batas rating
    if (
        (data["user_rating"].dropna() < 1).any()
        or (data["user_rating"].dropna() > 5).any()
    ):
        errors.append(
            "user_rating harus berada pada rentang 1 sampai 5."
        )

    # Validasi perangkat
    data["device_type"] = data["device_type"].astype("string").str.strip()

    invalid_device = (
        data["device_type"].isna()
        | ~data["device_type"].isin(DEVICE_OPTIONS)
    )

    if invalid_device.any():
        errors.append(
            "device_type hanya boleh Mobile, Desktop, atau Tablet."
        )

    return data, errors


# =========================================================
# 5. MONITORING AKTIVITAS
# =========================================================

def get_monitoring_summary():
    """Membaca ringkasan aktivitas dari file log lokal."""

    if not LOG_PATH.exists():
        return []

    events = []

    try:
        with open(LOG_PATH, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []

    return events


# =========================================================
# 6. TAMPILAN UTAMA
# =========================================================

st.title("🛒 Prediksi Jumlah Pembelian E-Commerce")

st.write(
    "Aplikasi prototipe Machine Learning untuk memprediksi "
    "jumlah pembelian (purchases) berdasarkan perilaku "
    "dan interaksi pengguna."
)

st.caption("Deployment lokal menggunakan Streamlit.")


# =========================================================
# 7. INFORMASI MODEL
# =========================================================

with st.sidebar:
    st.header("Informasi Model")

    st.write(
        f"**Model:** "
        f"{metadata.get('model_name', 'Ridge Regression')}"
    )

    cv_score = metadata.get("cv_r2_mean")

    if cv_score is not None:
        st.metric("CV R² Mean", f"{cv_score:.4f}")

    test_metrics = metadata.get("test_metrics", {})

    if test_metrics:
        st.divider()
        st.subheader("Evaluasi Model")

        for metric, label in [
            ("R2", "Test R²"),
            ("MAE", "Test MAE"),
            ("RMSE", "Test RMSE"),
        ]:
            value = test_metrics.get(metric)

            if value is not None:
                st.write(f"**{label}:** {value:.4f}")

    st.divider()
    st.caption("Model: best_purchase_predictor.pkl")


# =========================================================
# 8. MENU PREDIKSI
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "Prediksi Satuan",
    "Prediksi Batch (CSV / Excel)",
    "Monitoring",
])


# =========================================================
# 9. PREDIKSI SATUAN
# =========================================================

with tab1:

    st.subheader("Masukkan Data Interaksi Pengguna")

    col1, col2 = st.columns(2)

    with col1:
        product_views = st.number_input(
            "product_views (Tampilan Produk)",
            min_value=0.0,
            value=450.0,
            step=1.0,
        )

        cart_additions = st.number_input(
            "cart_additions (Tambah ke Keranjang)",
            min_value=0.0,
            value=90.0,
            step=1.0,
        )

        ad_clicks = st.number_input(
            "ad_clicks (Klik Iklan)",
            min_value=0.0,
            value=120.0,
            step=1.0,
        )

        device_type = st.selectbox(
            "device_type (Jenis Perangkat)",
            options=DEVICE_OPTIONS,
        )

    with col2:
        discount_rate = st.slider(
            "discount_rate (Tingkat Diskon)",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.01,
        )

        user_rating = st.slider(
            "user_rating (Rating Pengguna)",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
        )

        time_on_site = st.number_input(
            "time_on_site (Durasi di Situs / menit)",
            min_value=0.0,
            value=10.0,
            step=0.5,
        )

    if st.button("Prediksi Pembelian", type="primary"):

        start_time = time.perf_counter()

        input_df = pd.DataFrame([{
            "product_views": product_views,
            "cart_additions": cart_additions,
            "ad_clicks": ad_clicks,
            "discount_rate": discount_rate,
            "user_rating": user_rating,
            "time_on_site": time_on_site,
            "device_type": device_type,
        }])

        clean_df, errors = validate_data(input_df)

        if errors:
            for error in errors:
                st.error(error)

            log_event(
                "single_prediction",
                "validation_error",
                records=1,
            )

        else:
            try:
                prediction = float(model.predict(clean_df)[0])

                duration = (
                    time.perf_counter() - start_time
                ) * 1000

                if not np.isfinite(prediction):
                    raise ValueError(
                        "Hasil prediksi bukan angka valid."
                    )

                st.success(
                    f"Perkiraan Jumlah Pembelian "
                    f"(purchases): {prediction:.2f}"
                )

                st.caption(
                    f"Waktu proses: {duration:.2f} ms"
                )

                log_event(
                    "single_prediction",
                    "success",
                    records=1,
                    prediction=round(prediction, 2),
                    duration_ms=round(duration, 2),
                )

            except Exception as error:
                log_event(
                    "single_prediction",
                    "error",
                    error_type=type(error).__name__,
                )

                st.error(
                    "Prediksi gagal dilakukan. "
                    "Periksa input dan kesesuaian model."
                )


# =========================================================
# 10. PREDIKSI BATCH
# =========================================================

with tab2:

    st.subheader("Unggah Dataset CSV atau Excel")

    st.info(
        "Pastikan file memiliki tujuh kolom fitur "
        "yang diperlukan oleh model."
    )

    uploaded_file = st.file_uploader(
        "Pilih berkas",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:

            st.error(
                f"Ukuran file maksimal {MAX_UPLOAD_MB} MB."
            )

            log_event(
                "batch_prediction",
                "validation_error",
                error_type="file_too_large",
            )

        else:

            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    batch_df = pd.read_csv(
                        BytesIO(file_bytes)
                    )
                else:
                    batch_df = pd.read_excel(
                        BytesIO(file_bytes),
                        engine="openpyxl",
                    )

                st.write(
                    f"Jumlah data yang diunggah: "
                    f"{len(batch_df)} baris"
                )

                if batch_df.empty:
                    st.error("Dataset tidak memiliki data.")

                else:

                    clean_df, errors = validate_data(batch_df)

                    if errors:

                        st.error(
                            "Dataset belum memenuhi persyaratan:"
                        )

                        for error in errors:
                            st.write(f"- {error}")

                        log_event(
                            "batch_prediction",
                            "validation_error",
                            records=len(batch_df),
                        )

                    else:

                        if st.button(
                            "Proses Prediksi Batch",
                            type="primary",
                        ):

                            start_time = time.perf_counter()

                            try:
                                predictions = model.predict(
                                    clean_df[REQUIRED_COLS]
                                )

                                if not np.isfinite(predictions).all():
                                    raise ValueError(
                                        "Terdapat hasil prediksi tidak valid."
                                    )

                                result_df = batch_df.copy()

                                result_df[
                                    "predicted_purchases"
                                ] = np.round(predictions, 2)

                                duration = (
                                    time.perf_counter()
                                    - start_time
                                ) * 1000

                                st.success(
                                    "Prediksi batch berhasil dilakukan!"
                                )

                                st.dataframe(
                                    result_df.head(20),
                                    use_container_width=True,
                                )

                                st.metric(
                                    "Jumlah data berhasil diprediksi",
                                    len(result_df),
                                )

                                st.caption(
                                    f"Waktu proses: {duration:.2f} ms"
                                )

                                csv_bytes = result_df.to_csv(
                                    index=False
                                ).encode("utf-8-sig")

                                st.download_button(
                                    label="Unduh Hasil Prediksi (CSV)",
                                    data=csv_bytes,
                                    file_name="prediksi_purchases.csv",
                                    mime="text/csv",
                                )

                                log_event(
                                    "batch_prediction",
                                    "success",
                                    records=len(result_df),
                                    duration_ms=round(duration, 2),
                                    prediction_min=round(
                                        float(np.min(predictions)), 2
                                    ),
                                    prediction_max=round(
                                        float(np.max(predictions)), 2
                                    ),
                                )

                            except Exception as error:

                                log_event(
                                    "batch_prediction",
                                    "error",
                                    error_type=type(error).__name__,
                                )

                                st.error(
                                    "Prediksi batch gagal. "
                                    "Periksa struktur dataset dan model."
                                )

            except Exception as error:

                log_event(
                    "file_reading",
                    "error",
                    error_type=type(error).__name__,
                )

                st.error(
                    "File gagal dibaca. "
                    "Pastikan format CSV atau Excel benar."
                )


# =========================================================
# 11. MONITORING DAN LOGGING
# =========================================================

with tab3:

    st.subheader("Monitoring Aktivitas Aplikasi")

    events = get_monitoring_summary()

    successful = [
        e for e in events if e.get("status") == "success"
    ]

    failed = [
        e for e in events
        if e.get("status") in ["error", "validation_error"]
    ]

    total_records = sum(
        int(e.get("records", 0))
        for e in successful
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Aktivitas Berhasil",
        len(successful),
    )

    col2.metric(
        "Total Aktivitas Gagal",
        len(failed),
    )

    col3.metric(
        "Total Data Diprediksi",
        total_records,
    )

    st.divider()

    st.write("**Riwayat Aktivitas Terbaru**")

    if events:

        recent_events = events[-10:][::-1]

        monitoring_df = pd.DataFrame([
            {
                "Waktu": e.get("timestamp", ""),
                "Aktivitas": e.get("action", ""),
                "Status": e.get("status", ""),
                "Jumlah Data": e.get("records", 0),
                "Waktu Proses (ms)": e.get("duration_ms", "-"),
            }
            for e in recent_events
        ])

        st.dataframe(
            monitoring_df,
            use_container_width=True,
        )

        st.caption(
            "Log disimpan secara lokal pada prediction_logs.jsonl."
        )

    else:
        st.info(
            "Belum ada aktivitas yang tercatat. "
            "Lakukan prediksi terlebih dahulu."
        )

    if st.button("Muat Ulang Monitoring"):
        st.rerun()


# =========================================================
# 12. INFORMASI DAN DOKUMENTASI
# =========================================================

with st.expander("Petunjuk Penggunaan Aplikasi"):

    st.markdown("""
    **Prediksi Satuan**
    1. Masukkan data interaksi pengguna.
    2. Pilih jenis perangkat.
    3. Klik tombol Prediksi Pembelian.
    4. Hasil prediksi akan ditampilkan.

    **Prediksi Batch**
    1. Siapkan file CSV atau Excel.
    2. Pastikan tujuh kolom fitur tersedia.
    3. Unggah file.
    4. Klik Proses Prediksi Batch.
    5. Unduh hasil prediksi dalam format CSV.

    **Monitoring**
    - Melihat jumlah aktivitas berhasil.
    - Melihat aktivitas gagal.
    - Melihat jumlah data yang diprediksi.
    - Melihat riwayat aktivitas aplikasi.

    **Catatan**
    Aplikasi ini merupakan prototipe deployment lokal.
    Log aktivitas tersimpan di komputer tempat aplikasi berjalan.
    """)

st.divider()

st.caption(
    "Prediksi Pembelian E-Commerce | "
    "Ridge Regression | Streamlit Local Deployment"
)
