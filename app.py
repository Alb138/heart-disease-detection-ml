import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Deteksi Risiko Penyakit Jantung",
    page_icon="🫀",
    layout="centered",
)

# Hilangkan icon anchor (⇗) di samping heading
st.markdown("""
<style>
[data-testid="stHeaderActionElements"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

CHOL_MEDIAN = 237.0

FEATURE_NAMES = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Cholesterol_missing', 'Sex_M',
    'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_Normal', 'RestingECG_ST',
    'ExerciseAngina_Y',
    'ST_Slope_Flat', 'ST_Slope_Up',
]

GOOGLE_FORM_URL = 'https://forms.gle/RJFoDA8ZBbnUhjo58'

MODEL_PATH  = os.path.join('models', 'svm_model.pkl')
SCALER_PATH = os.path.join('data', 'scaler.joblib')

@st.cache_resource
def load_artifacts():
    for path, label in [(MODEL_PATH, 'svm_model.pkl'), (SCALER_PATH, 'scaler.joblib')]:
        if not os.path.exists(path):
            st.error(f"❌ File tidak ditemukan: `{path}`")
            st.stop()
    return joblib.load(SCALER_PATH), joblib.load(MODEL_PATH)

def preprocess(inputs: dict) -> pd.DataFrame:
    chol_raw     = inputs['Cholesterol']
    chol_missing = 1 if chol_raw == 0 else 0
    chol_val     = CHOL_MEDIAN if chol_raw == 0 else float(chol_raw)

    row = {
        'Age':                 float(inputs['Age']),
        'RestingBP':           float(inputs['RestingBP']),
        'Cholesterol':         chol_val,
        'FastingBS':           float(inputs['FastingBS']),
        'MaxHR':               float(inputs['MaxHR']),
        'Oldpeak':             float(inputs['Oldpeak']),
        'Cholesterol_missing': float(chol_missing),
        'Sex_M':               1.0 if inputs['Sex'] == 'M' else 0.0,
        'ChestPainType_ATA':   1.0 if inputs['ChestPainType'] == 'ATA' else 0.0,
        'ChestPainType_NAP':   1.0 if inputs['ChestPainType'] == 'NAP' else 0.0,
        'ChestPainType_TA':    1.0 if inputs['ChestPainType'] == 'TA'  else 0.0,
        'RestingECG_Normal':   1.0 if inputs['RestingECG'] == 'Normal' else 0.0,
        'RestingECG_ST':       1.0 if inputs['RestingECG'] == 'ST'     else 0.0,
        'ExerciseAngina_Y':    1.0 if inputs['ExerciseAngina'] == 'Y'  else 0.0,
        'ST_Slope_Flat':       1.0 if inputs['ST_Slope'] == 'Flat'     else 0.0,
        'ST_Slope_Up':         1.0 if inputs['ST_Slope'] == 'Up'       else 0.0,
    }

    scaler, _ = load_artifacts()
    df = pd.DataFrame([row])[FEATURE_NAMES]
    return pd.DataFrame(scaler.transform(df), columns=FEATURE_NAMES)


st.title("🫀 Deteksi Dini Risiko Penyakit Jantung")
st.markdown(
    "Sistem pendukung keputusan berbasis **Support Vector Machine (SVM-RBF)** "
    "untuk skrining awal risiko penyakit jantung dari parameter pemeriksaan rutin."
)
st.divider()

# --- Contoh Data Valid ---
SAMPLE_DATA = {
    "Risiko Rendah – Pria 40th, ATA, BP normal": {
        "Age": 40, "Sex": "M", "ChestPainType": "ATA",
        "RestingBP": 140, "Cholesterol": 289, "FastingBS": 0,
        "RestingECG": "Normal", "MaxHR": 172, "ExerciseAngina": "N",
        "Oldpeak": 0.0, "ST_Slope": "Up",
    },
    "Risiko Tinggi – Pria 52th, ASY, depresi ST": {
        "Age": 52, "Sex": "M", "ChestPainType": "ASY",
        "RestingBP": 140, "Cholesterol": 266, "FastingBS": 0,
        "RestingECG": "Normal", "MaxHR": 134, "ExerciseAngina": "Y",
        "Oldpeak": 2.0, "ST_Slope": "Flat",
    },
}

with st.expander("💡 Contoh Data Valid dari Dataset"):
    st.markdown(
        "Pilih salah satu contoh di bawah untuk mengisi formulir secara otomatis. "
        "Data diambil langsung dari dataset *Heart Failure Prediction*."
    )
    selected_sample = st.selectbox(
        "Pilih contoh data:",
        options=["— Pilih contoh —"] + list(SAMPLE_DATA.keys()),
        key="sample_selector",
    )
 
    if selected_sample != "— Pilih contoh —":
        s = SAMPLE_DATA[selected_sample]
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown(f"- **Usia:** {s['Age']} tahun")
            st.markdown(f"- **Jenis Kelamin:** {'Pria' if s['Sex'] == 'M' else 'Wanita'}")
            st.markdown(f"- **Tipe Nyeri Dada:** {s['ChestPainType']}")
            st.markdown(f"- **Tekanan Darah:** {s['RestingBP']} mmHg")
            st.markdown(f"- **Kolesterol:** {s['Cholesterol']} mg/dL")
            st.markdown(f"- **Gula Darah Puasa >120:** {'Ya' if s['FastingBS'] == 1 else 'Tidak'}")
        with col_s2:
            st.markdown(f"- **ECG Istirahat:** {s['RestingECG']}")
            st.markdown(f"- **Detak Jantung Maks:** {s['MaxHR']} bpm")
            st.markdown(f"- **Angina saat Olahraga:** {'Ya' if s['ExerciseAngina'] == 'Y' else 'Tidak'}")
            st.markdown(f"- **Oldpeak:** {s['Oldpeak']}")
            st.markdown(f"- **Kemiringan ST:** {s['ST_Slope']}")
        st.info("ℹ️ Salin nilai-nilai di atas ke formulir di bawah untuk mencoba prediksi.")
        
st.subheader("📋 Data Pasien")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia (tahun)", 20, 100, 50)
        sex = st.selectbox(
            "Jenis Kelamin", ["M", "F"],
            format_func=lambda x: "Pria" if x == "M" else "Wanita",
        )
        chest_pain = st.selectbox(
            "Tipe Nyeri Dada",
            ["ASY", "ATA", "NAP", "TA"],
            format_func=lambda x: {
                "ASY": "ASY – Asimtomatik",
                "ATA": "ATA – Angina Atipikal",
                "NAP": "NAP – Nyeri Non-Anginal",
                "TA":  "TA  – Angina Tipikal",
            }[x],
        )
        resting_bp = st.number_input("Tekanan Darah Istirahat (mmHg)", 60, 250, 120)
        cholesterol = st.number_input(
            "Kolesterol Serum (mg/dL)", 0, 700, 200,
            help="Masukkan 0 jika data kolesterol tidak tersedia",
        )
        fasting_bs = st.selectbox(
            "Gula Darah Puasa > 120 mg/dL", [0, 1],
            format_func=lambda x: "Ya (>120 mg/dL)" if x == 1 else "Tidak (≤120 mg/dL)",
        )

    with col2:
        resting_ecg = st.selectbox(
            "Hasil ECG Istirahat",
            ["Normal", "LVH", "ST"],
            format_func=lambda x: {
                "Normal": "Normal",
                "LVH":    "LVH – Hipertrofi Ventrikel Kiri",
                "ST":     "ST  – Kelainan Gelombang ST-T",
            }[x],
        )
        max_hr = st.number_input("Detak Jantung Maksimum (bpm)", 60, 220, 140)
        ex_angina = st.selectbox(
            "Angina Akibat Olahraga", ["N", "Y"],
            format_func=lambda x: "Tidak" if x == "N" else "Ya",
        )
        oldpeak = st.number_input(
            "Oldpeak (Depresi Segmen ST)", -3.0, 7.0, 0.0, step=0.1,
            help="Nilai numerik depresi ST saat olahraga vs istirahat",
        )
        st_slope = st.selectbox(
            "Kemiringan Segmen ST",
            ["Up", "Flat", "Down"],
            format_func=lambda x: {
                "Up":   "Up   – Naik",
                "Flat": "Flat – Datar",
                "Down": "Down – Turun",
            }[x],
        )

    submitted = st.form_submit_button("🔍 Analisis Risiko", use_container_width=True)


if submitted:
    inputs = {
        "Age": age, "Sex": sex, "ChestPainType": chest_pain,
        "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": ex_angina,
        "Oldpeak": oldpeak, "ST_Slope": st_slope,
    }

    _, model = load_artifacts()
    X        = preprocess(inputs)
    pred     = int(model.predict(X)[0])
    prob     = float(model.predict_proba(X)[0][1])

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    col_res, col_prob = st.columns(2)

    with col_res:
        if pred == 1:
            st.error("### ⚠️ Risiko Tinggi\nTerindikasi penyakit jantung")
        else:
            st.success("### ✅ Risiko Rendah\nTidak terindikasi penyakit jantung")

    with col_prob:
        st.metric("Probabilitas Risiko", f"{prob * 100:.1f}%")
        st.progress(float(prob))
        level = "🔴 Tinggi" if prob >= 0.7 else ("🟡 Sedang" if prob >= 0.4 else "🟢 Rendah")
        st.caption(f"Tingkat risiko: {level}")

    if cholesterol == 0:
        st.info(
            "ℹ️ Kolesterol diisi 0 (tidak tersedia). "
            "Sistem menggunakan median populasi training (237 mg/dL)."
        )

    st.caption(
        "⚕️ **Disclaimer:** Hasil ini hanya untuk skrining awal dan "
        "tidak menggantikan diagnosis medis profesional."
    )


st.divider()

with st.expander("📝 Feedback dari User Testing"):
    st.markdown(
        "Bantu kami meningkatkan sistem ini dengan mengisi survey singkat berikut. "
        "Feedback kamu sangat berarti untuk pengembangan lebih lanjut."
    )
    st.link_button(
        "📋 Isi Survey Feedback (Google Form)",
        GOOGLE_FORM_URL,
        use_container_width=True,
    )
