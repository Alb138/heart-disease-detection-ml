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

# ── Google Fonts + Global CSS ─────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&family=Lora:wght@600;700&display=swap" rel="stylesheet">

<style>
/* ── Reset & base ─────────────────────────────── */
[data-testid="stHeaderActionElements"] { display: none !important; }
.stApp { background: #FAF7F5; }
.block-container { padding-top: 1.5rem !important; max-width: 820px !important; }
html, body, .stApp { font-family: 'DM Sans', sans-serif; }

/* ── Submit button ────────────────────────────── */
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #B91C1C 0%, #991B1B 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 1.05rem !important; letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important; font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFormSubmitButton"] > button:hover { opacity: 0.88 !important; }

/* ── Link button ──────────────────────────────── */
.stLinkButton a {
    background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important;
    color: white !important; border-radius: 10px !important;
    border: none !important; font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Progress bar ─────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #EF4444, #B91C1C) !important;
}

/* ── Metric card ──────────────────────────────── */
[data-testid="metric-container"] {
    background: white !important; border-radius: 12px !important;
    padding: 1rem !important; border: 1px solid #F3E8E8 !important;
    box-shadow: 0 2px 8px rgba(185,28,28,0.08) !important;
}
[data-testid="stMetricValue"] { color: #B91C1C !important; font-weight: 700 !important; }

/* ── Expanders ────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #EDE9E7 !important; border-radius: 14px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04) !important;
    background: white !important; overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important; font-family: 'DM Sans', sans-serif !important;
}

/* ── st.info ──────────────────────────────────── */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 12px !important;
}

/* ── Selectbox & number input labels ─────────── */
[data-testid="stWidgetLabel"] { font-weight: 500 !important; color: #3D3330 !important; }

/* ── Form background ──────────────────────────── */
[data-testid="stForm"] {
    background: white !important; border-radius: 16px !important;
    padding: 1.5rem !important; border: 1px solid #F0EBE8 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
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

# ── Path resolution ───────────────────────────────────────────────────────────
def _find(filename, candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]

MODEL_PATH  = _find('svm_model.pkl',  ['svm_model.pkl', os.path.join('models', 'svm_model.pkl')])
SCALER_PATH = _find('scaler.joblib',  ['scaler.joblib', os.path.join('data', 'scaler.joblib'),
                                        os.path.join('data', 'processed', 'scaler.joblib')])

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
        'Age': float(inputs['Age']), 'RestingBP': float(inputs['RestingBP']),
        'Cholesterol': chol_val, 'FastingBS': float(inputs['FastingBS']),
        'MaxHR': float(inputs['MaxHR']), 'Oldpeak': float(inputs['Oldpeak']),
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

# ── Helper UI components ──────────────────────────────────────────────────────
def section_header(icon, title):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:0.25rem 0 1rem;">
      <div style="width:4px;height:28px;background:linear-gradient(180deg,#EF4444,#B91C1C);
                  border-radius:3px;flex-shrink:0;"></div>
      <span style="font-family:'Lora',serif;font-size:1.2rem;font-weight:700;
                   color:#1C1917;">{icon} {title}</span>
    </div>
    """, unsafe_allow_html=True)

def colored_divider():
    st.markdown("""
    <div style="height:1.5px;background:linear-gradient(90deg,#EF4444,#FCA5A5,transparent);
                border-radius:1px;margin:1.8rem 0;"></div>
    """, unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #7F1D1D 0%, #B91C1C 45%, #EF4444 100%);
    padding: 2rem 2.5rem 1.8rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(185,28,28,0.28);
    position: relative; overflow: hidden;
">
  <div style="position:absolute;top:-30px;right:-30px;width:160px;height:160px;
              background:rgba(255,255,255,0.06);border-radius:50%;"></div>
  <div style="position:absolute;bottom:-50px;right:60px;width:100px;height:100px;
              background:rgba(255,255,255,0.04);border-radius:50%;"></div>
  <div style="display:flex;align-items:center;gap:1.25rem;position:relative;z-index:1;">
    <span style="font-size:3.5rem;filter:drop-shadow(0 2px 8px rgba(0,0,0,0.3));">🫀</span>
    <div>
      <h1 style="font-family:'Lora',serif;color:white;margin:0;font-size:1.75rem;
                 font-weight:700;line-height:1.25;text-shadow:0 1px 4px rgba(0,0,0,0.2);">
        Deteksi Dini Risiko Penyakit Jantung
      </h1>
      <p style="color:rgba(255,255,255,0.82);margin:0.45rem 0 0;font-size:0.9rem;
                font-family:'DM Sans',sans-serif;font-weight:400;">
        Sistem Pendukung Keputusan · Support Vector Machine (SVM-RBF)
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Contoh data valid ─────────────────────────────────────────────────────────
with st.expander("💡 Contoh Data Valid dari Dataset"):
    st.markdown(
        "Pilih salah satu contoh di bawah untuk referensi pengisian formulir. "
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
            st.markdown(f"- **Jenis Kelamin:** {'Pria' if s['Sex']=='M' else 'Wanita'}")
            st.markdown(f"- **Tipe Nyeri Dada:** {s['ChestPainType']}")
            st.markdown(f"- **Tekanan Darah:** {s['RestingBP']} mmHg")
            st.markdown(f"- **Kolesterol:** {s['Cholesterol']} mg/dL")
            st.markdown(f"- **Gula Darah Puasa >120:** {'Ya' if s['FastingBS']==1 else 'Tidak'}")
        with col_s2:
            st.markdown(f"- **ECG Istirahat:** {s['RestingECG']}")
            st.markdown(f"- **Detak Jantung Maks:** {s['MaxHR']} bpm")
            st.markdown(f"- **Angina saat Olahraga:** {'Ya' if s['ExerciseAngina']=='Y' else 'Tidak'}")
            st.markdown(f"- **Oldpeak:** {s['Oldpeak']}")
            st.markdown(f"- **Kemiringan ST:** {s['ST_Slope']}")
        st.info("ℹ️ Salin nilai-nilai di atas ke formulir di bawah untuk mencoba prediksi.")

colored_divider()
section_header("📋", "Data Pasien")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Usia (tahun)", 20, 100, 50,
            help="Usia pasien dalam tahun. Risiko penyakit jantung meningkat seiring usia.")
        sex = st.selectbox("Jenis Kelamin", ["M","F"],
            format_func=lambda x: "Pria" if x=="M" else "Wanita",
            help="Pria umumnya memiliki risiko lebih tinggi pada usia yang sama.")
        chest_pain = st.selectbox("Tipe Nyeri Dada", ["ASY","ATA","NAP","TA"],
            format_func=lambda x: {"ASY":"ASY – Asimtomatik","ATA":"ATA – Angina Atipikal",
                                    "NAP":"NAP – Nyeri Non-Anginal","TA":"TA – Angina Tipikal"}[x],
            help="ASY (tanpa nyeri dada) adalah tipe paling berisiko. TA adalah angina klasik.")
        resting_bp = st.number_input("Tekanan Darah Istirahat (mmHg)", 60, 250, 120,
            help="Normal: 90–120 mmHg. Hipertensi (>130) meningkatkan risiko.")
        cholesterol = st.number_input("Kolesterol Serum (mg/dL)", 0, 700, 200,
            help="Normal <200. Tinggi ≥240. Isi 0 jika tidak tersedia (sistem pakai median 237).")
        fasting_bs = st.selectbox("Gula Darah Puasa > 120 mg/dL", [0,1],
            format_func=lambda x: "Ya (>120 mg/dL)" if x==1 else "Tidak (≤120 mg/dL)",
            help=">120 mg/dL mengindikasikan kemungkinan diabetes — faktor risiko signifikan.")
    with col2:
        resting_ecg = st.selectbox("Hasil ECG Istirahat", ["Normal","LVH","ST"],
            format_func=lambda x: {"Normal":"Normal","LVH":"LVH – Hipertrofi Ventrikel Kiri",
                                    "ST":"ST – Kelainan Gelombang ST-T"}[x],
            help="LVH = penebalan dinding jantung. ST = kelainan konduksi listrik jantung.")
        max_hr = st.number_input("Detak Jantung Maksimum (bpm)", 60, 220, 140,
            help="Nilai lebih rendah dari (220 - usia) dapat mengindikasikan gangguan fungsi jantung.")
        ex_angina = st.selectbox("Angina Akibat Olahraga", ["N","Y"],
            format_func=lambda x: "Tidak" if x=="N" else "Ya",
            help="Nyeri dada saat aktivitas fisik adalah tanda kuat penyempitan arteri koroner.")
        oldpeak = st.number_input("Oldpeak (Depresi Segmen ST)", -3.0, 7.0, 0.0, step=0.1,
            help="Depresi ST pada EKG saat olahraga vs istirahat (mV). Nilai >0 mengindikasikan iskemia.")
        st_slope = st.selectbox("Kemiringan Segmen ST", ["Up","Flat","Down"],
            format_func=lambda x: {"Up":"Up – Naik","Flat":"Flat – Datar","Down":"Down – Turun"}[x],
            help="Up = normal. Flat = berpotensi abnormal. Down = risiko tinggi iskemia.")

    submitted = st.form_submit_button("🔍  Analisis Risiko", use_container_width=True)

# ── Prediction result ─────────────────────────────────────────────────────────
if submitted:
    inputs = {
        "Age": age, "Sex": sex, "ChestPainType": chest_pain,
        "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": ex_angina,
        "Oldpeak": oldpeak, "ST_Slope": st_slope,
    }
    _, model = load_artifacts()
    X    = preprocess(inputs)
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    colored_divider()
    section_header("📊", "Hasil Prediksi")

    col_res, col_prob = st.columns([1.1, 1])

    with col_res:
        if pred == 1:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #7F1D1D, #B91C1C, #EF4444);
                border-radius: 16px; padding: 1.75rem 1.5rem;
                text-align: center;
                box-shadow: 0 6px 24px rgba(185,28,28,0.32);
            ">
              <div style="font-size:2.8rem;margin-bottom:0.5rem;">⚠️</div>
              <div style="font-family:'Lora',serif;font-size:1.35rem;font-weight:700;
                          color:white;margin-bottom:0.3rem;">Risiko Tinggi</div>
              <div style="color:rgba(255,255,255,0.82);font-size:0.9rem;">
                Terindikasi penyakit jantung
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #14532D, #15803D, #16A34A);
                border-radius: 16px; padding: 1.75rem 1.5rem;
                text-align: center;
                box-shadow: 0 6px 24px rgba(21,128,61,0.30);
            ">
              <div style="font-size:2.8rem;margin-bottom:0.5rem;">✅</div>
              <div style="font-family:'Lora',serif;font-size:1.35rem;font-weight:700;
                          color:white;margin-bottom:0.3rem;">Risiko Rendah</div>
              <div style="color:rgba(255,255,255,0.82);font-size:0.9rem;">
                Tidak terindikasi penyakit jantung
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_prob:
        level_color = "#B91C1C" if prob >= 0.7 else ("#D97706" if prob >= 0.4 else "#15803D")
        level_label = "Tinggi" if prob >= 0.7 else ("Sedang" if prob >= 0.4 else "Rendah")
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1.5rem;
                    border:1px solid #F0EBE8;box-shadow:0 4px 16px rgba(0,0,0,0.06);
                    height:100%;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;text-align:center;gap:0.6rem;">
          <div style="color:#78716C;font-size:0.85rem;font-weight:500;
                      text-transform:uppercase;letter-spacing:0.06em;">Probabilitas Risiko</div>
          <div style="font-family:'Lora',serif;font-size:2.8rem;font-weight:700;
                      color:{level_color};line-height:1;">{prob*100:.1f}%</div>
          <div style="width:100%;background:#F5F0EE;border-radius:6px;height:8px;overflow:hidden;">
            <div style="width:{prob*100:.1f}%;background:linear-gradient(90deg,{level_color},{level_color}99);
                        height:100%;border-radius:6px;transition:width 0.6s;"></div>
          </div>
          <div style="background:{level_color}18;color:{level_color};padding:4px 14px;
                      border-radius:20px;font-size:0.82rem;font-weight:600;">
            Risiko {level_label}
          </div>
        </div>
        """, unsafe_allow_html=True)

    if cholesterol == 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("ℹ️ Kolesterol diisi 0 (tidak tersedia). Sistem menggunakan median populasi training (237 mg/dL).")

    st.markdown("""
    <div style="margin-top:1.2rem;padding:0.9rem 1.2rem;background:#FDF8F6;
                border-radius:10px;border-left:3px solid #D6CCC8;">
      <span style="color:#78716C;font-size:0.82rem;">
        ⚕️ <strong>Disclaimer:</strong> Hasil ini hanya untuk skrining awal dan
        tidak menggantikan diagnosis medis profesional. Konsultasikan dengan dokter
        untuk evaluasi lebih lanjut.
      </span>
    </div>
    """, unsafe_allow_html=True)

# ── Tentang Model ─────────────────────────────────────────────────────────────
colored_divider()
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    <div style="margin-bottom:1rem;">
      <span style="font-family:'Lora',serif;font-size:1rem;font-weight:700;color:#1C1917;">
        Support Vector Machine — RBF Kernel
      </span>
      <span style="background:#FEF2F2;color:#B91C1C;padding:3px 10px;border-radius:20px;
                   font-size:0.78rem;font-weight:600;margin-left:8px;">Model Utama</span>
    </div>
    <p style="color:#57534E;font-size:0.9rem;margin-bottom:1.2rem;">
      Dataset: <strong>Heart Failure Prediction</strong> — fedesoriano (Kaggle) · 
      918 pasien · 11 fitur klinis · target biner (Sehat / Penyakit Jantung)
    </p>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        (m1, "F1-Score",  "0.9057", "#B91C1C"),
        (m2, "ROC-AUC",   "0.9460", "#1D4ED8"),
        (m3, "Recall",    "0.9412", "#15803D"),
        (m4, "Akurasi",   "0.8913", "#7C3AED"),
        (m5, "Precision", "0.8727", "#B45309"),
    ]
    for col, label, val, color in metrics:
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:0.9rem 0.5rem;
                        border-top:3px solid {color};text-align:center;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06);">
              <div style="font-size:1.3rem;font-weight:700;color:{color};
                          font-family:'Lora',serif;">{val}</div>
              <div style="font-size:0.72rem;color:#78716C;margin-top:3px;font-weight:500;">
                {label}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ── Feedback ──────────────────────────────────────────────────────────────────
colored_divider()
with st.expander("📝 Feedback dari User Testing"):
    st.markdown(
        "Bantu kami meningkatkan sistem ini dengan mengisi survey singkat berikut. "
        "Feedback kamu sangat berarti untuk pengembangan lebih lanjut."
    )
    st.link_button(
        "📋  Isi Survey Feedback (Google Form)",
        GOOGLE_FORM_URL,
        use_container_width=True,
    )
