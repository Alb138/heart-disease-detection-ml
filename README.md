# Heart Disease Risk Detection System

> Sistem Pendukung Keputusan Deteksi Dini Risiko Penyakit Jantung  
> Berbasis **Support Vector Machine (SVM-RBF)**

**Live App:** [heart-disease-detection-ml-app.streamlit.app](https://heart-disease-detection-ml-app.streamlit.app/)

---

## Overview

Proyek ini membangun sebuah *clinical decision support system* berbasis Machine Learning untuk mendeteksi risiko penyakit jantung dari parameter pemeriksaan klinis rutin. Sistem dirancang sebagai **alat bantu second opinion** bagi tenaga medis di klinik, puskesmas, atau fasilitas kesehatan lainnya — bukan untuk self-assessment mandiri oleh pasien.

Pipeline yang dibangun mencakup end-to-end: data preprocessing, feature engineering, model training & evaluation, hingga deployment sebagai web application.

---

## Dataset

**Heart Failure Prediction Dataset** — [fedesoriano (Kaggle, 2021)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

Merupakan gabungan dari 5 dataset klinis independen (Cleveland, Budapest, Long Beach, Switzerland, VA).

| Properti | Nilai |
|----------|-------|
| Total observasi | 918 pasien |
| Fitur prediktor | 11 fitur |
| Target | `HeartDisease` (0 = Sehat, 1 = Penyakit Jantung) |
| Class balance | 508 positif (55.3%) / 410 negatif (44.7%) |
| Missing values (NaN) | 0 |

### Fitur Dataset

| Nama Fitur | Tipe | Deskripsi |
|------------|------|-----------|
| `Age` | Numerik | Usia pasien (tahun) |
| `Sex` | Kategorikal | Jenis kelamin (M/F) |
| `ChestPainType` | Kategorikal | Tipe nyeri dada (ASY/ATA/NAP/TA) |
| `RestingBP` | Numerik | Tekanan darah istirahat (mmHg) |
| `Cholesterol` | Numerik | Kolesterol serum (mg/dL) |
| `FastingBS` | Biner | Gula darah puasa > 120 mg/dL (0/1) |
| `RestingECG` | Kategorikal | Hasil EKG istirahat (Normal/LVH/ST) |
| `MaxHR` | Numerik | Detak jantung maksimum (bpm) |
| `ExerciseAngina` | Kategorikal | Angina akibat olahraga (Y/N) |
| `Oldpeak` | Numerik | Depresi segmen ST (mV) |
| `ST_Slope` | Kategorikal | Kemiringan segmen ST (Up/Flat/Down) |

---

## Preprocessing Pipeline

1. **Handle `RestingBP = 0`** — 1 baris data entry error, diimputasi dengan median (130 mmHg)
2. **MNAR handling pada `Cholesterol = 0`** — 172 baris (18.7%) dikode 0, terbukti *Missing Not At Random* (88.4% adalah pasien positif). Ditangani dengan:
   - Menambah flag biner `Cholesterol_missing`
   - Imputasi median non-zero (237 mg/dL)
3. **One-Hot Encoding** dengan `drop_first=True` → 16 fitur total
4. **Train/Test Split** — 80/20 stratified (`random_state=42`)
5. **StandardScaler** — fit hanya pada `X_train` (tidak ada data leakage)

> **Catatan:** Class imbalance ratio 0.807 tergolong *mild* — SMOTE tidak diperlukan.

---

## Model & Hasil Evaluasi

### Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| **SVM-RBF** | **0.8913** | 0.8727 | **0.9412** | **0.9057** | **0.9460** |
| LR (Baseline) | 0.8913 | 0.8868 | 0.9216 | 0.9038 | 0.9327 |
| Random Forest | 0.8913 | **0.9020** | 0.9020 | 0.9020 | 0.9385 |
| SVM-Linear | 0.8750 | 0.8692 | 0.9118 | 0.8900 | 0.9344 |
| SVM-Polynomial | 0.8696 | 0.8750 | 0.8922 | 0.8835 | 0.9293 |

### Model yang Di-deploy: **SVM-RBF** (C=1, gamma='scale')

- **False Negative (FN) = 6** — hanya 6 dari 102 pasien positif yang tidak terdeteksi
- Hyperparameter tuning via GridSearchCV dilakukan, namun parameter default terbukti lebih baik di test set (*near-optimal for this dataset size*)
- Random Forest digunakan sebagai model pembanding untuk analisis feature importance

---

## Menjalankan Secara Lokal

### Prasyarat
- Python 3.11
- Dataset `heart.csv` dari [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) → taruh di `data/raw/`

### Instalasi

```bash
# 1. Clone repositori
git clone https://github.com/Alb138/heart-disease-detection-ml.git
cd heart-disease-detection-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan preprocessing (opsional, file processed sudah tersedia)
jupyter notebook notebooks/02_preprocessing.ipynb

# 4. Jalankan aplikasi dari ROOT repo
streamlit run app.py
```

Aplikasi akan tersedia di `http://localhost:8501`

### Requirements

```
streamlit
scikit-learn
pandas
numpy
joblib
```
