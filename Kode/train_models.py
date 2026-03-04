import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ======================================================
# SETUP
# ======================================================
os.makedirs("models", exist_ok=True)

DATASET_PATH = "data/dataset_bayes_randomforest.csv"
MODEL_DIR = "models"
NOISE_RATE = 0.1

# ======================================================
# LOAD DATASET
# ======================================================
df = pd.read_csv(DATASET_PATH)

# ======================================================
# DEFINISI KOLOM
# ======================================================
meta_cols = [
    "ID","Nama_Penyakit","Kategori","Tingkat_Keparahan",
    "Obat_Cocok","Rekomendasi_Mandiri","Kapan_ke_Dokter",
    "prevalence_score","Probabilitas_Penyakit",
    "Kategori_Enc","Keparahan_Enc","Label_Penyakit","score_bayes"
]

feature_cols = [c for c in df.columns if c not in meta_cols]

X = df[feature_cols]
y = df["Label_Penyakit"]

print(f"Jumlah fitur gejala : {len(feature_cols)}")
print(f"Jumlah penyakit     : {y.nunique()}")

# ======================================================
# TAMBAH NOISE (SIMULASI DATA DUNIA NYATA)
# ======================================================
def add_noise(X, noise_rate=0.1):
    X_noisy = X.copy()
    for col in X.columns:
        mask = np.random.rand(len(X)) < noise_rate
        X_noisy.loc[mask, col] = 1 - X_noisy.loc[mask, col]
    return X_noisy

X = add_noise(X, noise_rate=NOISE_RATE)

# ======================================================
# SPLIT DATA (STRATIFIED)
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# MODEL 1 — BERNOULLI NAIVE BAYES
# ======================================================
nb = BernoulliNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

print("\n================ Bernoulli Naive Bayes ================")
print(classification_report(
    y_test,
    y_pred_nb,
    zero_division=0
))

# ======================================================
# MODEL 2 — RANDOM FOREST (IMBALANCE SAFE)
# ======================================================
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n================== Random Forest ==================")
print(classification_report(
    y_test,
    y_pred_rf,
    zero_division=0
))

# ======================================================
# SIMPAN MODEL & METADATA
# ======================================================
joblib.dump(nb, f"{MODEL_DIR}/model_nb_gejala.pkl")
joblib.dump(rf, f"{MODEL_DIR}/model_rf_gejala.pkl")
joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")

label_map = (
    df[["Label_Penyakit", "Nama_Penyakit"]]
    .drop_duplicates()
    .set_index("Label_Penyakit")["Nama_Penyakit"]
    .to_dict()
)

joblib.dump(label_map, f"{MODEL_DIR}/label_map.pkl")

print("\n✅ TRAINING SELESAI")
print(f"✅ Noise rate       : {NOISE_RATE}")
print("✅ Model tersimpan  : /models")
