import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import altair as alt
from fpdf import FPDF
from datetime import datetime
from symptom_mapping import text_to_symptom_vector

# ---------------------------------------------------------
# 0. AUTO-TRAIN JIKA MODEL BELUM ADA
# ---------------------------------------------------------
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(BASE_DIR, "models/feature_cols.pkl")):
    with st.spinner("⏳ Sedang menyiapkan model AI, mohon tunggu..."):
        subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "train_models.py")],
            cwd=BASE_DIR,  # jalankan dari folder Kode/
            check=True
        )

# ---------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CSS
# ---------------------------------------------------------
st.set_page_config(page_title="HealthCare - AI Health", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    .stApp { background-color: #0e1117; font-family: 'Plus Jakarta Sans', sans-serif; }
    
    /* HERO SECTION (Tampilan Awal) */
    .hero-container {
        text-align: center;
        padding: 60px 20px;
        background: linear-gradient(180deg, rgba(32,44,51,0) 0%, rgba(32,44,51,0.5) 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        border: 1px solid #1f2937;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        color: #a0aec0;
        margin-bottom: 30px;
    }
    
    /* CHAT BUBBLES STYLE */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarIcon-user"]) {
        flex-direction: row-reverse; text-align: right;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarIcon-user"]) div[data-testid="stMarkdownContainer"] {
        background-color: #005c4b; color: #fff; padding: 10px 15px; 
        border-radius: 15px 0px 15px 15px; max-width: 80%;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarIcon-assistant"]) div[data-testid="stMarkdownContainer"] {
        background-color: #202c33; color: #fff; padding: 10px 15px; 
        border-radius: 0px 15px 15px 15px; max-width: 80%; border: 1px solid #374151;
    }
    .stChatMessageAvatarBackground { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. LOAD MODEL & DATA
# ---------------------------------------------------------
@st.cache_resource
def load_models_and_data():
    try:
        import os
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # Lalu ubah semua path
        df = pd.read_csv(os.path.join(BASE_DIR, "data/dataset_bayes_randomforest.csv"))
        model_nb = joblib.load(os.path.join(BASE_DIR, "models/model_nb_gejala.pkl"))
        model_rf = joblib.load(os.path.join(BASE_DIR, "models/model_rf_gejala.pkl"))
        feature_cols = joblib.load(os.path.join(BASE_DIR, "models/feature_cols.pkl"))
        
        label_col = "Label_Penyakit"
        disease_name_col = "Nama_Penyakit"
        
        label_to_name = dict(df[[label_col, disease_name_col]].drop_duplicates().values)
        
        return df, model_nb, model_rf, feature_cols, label_col, label_to_name
    except Exception as e:
        st.error(f"System Error (Load Data): {e}")
        return None, None, None, [], None, {}

df, model_nb, model_rf, feature_cols, label_col, label_to_name = load_models_and_data()  # [file:309]

# ---------------------------------------------------------
# 3. FUNGSI LOGIKA (BACKEND)
# ---------------------------------------------------------
def get_all_probabilities(symptom_vector, method="Random Forest"):
    X = pd.DataFrame([symptom_vector], columns=feature_cols)
    model = model_nb if method == "Naive Bayes" else model_rf
    proba = model.predict_proba(X)[0]
    classes = model.classes_
    
    results = []
    for i, label in enumerate(classes):
        name = label_to_name.get(label, str(label))
        results.append({"name": name, "label": label, "prob": float(proba[i])})
    
    results.sort(key=lambda x: x["prob"], reverse=True)
    return results  # [file:309]

def filter_valid_diseases(symptom_vector):
    active_idx = [i for i, v in enumerate(symptom_vector) if v == 1]
    if not active_idx:
        return df.copy()
    valid_rows = []
    for _, row in df.iterrows():
        if any(row[feature_cols[i]] == 1 for i in active_idx):
            valid_rows.append(row)
    return pd.DataFrame(valid_rows)

def dampen_confidence(prob, symptom_count):
    if symptom_count <= 1:
        return prob * 0.35
    elif symptom_count == 2:
        return prob * 0.6
    return prob

def choose_next_symptom(candidates, current_vector, asked_set):
    rows = df[df[label_col].isin([c["label"] for c in candidates])]
    if rows.empty:
        return None

    scores = {}
    for col in feature_cols:
        if col in asked_set:
            continue
        idx = feature_cols.index(col)
        if current_vector[idx] == 1:
            continue
        freq = rows[col].sum()
        if freq == 0 or freq == len(rows):
            continue
        scores[col] = freq / len(rows)

    if not scores:
        return None

    best_symptom = min(scores.items(), key=lambda x: abs(x[1] - 0.5))[0]
    return best_symptom  # [file:309]

def create_pdf(patient_data, results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Laporan Hasil Diagnosa AI", ln=True, align='C')
    
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt=f"Waktu: {datetime.now().strftime('%d-%m-%Y %H:%M')}", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Metode: {patient_data['method']}", ln=True)
    pdf.ln(5)
    
    for i, res in enumerate(results[:3], 1):
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(0, 10, txt=f"{i}. {res['name']} (Keyakinan: {int(res['prob']*100)}%)", ln=True)
        
        row = df[df["Nama_Penyakit"] == res['name']].iloc[0]
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, txt=f"   Saran: {row.get('Rekomendasi_Mandiri', '-')}")
        pdf.multi_cell(0, 6, txt=f"   Obat: {row.get('Obat_Cocok', '-')}")
        pdf.ln(2)
        
    return pdf.output(dest='S').encode('latin-1')

# ---------------------------------------------------------
# 4. SIDEBAR & STATE MANAGEMENT
# ---------------------------------------------------------
with st.sidebar:
    st.title("🩺 HealthCare")
    st.caption("AI Health Assistant")
    st.divider()
    
    selected_method = st.radio("Metode Analisa:", ["Random Forest", "Naive Bayes"])
    st.info(f"Menggunakan: **{selected_method}**")
    
    st.divider()
    if st.button("🔄 Reset / Mulai Baru", use_container_width=True):
        st.session_state.clear()
        st.rerun()

if "current_method" not in st.session_state:
    st.session_state.current_method = selected_method
if st.session_state.current_method != selected_method:
    st.session_state.clear()
    st.rerun()
st.session_state.method = selected_method

# Inisialisasi Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Halo! Saya HealthCare. Saya akan membantu menganalisis gejala dengan **{selected_method}**. Apa keluhan yang kamu rasakan?"
    }]
if "mode" not in st.session_state:
    st.session_state.mode = "awal"
if "symptom_vector" not in st.session_state:
    st.session_state.symptom_vector = [0] * len(feature_cols)
if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = set()
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "last_asked_symptom" not in st.session_state:
    st.session_state.last_asked_symptom = None

# ---------------------------------------------------------
# 5. HERO SECTION
# ---------------------------------------------------------
input_override = None

if len(st.session_state.messages) == 1:
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Chatbot Kesehatan AI</div>
            <div class="hero-subtitle">Deteksi Gejala • Analisa Cerdas • Solusi Medis</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🤒 Saya Demam Tinggi", use_container_width=True):
            input_override = "Saya demam tinggi dan badan panas"
    with col2:
        if st.button("🤧 Batuk & Pilek", use_container_width=True):
            input_override = "Saya batuk batuk dan hidung tersumbat"
    with col3:
        if st.button("🤕 Sakit Kepala", use_container_width=True):
            input_override = "Kepala saya sakit sekali dan pusing"

# ---------------------------------------------------------
# 6. RENDER HISTORY CHAT
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg.get("is_result", False):
            chart_df = pd.DataFrame(msg["results"][:3])
            chart_df["Persentase"] = chart_df["prob"]
            chart_df.rename(columns={"name": "Penyakit"}, inplace=True)
            
            c = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('Persentase', axis=None),
                y=alt.Y('Penyakit', sort='-x'),
                color=alt.Color('Persentase', scale=alt.Scale(scheme='greens'), legend=None),
                tooltip=['Penyakit', alt.Tooltip('Persentase', format='.1%')]
            ).properties(height=150)
            st.altair_chart(c, use_container_width=True)
            
            if msg.get("pdf_data"):
                st.download_button(
                    label="📄 Download Laporan PDF",
                    data=msg["pdf_data"],
                    file_name=f"Diagnosa_{int(time.time())}.pdf",
                    mime="application/pdf",
                    key=f"dl_{msg['timestamp']}"
                )

# ---------------------------------------------------------
# 7. INPUT USER
# ---------------------------------------------------------
user_input = st.chat_input("Ketik keluhanmu di sini...")
final_input = input_override if input_override else user_input

# ---------------------------------------------------------
# 8. LOGIC PROSES INPUT USER (DIPERBAIKI)
# ---------------------------------------------------------
if final_input:
    if not input_override:
        st.chat_message("user").markdown(final_input)
    
    st.session_state.messages.append({"role": "user", "content": final_input})
    
    response_text = ""
    is_final = False
    final_results = []
    current_vector = list(st.session_state.symptom_vector)
    
    # MODE 1: teks bebas
    if st.session_state.mode == "awal":
        vec_update, flags = text_to_symptom_vector(final_input)
        vec_update = np.array(vec_update).ravel()
        for i, val in enumerate(vec_update):
            if val == 1:
                current_vector[i] = 1
        for k, v in flags.items():
            if v == 1:
                st.session_state.asked_questions.add(k)
    
    # MODE 2: jawab ya/tidak
    elif st.session_state.mode == "tanya":
        last_q = st.session_state.last_asked_symptom
        ans = final_input.lower()
        if any(x in ans for x in ['ya', 'y', 'iya', 'betul', 'benar']):
            if last_q in feature_cols:
                idx = feature_cols.index(last_q)
                current_vector[idx] = 1
        elif any(x in ans for x in ['tidak', 'gak', 'enggak', 'bukan', 'no']):
            pass
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Maaf, mohon jawab **ya** atau **tidak** supaya analisanya tepat."
            })
            st.rerun()
    
    st.session_state.symptom_vector = current_vector
    symptom_count = int(sum(current_vector))
    
    valid_df = filter_valid_diseases(current_vector)
    valid_labels = set(valid_df[label_col])
    
    raw_probs = get_all_probabilities(current_vector, st.session_state.method)
    all_probs = []
    for p in raw_probs:
        if p["label"] in valid_labels:
            p["prob"] = dampen_confidence(p["prob"], symptom_count)
            all_probs.append(p)
    all_probs.sort(key=lambda x: x["prob"], reverse=True)
    
    if not all_probs:
        response_text = "Untuk gejala seperti ini, datanya belum cukup. Coba jelaskan lebih detail."
        st.session_state.mode = "awal"
    else:
        top_conf = all_probs[0]["prob"]
        
        # KEPUTUSAN: STOP ATAU LANJUT
        if symptom_count == 0:
            response_text = (
                "Saya belum menangkap gejala medis yang spesifik. "
                "Coba jelaskan lebih detail, misalnya demam, batuk, sesak napas, nyeri, dan sebagainya."
            )
            st.session_state.mode = "awal"
        
        else:
            if symptom_count == 1:
                can_finalize = False
            elif symptom_count == 2:
                can_finalize = (top_conf > 0.85) or (st.session_state.question_count >= 10)
            else:
                can_finalize = (top_conf > 0.75) or (st.session_state.question_count >= 8)
            
            if can_finalize:
                is_final = True
                final_results = all_probs
                lines = []
                for r in all_probs[:3]:
                    pct = int(r['prob'] * 100)
                    d_row = df[df[label_col] == r['label']].iloc[0]
                    obat = d_row.get('Obat_Cocok', '-')
                    lines.append(f"**{r['name']}** (~{pct}%)\n   💊 *Rekomendasi awal: {obat}*")
                
                opening = (
                    f"Saat ini pola gejala paling mendekati **{all_probs[0]['name']}** "
                    f"dengan kemungkinan sekitar **{int(top_conf*100)}%**.\n\n"
                    "Namun masih ada beberapa kemungkinan lain:"
                )
                response_text = (
                    "### 🩺 Hasil Analisa Sementara\n"
                    f"{opening}\n\n" +
                    "\n\n".join(lines) +
                    "\n\nIni hanya alat bantu awal, bukan pengganti pemeriksaan dokter ya."
                )
                
                st.session_state.mode = "awal"
                st.session_state.question_count = 0
                st.session_state.asked_questions = set()
                st.session_state.symptom_vector = [0] * len(feature_cols)
            
            else:
                # LANJUT NANYA: gunakan beberapa kandidat sekaligus
                k = 5
                candidates = all_probs[:k]
                st.session_state.mode = "tanya"
                st.session_state.question_count += 1
                
                next_q = choose_next_symptom(candidates, current_vector, st.session_state.asked_questions)
                
                if next_q:
                    st.session_state.last_asked_symptom = next_q
                    st.session_state.asked_questions.add(next_q)
                    clean_name = next_q.replace("_", " ")
                    nama_calon = ", ".join(sorted({c["name"] for c in candidates[:3]}))
                    if symptom_count == 1:
                        pre = (
                            "Dari keluhan awal, ada beberapa kemungkinan seperti "
                            f"**{nama_calon}**. Untuk memperjelas, "
                        )
                    else:
                        pre = (
                            "Baik. "
                            "Supaya tidak salah diagnosa, "
                        )
                    response_text = (
                        pre +
                        f"apakah kamu juga merasakan **{clean_name}**? (ya/tidak)"
                    )
                else:
                    is_final = True
                    final_results = all_probs
                    response_text = (
                        "Informasi gejala yang kamu berikan sudah cukup untuk gambaran awal. "
                        "Berikut beberapa kemungkinan teratas:\n\n" +
                        "\n".join([
                            f"- **{r['name']}** (~{int(r['prob']*100)}%)"
                            for r in all_probs[:3]
                        ]) +
                        "\n\nTetap konsultasikan ke tenaga kesehatan untuk pemeriksaan pasti."
                    )
                    st.session_state.mode = "awal"
                    st.session_state.question_count = 0
                    st.session_state.asked_questions = set()
                    st.session_state.symptom_vector = [0] * len(feature_cols)
    
    msg_data = {
        "role": "assistant",
        "content": response_text,
        "timestamp": time.time()
    }
    if is_final:
        msg_data["is_result"] = True
        msg_data["results"] = final_results
        msg_data["pdf_data"] = create_pdf({"method": st.session_state.method}, final_results)
    
    st.session_state.messages.append(msg_data)
    st.rerun()
