import re
import numpy as np
import joblib
import os

# ===============================
# LOAD FEATURE MODEL
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
feature_cols = joblib.load(os.path.join(BASE_DIR, "models/feature_cols.pkl"))

def normalize(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# ===============================
# MASTER SYMPTOM REGEX (REAL NLP)
# key HARUS SAMA dengan feature_cols
# ===============================
RAW_SYMPTOMS = {

    # ================= GIGI & MULUT =================
    "nyeri_gigi": [
        r"sakit gigi",
        r"gigi sakit",
        r"nyeri gigi",
        r"gigi nyeri",
        r"cenut",
        r"cenut cenut",
        r"ngilu",
        r"gigi berdenyut",
        r"sakit di gigi"
    ],
    "gusi_bengkak": [
        r"gusi bengkak",
        r"gusi sakit",
        r"gusi nyeri"
    ],
    "bau_mulut": [
        r"bau mulut",
        r"mulut bau",
        r"nafas bau"
    ],

    # ================= DEMAM =================
    "demam": [
        r"demam",
        r"panas",
        r"panas tinggi",
        r"meriang",
        r"badan panas"
    ],

    # ================= PERNAPASAN =================
    "batuk": [
        r"batuk",
        r"batuk terus",
        r"batuk parah"
    ],
    "batuk_kering": [
        r"batuk kering",
        r"batuk tanpa dahak"
    ],
    "pilek": [
        r"pilek",
        r"hidung meler",
        r"ingusan"
    ],
    "sesak_napas": [
        r"sesak napas",
        r"sulit bernapas",
        r"nafas berat"
    ],

    # ================= KEPALA =================
    "sakit_kepala": [
        r"sakit kepala",
        r"kepala sakit",
        r"pusing",
        r"pening"
    ],

    # ================= OTOT =================
    "nyeri_otot": [
        r"nyeri otot",
        r"pegal",
        r"badan pegal",
        r"linu"
    ],

    # ================= PENCERNAAN =================
    "mual": [
        r"mual",
        r"enek"
    ],
    "muntah": [
        r"muntah",
        r"muntaber"
    ],
    "diare": [
        r"diare",
        r"mencret",
        r"bab cair"
    ],
    "sakit_perut": [
        r"sakit perut",
        r"perut sakit",
        r"nyeri perut"
    ],

    # ================= KULIT =================
    "ruam": [
        r"ruam",
        r"bintik merah",
        r"bercak merah"
    ],
    "gatal": [
        r"gatal",
        r"kulit gatal"
    ],

    # ================= UMUM =================
    "lemas": [
        r"lemas",
        r"tidak bertenaga",
        r"lelah",
        r"capek"
    ],
    "kehilangan_nafsu_makan": [
        r"tidak nafsu makan",
        r"hilang nafsu makan",
        r"susah makan"
    ]
}

# ===============================
# FILTER ONLY FEATURE MODEL
# ===============================
SYMPTOM_MAP = {
    k: v for k, v in RAW_SYMPTOMS.items()
    if k in feature_cols
}

# ===============================
# NEGATION PATTERN
# ===============================
NEGATION_WORDS = r"(tidak|ga|gak|enggak|bukan|tanpa|tak|nggak|no)\s+"

# ===============================
def text_to_symptom_vector(text: str):
    text = normalize(text)

    flags = {col: 0 for col in feature_cols}

    for symptom, patterns in SYMPTOM_MAP.items():
        for p in patterns:
            match = re.search(p, text)
            if match:
                # Cek apakah ada kata negasi sebelum symptom
                start = match.start()
                prefix = text[max(0, start-20):start]  # ambil 20 karakter sebelum keyword
                if re.search(NEGATION_WORDS + r"$", prefix):
                    flags[symptom] = 0  # dinegasi, abaikan
                else:
                    flags[symptom] = 1
                break

    x = np.array([[flags[col] for col in feature_cols]])
    return x, flags