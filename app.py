"""
Bac Prediction ML — Interface analytique dark
Author: Martine Ouedraogo
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Bac Prediction ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CSS — DARK ANALYTIQUE
# =====================================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap" rel="stylesheet">

<style>
/* ── Variables ────────────────────────────────────────────────────── */
:root {
    --bg:       #060912;
    --surf:     #0c1018;
    --surf2:    #111827;
    --border:   rgba(0,212,255,.12);
    --cyan:     #00d4ff;
    --green:    #00ff88;
    --red:      #ff4d6d;
    --amber:    #fbbf24;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --dim:      #1e293b;
}

/* ── Keyframes ────────────────────────────────────────────────────── */
@keyframes fadeUp   { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:none} }
@keyframes barFill  { from{width:0} to{width:var(--w)} }
@keyframes popIn    { 0%{transform:scale(.88);opacity:0} 70%{transform:scale(1.02)} 100%{transform:scale(1);opacity:1} }
@keyframes glow     { 0%,100%{box-shadow:0 0 10px rgba(0,212,255,.15)} 50%{box-shadow:0 0 28px rgba(0,212,255,.35),0 0 56px rgba(0,212,255,.1)} }
@keyframes pulse    { 0%,100%{opacity:1} 50%{opacity:.55} }
@keyframes scanline { from{transform:translateY(-100%)} to{transform:translateY(200%)} }

/* ── Global ───────────────────────────────────────────────────────── */
html,body,.stApp {
    font-family:'Times New Roman',serif !important;
    background-color:var(--bg) !important;
    color:var(--text) !important;
}
.stApp {
    background-image:radial-gradient(rgba(0,212,255,.05) 1px,transparent 1px) !important;
    background-size:28px 28px !important;
}
header[data-testid="stHeader"] { display:none !important; }
#MainMenu,footer { visibility:hidden; }
hr { border-color:rgba(0,212,255,.1) !important; }
h1,h2,h3,h4 { font-family:'Times New Roman',serif !important; color:var(--text) !important; }
p   { color:#94a3b8; }
label { color:var(--muted) !important; font-weight:500 !important; font-size:.84rem !important; }

/* ── Sidebar ──────────────────────────────────────────────────────── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color:#070a12 !important;
    border-right:1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color:var(--muted) !important; font-family:'Times New Roman',serif !important; }
section[data-testid="stSidebar"] hr { border-color:#1e293b !important; }

section[data-testid="stSidebar"] .stRadio > label { display:none !important; }
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
    display:flex !important; align-items:center !important;
    padding:.5rem .9rem !important; border-radius:6px !important;
    border-left:2px solid transparent !important;
    font-size:.86rem !important; font-weight:500 !important;
    color:#374151 !important; cursor:pointer !important;
    transition:all .15s ease !important; margin-bottom:2px !important;
    font-family:'Times New Roman',serif !important;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
    background:rgba(0,212,255,.06) !important;
    color:#64748b !important;
    border-left-color:rgba(0,212,255,.3) !important;
}
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:has(input:checked) {
    color:var(--cyan) !important;
    background:rgba(0,212,255,.08) !important;
    border-left-color:var(--cyan) !important;
    font-weight:600 !important;
}
section[data-testid="stSidebar"] .stRadio input { display:none !important; }

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button {
    background:transparent !important; color:var(--cyan) !important;
    border:1px solid rgba(0,212,255,.3) !important; border-radius:6px !important;
    padding:.52rem 1.3rem !important; font-family:'Times New Roman',serif !important;
    font-weight:600 !important; font-size:.87rem !important;
    letter-spacing:.02em !important; transition:all .2s !important;
}
.stButton > button:hover {
    background:rgba(0,212,255,.1) !important; border-color:var(--cyan) !important;
    box-shadow:0 0 18px rgba(0,212,255,.2) !important;
}
.btn-cta .stButton > button {
    background:var(--cyan) !important; color:#060912 !important;
    border:none !important; font-weight:700 !important;
    padding:.7rem 0 !important; font-size:.95rem !important;
    box-shadow:0 4px 20px rgba(0,212,255,.3) !important; width:100% !important;
}
.btn-cta .stButton > button:hover {
    background:#22e0ff !important;
    box-shadow:0 4px 30px rgba(0,212,255,.5) !important;
    transform:translateY(-1px) !important;
}
.btn-preset .stButton > button {
    font-size:.78rem !important; padding:.4rem .8rem !important;
    color:#64748b !important; border-color:rgba(255,255,255,.07) !important;
}
.btn-preset .stButton > button:hover { color:var(--cyan) !important; }

/* ── Form inputs ──────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div      { background:var(--cyan) !important; }
[data-testid="stSlider"] > div > div > div > div {
    background:var(--surf2) !important; border:1px solid var(--border) !important;
}
input[type="number"] {
    background:var(--surf) !important; border:1px solid var(--border) !important;
    color:var(--text) !important; border-radius:6px !important;
    font-family:'Times New Roman',serif !important;
}
[data-testid="stSelectbox"] > div > div {
    background:var(--surf) !important; border-color:var(--border) !important;
    color:var(--text) !important; border-radius:6px !important;
}
[data-testid="stExpander"] {
    background:var(--surf) !important; border:1px solid var(--border) !important;
    border-radius:8px !important;
}

/* ── DataFrames ───────────────────────────────────────────────────── */
.stDataFrame { border-radius:10px; overflow:hidden; }
.stDataFrame table                  { background:var(--surf) !important; }
.stDataFrame table thead th {
    background:var(--surf2) !important; color:var(--muted) !important;
    font-size:.75rem !important; font-weight:700 !important;
    text-transform:uppercase; letter-spacing:.08em;
    border-bottom:1px solid var(--border) !important;
}
.stDataFrame table tbody td {
    color:var(--text) !important; border-color:rgba(255,255,255,.03) !important;
    font-family:'Times New Roman',serif !important; font-size:.82rem !important;
}
.stDataFrame table tbody tr:hover td { background:rgba(0,212,255,.04) !important; }

/* ── KPI grid ─────────────────────────────────────────────────────── */
.kpi-grid {
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:1rem; margin-bottom:2rem;
}
.kpi-card {
    background:var(--surf); border:1px solid var(--border);
    border-radius:12px; padding:1.4rem 1.6rem;
    position:relative; overflow:hidden;
    animation:fadeUp .6s ease both;
    transition:border-color .3s,box-shadow .3s;
}
.kpi-card:hover { border-color:rgba(0,212,255,.3); animation:glow 2s ease infinite; }
.kpi-card::after {
    content:''; position:absolute; top:0;left:0;right:0; height:2px;
    background:linear-gradient(90deg,var(--cyan),var(--green));
}
.kpi-icon  { font-size:1.25rem; margin-bottom:.55rem; }
.kpi-label {
    font-size:.6rem; font-weight:700; letter-spacing:.16em;
    color:var(--muted); text-transform:uppercase; margin-bottom:.45rem;
}
.kpi-value {
    font-family:'Times New Roman',serif;
    font-size:2.1rem; font-weight:700; color:var(--cyan); line-height:1;
}
.kpi-unit  { font-size:1.1rem; opacity:.7; }
.kpi-sub   { font-size:.71rem; color:var(--muted); margin-top:.3rem; }
.kpi-bar   { margin-top:.9rem; height:3px; background:rgba(255,255,255,.05); border-radius:2px; overflow:hidden; }
.kpi-fill  {
    height:100%; background:linear-gradient(90deg,var(--cyan),var(--green));
    width:0; animation:barFill 1.8s cubic-bezier(.4,0,.2,1) forwards;
}

/* ── Section label ────────────────────────────────────────────────── */
.sec-label {
    font-size:.6rem; font-weight:700; letter-spacing:.18em;
    text-transform:uppercase; color:var(--muted);
    margin-bottom:.9rem; display:flex; align-items:center; gap:.6rem;
}
.sec-label::after { content:''; flex:1; height:1px; background:var(--border); }

/* ── Model rows ───────────────────────────────────────────────────── */
.model-list { display:flex; flex-direction:column; gap:.6rem; }
.model-row {
    background:var(--surf); border:1px solid var(--border);
    border-radius:10px; padding:1rem 1.3rem;
    display:grid; grid-template-columns:180px 1fr;
    align-items:center; gap:1.5rem;
    animation:fadeUp .5s ease both; position:relative;
    transition:border-color .2s;
}
.model-row:hover          { border-color:rgba(0,212,255,.25); }
.model-row.best           { border-color:rgba(0,212,255,.28); }
.model-row.best::before   {
    content:''; position:absolute; left:0;top:0;bottom:0;
    width:3px; border-radius:10px 0 0 10px;
    background:linear-gradient(180deg,var(--cyan),var(--green));
}
.model-num  { font-family:'Times New Roman',serif; font-size:.65rem; color:var(--muted); }
.model-name { font-family:'Times New Roman',serif; font-size:.88rem; font-weight:700; color:var(--text); margin-top:.1rem; }
.best-tag   {
    display:inline-block; margin-left:.4rem; vertical-align:middle;
    background:rgba(0,212,255,.12); border:1px solid rgba(0,212,255,.3);
    color:var(--cyan); font-size:.55rem; font-weight:700;
    letter-spacing:.1em; padding:.12rem .45rem; border-radius:4px;
}
.bars       { display:flex; flex-direction:column; gap:.45rem; }
.bar-row    { display:grid; grid-template-columns:3.2rem 1fr 4.5rem; align-items:center; gap:.6rem; }
.bar-lbl    { font-size:.58rem; font-weight:700; letter-spacing:.08em; color:var(--muted); text-transform:uppercase; }
.bar-track  { height:5px; background:rgba(255,255,255,.04); border-radius:3px; overflow:hidden; }
.bar-fill   { height:100%; border-radius:3px; width:0; animation:barFill 1.4s cubic-bezier(.4,0,.2,1) forwards; }
.bar-val    { font-family:'Times New Roman',serif; font-size:.73rem; font-weight:700; text-align:right; }

/* ── Result card ──────────────────────────────────────────────────── */
.res-card   {
    border-radius:12px; padding:1.75rem 2rem;
    animation:popIn .55s cubic-bezier(.34,1.56,.64,1) both;
    position:relative; overflow:hidden; margin-top:1rem;
}
.res-card.ok  { background:rgba(0,255,136,.05); border:1.5px solid rgba(0,255,136,.35); }
.res-card.mid { background:rgba(251,191,36,.05); border:1.5px solid rgba(251,191,36,.35); }
.res-card.bad { background:rgba(255,77,109,.05); border:1.5px solid rgba(255,77,109,.35); }
.res-pct {
    font-family:'Times New Roman',serif; font-size:3.8rem;
    font-weight:700; line-height:1;
}
.res-card.ok  .res-pct { color:var(--green); }
.res-card.mid .res-pct { color:var(--amber); }
.res-card.bad .res-pct { color:var(--red);   }
.res-verd {
    font-family:'Times New Roman',serif; font-size:1rem;
    font-weight:700; margin:.45rem 0 .2rem;
}
.res-card.ok  .res-verd { color:var(--green); }
.res-card.mid .res-verd { color:var(--amber); }
.res-card.bad .res-verd { color:var(--red);   }
.res-desc { font-size:.82rem; color:var(--muted); margin-top:.65rem; line-height:1.55; }

/* ── Tech stack grid ──────────────────────────────────────────────── */
.stack-grid {
    display:grid; grid-template-columns:repeat(3,1fr);
    gap:.55rem; margin:1rem 0;
}
.stack-item {
    background:var(--surf); border:1px solid var(--border);
    border-radius:8px; padding:.7rem .8rem; text-align:center;
    font-size:.75rem; font-weight:600; color:var(--muted);
    transition:all .2s; cursor:default;
}
.stack-item:hover { border-color:rgba(0,212,255,.3); color:var(--cyan); background:rgba(0,212,255,.05); }
.stack-item .si  { display:block; font-size:1.05rem; margin-bottom:.25rem; }

/* ── Pipeline steps ───────────────────────────────────────────────── */
.pipe-step { display:flex; gap:1rem; margin-bottom:.85rem; animation:fadeUp .5s ease both; }
.pipe-num  {
    font-family:'Times New Roman',serif; font-size:.65rem; font-weight:700;
    color:var(--cyan); background:rgba(0,212,255,.1); border:1px solid rgba(0,212,255,.2);
    border-radius:4px; padding:.2rem .45rem; height:fit-content;
    white-space:nowrap; letter-spacing:.05em; margin-top:.15rem;
}
.pipe-step h4 {
    font-family:'Times New Roman',serif !important; font-size:.86rem !important;
    font-weight:700 !important; color:var(--text) !important; margin:0 0 .15rem !important;
}
.pipe-step p  { font-size:.78rem; color:var(--muted); margin:0; line-height:1.5; }

/* ── Author card ──────────────────────────────────────────────────── */
.author-card {
    background:var(--surf); border:1px solid var(--border);
    border-radius:12px; padding:1.75rem; text-align:center;
    animation:fadeUp .6s ease both;
}
.author-av {
    width:60px; height:60px; border-radius:50%;
    background:linear-gradient(135deg,var(--cyan),var(--green));
    margin:0 auto .9rem; display:flex; align-items:center; justify-content:center;
    font-family:'Times New Roman',serif; font-size:1.2rem; font-weight:800; color:#060912;
}
.author-name  { font-family:'Times New Roman',serif; font-size:1.05rem; font-weight:700; color:var(--text); }
.author-role  { font-size:.75rem; color:var(--muted); margin-top:.2rem; line-height:1.5; }
.author-links { display:flex; gap:.45rem; justify-content:center; margin-top:1.1rem; flex-wrap:wrap; }
.al { display:inline-block; padding:.3rem .85rem; border-radius:5px; font-size:.73rem; font-weight:600;
      text-decoration:none !important; border:1px solid rgba(0,212,255,.22);
      color:var(--cyan) !important; transition:all .2s; }
.al:hover { background:rgba(0,212,255,.1); border-color:var(--cyan); }

/* ── Responsive ───────────────────────────────────────────────────── */
@media(max-width:768px){
    .kpi-grid          { grid-template-columns:repeat(2,1fr); }
    .stack-grid        { grid-template-columns:repeat(2,1fr); }
    .model-row         { grid-template-columns:1fr; }
}
</style>
""", unsafe_allow_html=True)

# CSS additionnel — titres de pages
st.markdown("""
<style>
.ph { margin-bottom:2rem; }
.ph-tag {
    font-family:'Times New Roman',serif;
    font-size:.58rem; font-weight:700; letter-spacing:.22em;
    color:var(--cyan); text-transform:uppercase; margin-bottom:.55rem;
    display:flex; align-items:center; gap:.5rem;
}
.ph-tag::before {
    content:''; display:inline-block; width:18px; height:2px;
    background:var(--cyan); border-radius:2px;
}
.ph-title {
    font-family:'Times New Roman',serif;
    font-size:3rem; font-weight:800; letter-spacing:-.04em; line-height:1.05;
    text-transform:none;
    color:var(--text);
    background:linear-gradient(120deg,#e2e8f0 40%,var(--cyan));
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text;
}
.ph-sub {
    font-size:1rem; color:#94a3b8; margin-top:.65rem; max-width:760px;
}
.ph-rule {
    margin-top:1.5rem; height:1px;
    background:linear-gradient(90deg,rgba(0,212,255,.35),transparent);
}
</style>
""", unsafe_allow_html=True)


def page_header(tag: str, title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div class="ph">
        <div class="ph-tag">{tag}</div>
        <div class="ph-title">{title}</div>
        <div class="ph-sub">{subtitle}</div>
        <div class="ph-rule"></div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# CONSTANTES
# =====================================================================
FEATURE_LABELS = {
    "lycee_mean":              "Moyenne générale (toutes matières)",
    "lycee_std":               "Régularité des notes (écart-type)",
    "lycee_min":               "Note la plus basse",
    "lycee_max":               "Note la plus haute",
    "lycee_count":             "Nombre d'évaluations",
    "lycee_mean_scientifique": "Sciences — Maths, Physique-Chimie, SVT",
    "lycee_mean_langues":      "Langues — Français, Anglais, Arabe",
    "lycee_mean_litteraire":   "Humanités — Philo, Histoire-Géo, Éd. islamique",
    "lycee_mean_autre":        "Autres — EPS, Informatique, Arts",
    "age":                     "Âge de l'élève",
}

FEATURE_HELP = {
    "lycee_mean":              "Moyenne de TOUTES les notes — l'indicateur principal.",
    "lycee_std":               "Faible écart-type = élève régulier. Élevé = résultats inégaux.",
    "lycee_min":               "Note la plus basse toutes matières confondues.",
    "lycee_max":               "Note la plus haute toutes matières confondues.",
    "lycee_count":             "Nombre total d'évaluations enregistrées sur l'année.",
    "lycee_mean_scientifique": "Moyenne en Mathématiques, Physique-Chimie et SVT.",
    "lycee_mean_langues":      "Moyenne en Français, Anglais et Arabe.",
    "lycee_mean_litteraire":   "Moyenne en Philosophie, Histoire-Géo et Éd. islamique.",
    "lycee_mean_autre":        "Moyenne dans les autres matières : EPS, Informatique, etc.",
    "age":                     "Âge de l'élève au moment du baccalauréat.",
}

FILIERE_OPTIONS = {
    "Sciences Mathématiques A  (option français)": {
        "level_2ème Année Bac Sciences Mathématiques A - Option Français": 1,
        "level_2ème Année Bac Sciences Physiques - Option Français": 0,
    },
    "Sciences Physiques  (option français)": {
        "level_2ème Année Bac Sciences Mathématiques A - Option Français": 0,
        "level_2ème Année Bac Sciences Physiques - Option Français": 1,
    },
    "Autre filière": {
        "level_2ème Année Bac Sciences Mathématiques A - Option Français": 0,
        "level_2ème Année Bac Sciences Physiques - Option Français": 0,
    },
}

PROFILES = {
    "⚠️ En difficulté": dict(
        lycee_mean=7.5, lycee_std=2.8, lycee_min=2.0, lycee_max=12.0, lycee_count=55,
        lycee_mean_scientifique=7.0, lycee_mean_langues=8.5,
        lycee_mean_litteraire=8.0,   lycee_mean_autre=9.0, age=18.0,
        filiere="Sciences Mathématiques A  (option français)",
    ),
    "📚 Niveau moyen": dict(
        lycee_mean=12.0, lycee_std=2.0, lycee_min=7.5, lycee_max=16.5, lycee_count=65,
        lycee_mean_scientifique=11.5, lycee_mean_langues=12.5,
        lycee_mean_litteraire=12.0,   lycee_mean_autre=12.5, age=17.5,
        filiere="Sciences Mathématiques A  (option français)",
    ),
    "🌟 Excellent": dict(
        lycee_mean=15.5, lycee_std=1.2, lycee_min=11.5, lycee_max=19.0, lycee_count=70,
        lycee_mean_scientifique=16.5, lycee_mean_langues=15.0,
        lycee_mean_litteraire=15.0,   lycee_mean_autre=14.5, age=17.0,
        filiere="Sciences Mathématiques A  (option français)",
    ),
}

# =====================================================================
# CHARGEMENT
# =====================================================================
OUTPUTS = Path(__file__).parent / "outputs"

@st.cache_data
def load_results():
    with open(OUTPUTS / "results.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_artifacts():
    return joblib.load(OUTPUTS / "best_model.pkl")

try:
    _raw      = load_results()
    artifacts = load_artifacts()
    meta      = _raw.pop("_meta", {})
    model     = artifacts["model"]
    scaler    = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    results   = _raw
except FileNotFoundError:
    st.error("Modèle introuvable — lance d'abord : `python train_models.py`")
    st.stop()

best_name = meta.get("best_model", "Logistic Regression")
best_acc  = meta.get("best_accuracy", 0.889)
best_auc  = results[best_name]["auc"]
baseline  = meta.get("baseline_accuracy", 0.71)
n_samples = meta.get("n_samples", 86)
n_feats   = meta.get("n_features", 12)
gain      = best_acc - baseline

# =====================================================================
# SESSION STATE
# =====================================================================
_def = PROFILES["📚 Niveau moyen"]
for _k, _v in _def.items():
    skey = "pred_filiere" if _k == "filiere" else f"pred_{_k}"
    if skey not in st.session_state:
        st.session_state[skey] = _v

# =====================================================================
# HELPERS HTML
# =====================================================================
def kpi_html(icon, label, value, unit, sub, w, delay):
    return f"""
    <div class="kpi-card" style="animation-delay:{delay}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
        <div class="kpi-sub">{sub}</div>
        <div class="kpi-bar">
            <div class="kpi-fill" style="--w:{w};animation-delay:{delay}"></div>
        </div>
    </div>"""


def model_rows_html(results, best_name):
    colors = {"ACC": "#00d4ff", "AUC": "#00ff88", "CV": "#fbbf24"}
    html = '<div class="model-list">'
    for i, (name, r) in enumerate(results.items()):
        is_best = name == best_name
        cls     = "best" if is_best else ""
        badge   = '<span class="best-tag">BEST</span>' if is_best else ""
        delay   = f"{i*.12}s"

        bars = ""
        for key, lbl in [("accuracy","ACC"),("auc","AUC"),("cv_mean","CV")]:
            v   = r[key]
            d   = f"{v*100:.1f}%" if key != "auc" else f"{v:.3f}"
            w   = f"{v*100:.1f}%"
            c   = colors[lbl]
            bd  = f"{i*.1+.3}s"
            bars += f"""
            <div class="bar-row">
                <div class="bar-lbl">{lbl}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="--w:{w};background:{c};animation-delay:{bd}"></div>
                </div>
                <div class="bar-val" style="color:{c}">{d}</div>
            </div>"""

        html += f"""
        <div class="model-row {cls}" style="animation-delay:{delay}">
            <div>
                <div class="model-num">#{i+1:02d}</div>
                <div class="model-name">{name}{badge}</div>
            </div>
            <div class="bars">{bars}</div>
        </div>"""
    return html + "</div>"


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding:.25rem .5rem 1.5rem">
        <div style="font-family:'Times New Roman',serif;font-size:1.25rem;
                    font-weight:800;color:#e2e8f0;letter-spacing:-.02em;">
            ⚡ Bac Prediction
        </div>
        <div style="font-size:.7rem;color:#374151;margin-top:.15rem;
                    font-family:'Times New Roman',serif;letter-spacing:.05em;">
            ML · CLASSIFICATION · PYTHON
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav", label_visibility="collapsed",
        options=["🏠  Vue d'ensemble", "📊  Comparaison interactive",
                 "🎯  Outil de prédiction", "📚  Méthodologie & Auteure"],
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding:.25rem .5rem 1rem">
        <div style="font-size:.6rem;font-weight:700;letter-spacing:.14em;
                    text-transform:uppercase;color:#1e293b;margin-bottom:.7rem">
            Meilleur modèle
        </div>
        <div style="font-family:'Times New Roman',serif;font-size:.88rem;
                    font-weight:700;color:#94a3b8">{best_name}</div>
        <div style="font-family:'Times New Roman',serif;font-size:1.6rem;
                    font-weight:700;color:#00d4ff;line-height:1.1;margin:.15rem 0">
            {best_acc*100:.1f}%
        </div>
        <div style="font-size:.7rem;color:#374151">
            accuracy · AUC&nbsp;{best_auc:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding:.25rem .5rem 1.25rem;font-size:.78rem;color:#374151;
                line-height:2;font-family:'Times New Roman',serif">
        <div>👥 {n_samples} étudiants</div>
        <div>🔢 {n_feats} variables</div>
        <div>🤖 4 algorithmes</div>
        <div>📊 5-fold CV</div>
        <div>🐍 Python · Scikit-Learn</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:.25rem .5rem;display:flex;flex-direction:column;gap:.4rem">
        <a href="https://github.com/MarteOued/bac-prediction-ml" target="_blank"
           style="font-size:.78rem;color:#374151 !important;text-decoration:none;
                  transition:color .2s" onmouseover="this.style.color='#00d4ff'"
           onmouseout="this.style.color='#374151'">
            ↗ GitHub
        </a>
        <a href="https://portfoliomarte.vercel.app" target="_blank"
           style="font-size:.78rem;color:#374151 !important;text-decoration:none"
           onmouseover="this.style.color='#00d4ff'" onmouseout="this.style.color='#374151'">
            ↗ Portfolio
        </a>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# PAGE 1 — VUE D'ENSEMBLE
# =====================================================================
if page == "🏠  Vue d'ensemble":

    page_header(
        "01 — Vue d'ensemble",
        "Prédiction de réussite au Bac",
        "Machine Learning · Classification binaire · 4 modèles comparés · 86 lycéens marocains",
    )

    # KPIs
    kpis_html = '<div class="kpi-grid">'
    kpis_html += kpi_html("🎯", "Meilleure Accuracy",
                          f"{best_acc*100:.1f}", "%",
                          best_name, f"{best_acc*100:.1f}%", "0s")
    kpis_html += kpi_html("📐", "Meilleur AUC",
                          f"{best_auc:.3f}", "",
                          "Area Under ROC Curve", f"{best_auc*100:.1f}%", ".1s")
    kpis_html += kpi_html("👥", "Étudiants",
                          str(n_samples), "",
                          "lycéens marocains", "86%", ".2s")
    kpis_html += kpi_html("⚡", "Gain vs. baseline",
                          f"+{gain*100:.0f}", "pp",
                          f"baseline {baseline*100:.0f}%", f"{min(gain*300,100):.0f}%", ".3s")
    kpis_html += "</div>"
    st.markdown(kpis_html, unsafe_allow_html=True)

    # Model comparison + visualizations
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="sec-label">Performance des 4 modèles</div>',
                    unsafe_allow_html=True)
        st.markdown(model_rows_html(results, best_name), unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="sec-label">Visualisations</div>', unsafe_allow_html=True)
        if (OUTPUTS / "01_model_comparison.png").exists():
            st.markdown("""
            <div style="background:var(--surf);border:1px solid var(--border);
                        border-radius:10px;padding:.75rem;margin-bottom:.75rem">
            """, unsafe_allow_html=True)
            st.image(str(OUTPUTS / "01_model_comparison.png"),
                     caption="Comparaison des 4 modèles", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if (OUTPUTS / "02_confusion_matrix.png").exists():
            st.markdown("""
            <div style="background:var(--surf);border:1px solid var(--border);
                        border-radius:10px;padding:.75rem">
            """, unsafe_allow_html=True)
            st.image(str(OUTPUTS / "02_confusion_matrix.png"),
                     caption=f"Matrice de confusion — {best_name}",
                     use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if (OUTPUTS / "03_feature_importance.png").exists():
        st.markdown("""
        <div style="background:var(--surf);border:1px solid var(--border);
                    border-radius:10px;padding:.75rem;margin-top:.75rem">
        """, unsafe_allow_html=True)
        st.image(str(OUTPUTS / "03_feature_importance.png"),
                 caption="Importance des variables — Random Forest",
                 use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# PAGE 2 — COMPARAISON INTERACTIVE
# =====================================================================
elif page == "📊  Comparaison interactive":

    page_header(
        "02 — Comparaison",
        "Comparaison interactive",
        "Analyse détaillée des 4 algorithmes sur toutes les métriques",
    )

    mnames = list(results.keys())
    CLRS   = {n: "#00d4ff" if n == best_name else "#1e3a5f" for n in mnames}
    clr_l  = [CLRS[n] for n in mnames]

    _grid = dict(gridcolor="rgba(255,255,255,.04)", zerolinecolor="rgba(0,0,0,0)")
    _layout = dict(
        template="plotly_dark",
        paper_bgcolor="#0c1018", plot_bgcolor="#0c1018",
        font=dict(family="DM Sans, sans-serif", color="#64748b"),
        margin=dict(t=55,b=40,l=45,r=20), height=380,
        legend=dict(bgcolor="#111827", bordercolor="rgba(0,212,255,.15)", borderwidth=1),
    )

    # Bar: accuracy test vs CV
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        name="Accuracy (test)", x=mnames,
        y=[results[n]["accuracy"]*100 for n in mnames],
        marker_color=clr_l,
        text=[f"{results[n]['accuracy']*100:.1f}%" for n in mnames],
        textposition="outside", textfont=dict(color="#94a3b8", size=11),
        width=.35,
    ))
    fig1.add_trace(go.Bar(
        name="CV 5-fold (moy.)", x=mnames,
        y=[results[n]["cv_mean"]*100 for n in mnames],
        marker_color=["rgba(0,212,255,.25)" if n==best_name else "rgba(30,58,95,.6)" for n in mnames],
        text=[f"{results[n]['cv_mean']*100:.1f}%" for n in mnames],
        textposition="outside", textfont=dict(color="#94a3b8", size=11),
        width=.35,
    ))
    fig1.add_hline(y=baseline*100, line_dash="dot", line_color="#374151",
                   annotation_text=f"Baseline ({baseline*100:.0f}%)",
                   annotation_font=dict(color="#475569", size=11))
    fig1.update_layout(**_layout, barmode="group",
                       title=dict(text="Accuracy : test vs. cross-validation",
                                  font=dict(family="Syne, sans-serif", size=14, color="#94a3b8")),
                       xaxis=_grid,
                       yaxis=dict(range=[50,100], **_grid))
    st.plotly_chart(fig1, use_container_width=True)

    # Line: all metrics
    mc = [("accuracy","Accuracy","#00d4ff"), ("precision","Précision","#00ff88"),
          ("recall","Rappel","#fbbf24"), ("f1","F1-Score","#f472b6"), ("auc","AUC","#818cf8")]
    fig2 = go.Figure()
    for mk, ml, mc_ in mc:
        fig2.add_trace(go.Scatter(
            x=mnames, y=[results[n][mk]*100 for n in mnames],
            mode="lines+markers", name=ml,
            line=dict(color=mc_, width=2.5),
            marker=dict(size=9, color=mc_, symbol="circle",
                        line=dict(color="#060912", width=2)),
        ))
    fig2.update_layout(**_layout,
                       title=dict(text="Toutes les métriques — vue d'ensemble",
                                  font=dict(family="Syne, sans-serif", size=14, color="#94a3b8")),
                       xaxis=_grid,
                       yaxis=dict(range=[60,100], **_grid))
    st.plotly_chart(fig2, use_container_width=True)

    # AUC bar
    fig3 = go.Figure(go.Bar(
        x=mnames, y=[results[n]["auc"] for n in mnames],
        marker_color=clr_l,
        text=[f"{results[n]['auc']:.3f}" for n in mnames],
        textposition="outside", textfont=dict(color="#94a3b8", size=11),
    ))
    fig3.add_hline(y=.5, line_dash="dot", line_color="#374151",
                   annotation_text="AUC aléatoire = 0.5",
                   annotation_font=dict(color="#475569", size=11))
    fig3.update_layout(**_layout, showlegend=False,
                       title=dict(text="AUC — Area Under the ROC Curve",
                                  font=dict(family="Syne, sans-serif", size=14, color="#94a3b8")),
                       xaxis=_grid,
                       yaxis=dict(range=[.4,1.0], **_grid))
    st.plotly_chart(fig3, use_container_width=True)


# =====================================================================
# PAGE 3 — PRÉDICTION
# =====================================================================
elif page == "🎯  Outil de prédiction":

    page_header(
        "03 — Prédiction",
        "Outil de prédiction",
        f"Modèle actif : {best_name}  ·  Accuracy {best_acc*100:.1f}%  ·  AUC {best_auc:.3f}",
    )

    # Profils
    st.markdown('<div class="sec-label">Profils exemples — cliquez pour préremplir</div>',
                unsafe_allow_html=True)
    pc1, pc2, pc3 = st.columns(3)
    for idx, (pname, pvals) in enumerate(PROFILES.items()):
        with [pc1, pc2, pc3][idx]:
            st.markdown('<div class="btn-preset">', unsafe_allow_html=True)
            if st.button(pname, use_container_width=True, key=f"p{idx}"):
                for pk, pv in pvals.items():
                    st.session_state["pred_filiere" if pk=="filiere" else f"pred_{pk}"] = pv
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # Formulaire
    with st.form("pred"):

        st.markdown('<div class="sec-label">Filière scolaire</div>', unsafe_allow_html=True)
        fl = list(FILIERE_OPTIONS.keys())
        sf = st.session_state.get("pred_filiere", fl[0])
        filiere = st.selectbox(
            "Filière (2ème année Bac)", fl,
            index=fl.index(sf) if sf in fl else 0,
            help="Choisissez la filière. Le modèle a été entraîné sur les deux filières scientifiques.",
        )

        st.markdown("---")
        st.markdown('<div class="sec-label">Notes générales — sur 20</div>',
                    unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3)
        with g1:
            lycee_mean = st.slider(FEATURE_LABELS["lycee_mean"], 0.0, 20.0, step=.25,
                value=float(st.session_state.get("pred_lycee_mean", 12.0)),
                help=FEATURE_HELP["lycee_mean"])
        with g2:
            lycee_min = st.slider(FEATURE_LABELS["lycee_min"], 0.0, 20.0, step=.25,
                value=float(st.session_state.get("pred_lycee_min", 7.5)),
                help=FEATURE_HELP["lycee_min"])
        with g3:
            lycee_max = st.slider(FEATURE_LABELS["lycee_max"], 0.0, 20.0, step=.25,
                value=float(st.session_state.get("pred_lycee_max", 16.5)),
                help=FEATURE_HELP["lycee_max"])

        g4, g5, g6 = st.columns(3)
        with g4:
            lycee_std = st.slider(FEATURE_LABELS["lycee_std"], 0.0, 8.0, step=.25,
                value=float(st.session_state.get("pred_lycee_std", 2.0)),
                help=FEATURE_HELP["lycee_std"])
        with g5:
            lycee_count = st.number_input(FEATURE_LABELS["lycee_count"],
                min_value=1, max_value=300, step=1,
                value=int(st.session_state.get("pred_lycee_count", 65)),
                help=FEATURE_HELP["lycee_count"])
        with g6:
            age = st.number_input(FEATURE_LABELS["age"],
                min_value=14.0, max_value=22.0, step=0.5,
                value=float(st.session_state.get("pred_age", 17.5)),
                help=FEATURE_HELP["age"])

        st.markdown("---")
        st.markdown('<div class="sec-label">Moyennes par domaine — sur 20</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            sci  = st.slider("🔬 Sciences", 0.0, 20.0, step=.25,
                value=float(st.session_state.get("pred_lycee_mean_scientifique", 11.5)),
                help=FEATURE_HELP["lycee_mean_scientifique"])
            st.caption("Maths · Physique-Chimie · SVT")
        with m2:
            lang = st.slider("🌐 Langues", 0.0, 20.0, step=.25,
                value=float(st.session_state.get("pred_lycee_mean_langues", 12.5)),
                help=FEATURE_HELP["lycee_mean_langues"])
            st.caption("Français · Anglais · Arabe")
        with m3:
            lit  = st.slider("📖 Humanités", 0.0, 20.0, step=.25,
                value=float(st.session_state.get("pred_lycee_mean_litteraire", 12.0)),
                help=FEATURE_HELP["lycee_mean_litteraire"])
            st.caption("Philo · Histoire-Géo · Éd. islamique")
        with m4:
            autre = st.slider("⚽ Autres", 0.0, 20.0, step=.25,
                value=float(st.session_state.get("pred_lycee_mean_autre", 12.5)),
                help=FEATURE_HELP["lycee_mean_autre"])
            st.caption("EPS · Informatique · Arts")

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="btn-cta">', unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡  Analyser le profil", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Résultat
    if submitted:
        level_vals = FILIERE_OPTIONS[filiere]
        raw = {
            "lycee_mean": lycee_mean, "lycee_std": lycee_std,
            "lycee_min":  lycee_min,  "lycee_max": lycee_max,
            "lycee_count": float(lycee_count),
            "lycee_mean_autre": autre, "lycee_mean_langues": lang,
            "lycee_mean_litteraire": lit, "lycee_mean_scientifique": sci,
            "age": float(age), **level_vals,
        }
        X_sc = scaler.transform(np.array([[raw[f] for f in feature_names]]))
        pred  = model.predict(X_sc)[0]
        proba = model.predict_proba(X_sc)[0]
        ps    = proba[1] * 100
        pf    = proba[0] * 100

        if ps >= 65:
            cls, icon, verd, desc, gauge_c = (
                "ok", "✅", "Réussite prédite",
                f"La probabilité de réussite au baccalauréat est de <strong style='color:#00ff88'>{ps:.1f}%</strong>. "
                "Le profil de cet élève correspond aux caractéristiques des élèves ayant réussi dans les données d'entraînement. "
                "Continuer sur cette trajectoire.",
                "#00ff88",
            )
        elif ps >= 45:
            cls, icon, verd, desc, gauge_c = (
                "mid", "⚡", "Profil à risque modéré",
                f"La probabilité de réussite est de <strong style='color:#fbbf24'>{ps:.1f}%</strong> "
                f"et d'échec de <strong style='color:#fbbf24'>{pf:.1f}%</strong>. "
                "Un accompagnement ciblé sur les matières faibles peut faire la différence.",
                "#fbbf24",
            )
        else:
            cls, icon, verd, desc, gauge_c = (
                "bad", "⚠️", "Risque d'échec élevé",
                f"La probabilité d'échec est de <strong style='color:#ff4d6d'>{pf:.1f}%</strong>. "
                "Un suivi pédagogique renforcé est fortement recommandé avant les examens.",
                "#ff4d6d",
            )

        rc1, rc2 = st.columns([1, 1], gap="large")

        with rc1:
            st.markdown(f"""
            <div class="res-card {cls}">
                <div style="font-size:1.3rem;margin-bottom:.5rem">{icon}</div>
                <div class="res-pct">{ps:.1f}<span style="font-size:2rem;opacity:.7">%</span></div>
                <div class="res-verd">{verd}</div>
                <div class="res-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        with rc2:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ps,
                number={"suffix":"%","font":{"size":38,"color":"#e2e8f0",
                                             "family":"JetBrains Mono, monospace"}},
                title={"text":"Probabilité de réussite",
                       "font":{"size":12,"color":"#64748b","family":"DM Sans, sans-serif"}},
                gauge={
                    "axis":{"range":[0,100],"tickfont":{"color":"#374151","size":10},
                            "tickcolor":"#1e293b"},
                    "bar":{"color":gauge_c,"thickness":.6},
                    "bgcolor":"#0c1018", "borderwidth":0,
                    "steps":[
                        {"range":[0,45], "color":"rgba(255,77,109,.08)"},
                        {"range":[45,65],"color":"rgba(251,191,36,.08)"},
                        {"range":[65,100],"color":"rgba(0,255,136,.08)"},
                    ],
                    "threshold":{"line":{"color":"#475569","width":2},
                                 "thickness":.8,"value":50},
                },
            ))
            fig_g.update_layout(
                paper_bgcolor="#0c1018", height=260,
                margin=dict(t=50,b=15,l=30,r=30),
                font=dict(family="DM Sans"),
            )
            st.plotly_chart(fig_g, use_container_width=True)

        with st.expander("📋 Détail des valeurs saisies"):
            rows = [(FEATURE_LABELS.get(f, f),
                     f"{raw[f]:.2f}" if isinstance(raw[f], float) else str(raw[f]))
                    for f in feature_names if "level_" not in f]
            rows.append(("Filière", filiere))
            df_r = pd.DataFrame(rows, columns=["Variable", "Valeur"])
            st.dataframe(df_r, use_container_width=True, hide_index=True)


# =====================================================================
# PAGE 4 — MÉTHODOLOGIE & AUTEURE
# =====================================================================
elif page == "📚  Méthodologie & Auteure":

    page_header(
        "04 — Méthodologie",
        "Méthodologie & Auteure",
        "Pipeline ML complet · du nettoyage des données à l'interface interactive",
    )

    col_pipe, col_meta = st.columns([3, 2], gap="large")

    with col_pipe:
        st.markdown('<div class="sec-label">Pipeline ML</div>', unsafe_allow_html=True)

        steps = [
            ("01", "Collecte & nettoyage",
             "43 108 lignes de notes lycée + 440 lignes de résultats bac. "
             "Nettoyage : format français (« 18,50 »), valeurs manquantes (« **** », « Abs. »), strip whitespace."),
            ("02", "Feature Engineering",
             "Agrégation par élève : moyenne, écart-type, min, max, médiane, count. "
             "Moyennes par domaine (sciences, langues, humanités, autres). "
             "Encodage one-hot de la filière."),
            ("03", "Cible binaire",
             "Réussite = moyenne bac ≥ 10/20. "
             "61 élèves réussissent (71%) · 25 échouent (29%) — dataset de 86 élèves communs."),
            ("04", "Entraînement",
             "4 algorithmes : Logistic Regression (C=0.5), Random Forest (n=300), "
             "Gradient Boosting (n=200, lr=0.05), MLP (64→32, alpha=0.01). "
             "Split 80/20 stratifié · StandardScaler."),
            ("05", "Évaluation",
             "Accuracy, Précision, Rappel, F1-Score, AUC. "
             "Cross-validation stratifiée 5-fold pour la robustesse."),
            ("06", "Interface",
             "Application Streamlit avec 4 sections, outil de prédiction interactif "
             "et visualisations Plotly. Modèle persistant via Joblib."),
        ]

        html_pipe = ""
        for num, title, desc in steps:
            html_pipe += f"""
            <div class="pipe-step" style="animation-delay:{int(num)*.08}s">
                <div class="pipe-num">{num}</div>
                <div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
            </div>"""
        st.markdown(html_pipe, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Résultats clés</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:var(--surf);border:1px solid var(--border);
                    border-radius:10px;padding:1.25rem 1.5rem;
                    font-size:.85rem;color:#94a3b8;line-height:1.8">
            Meilleur modèle : <span style="color:#00d4ff;font-family:'Times New Roman',serif;
            font-weight:700">{best_name}</span> ·
            Accuracy <span style="color:#00d4ff;font-family:'Times New Roman',serif">{best_acc*100:.1f}%</span> ·
            AUC <span style="color:#00ff88;font-family:'Times New Roman',serif">{best_auc:.3f}</span><br/>
            Gain vs. baseline :
            <span style="color:#fbbf24;font-family:'Times New Roman',serif">+{gain*100:.0f} pp</span>
            (baseline {baseline*100:.0f}%)<br/>
            Cross-validation 5-fold :
            <span style="color:#94a3b8;font-family:'Times New Roman',serif">
            {results[best_name]["cv_mean"]*100:.1f}% ± {results[best_name]["cv_std"]*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with col_meta:
        st.markdown('<div class="sec-label">Stack technique</div>', unsafe_allow_html=True)
        stack = [
            ("🐍","Python 3.10+"),("🤖","Scikit-Learn"),("🐼","Pandas"),
            ("🔢","NumPy"),     ("📊","Plotly"),      ("⚡","Streamlit"),
            ("📈","Matplotlib"),("💾","Joblib"),       ("🧪","Seaborn"),
        ]
        html_s = '<div class="stack-grid">'
        for ico, name in stack:
            html_s += f'<div class="stack-item"><span class="si">{ico}</span>{name}</div>'
        html_s += "</div>"
        st.markdown(html_s, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Auteure</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="author-card">
            <div class="author-av">MO</div>
            <div class="author-name">Martine Ouedraogo</div>
            <div class="author-role">
                Master 1 Informatique<br/>
                Data Science & Machine Learning<br/>
                Université Lumière Lyon 2 · 2026
            </div>
            <div class="author-links">
                <a class="al" href="https://portfoliomarte.vercel.app" target="_blank">Portfolio</a>
                <a class="al" href="https://github.com/MarteOued/bac-prediction-ml" target="_blank">GitHub</a>
                <a class="al" href="https://www.linkedin.com/in/marte-oued" target="_blank">LinkedIn</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Dataset</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:var(--surf);border:1px solid var(--border);
                    border-radius:10px;padding:1rem 1.25rem">
        <table style="width:100%;border-collapse:collapse;font-size:.78rem">
        <tr><td style="color:#475569;padding:.3rem 0">Étudiants lycée</td>
            <td style="text-align:right;font-family:'Times New Roman',serif;
                       color:#e2e8f0">186</td></tr>
        <tr><td style="color:#475569;padding:.3rem 0">Étudiants avec bac</td>
            <td style="text-align:right;font-family:'Times New Roman',serif;
                       color:#00d4ff;font-weight:700">{n_samples}</td></tr>
        <tr><td style="color:#475569;padding:.3rem 0">Lignes notes lycée</td>
            <td style="text-align:right;font-family:'Times New Roman',serif;
                       color:#e2e8f0">43 108</td></tr>
        <tr><td style="color:#475569;padding:.3rem 0">Lignes notes bac</td>
            <td style="text-align:right;font-family:'Times New Roman',serif;
                       color:#e2e8f0">440</td></tr>
        <tr><td style="color:#475569;padding:.3rem 0">Variables (features)</td>
            <td style="text-align:right;font-family:'Times New Roman',serif;
                       color:#e2e8f0">{n_feats}</td></tr>
        <tr><td style="color:#475569;padding:.3rem 0">Taux réussite baseline</td>
            <td style="text-align:right;font-family:'Times New Roman',serif;
                       color:#00ff88">{baseline*100:.0f}%</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("""
<div style="text-align:center;padding:3rem 0 1.5rem;
            font-family:'Times New Roman',serif;font-size:.68rem;
            color:#1e293b;letter-spacing:.08em">
    MARTINE OUEDRAOGO · M1 INFORMATIQUE · UNIVERSITÉ LUMIÈRE LYON 2 · 2026
</div>
""", unsafe_allow_html=True)
