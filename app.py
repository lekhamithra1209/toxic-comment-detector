"""Toxic Comment Detector - Streamlit app using a pre-trained multi-label model.

The .joblib file is a dict with:
  word_vec : TfidfVectorizer (word 1-2 grams)
  char_vec : TfidfVectorizer (char_wb 2-5 grams)
  models   : {label: LogisticRegression}
  labels   : list of label names
Features = hstack([word_vec, char_vec]) -> one binary classifier per label.
"""
import logging
import warnings
from pathlib import Path

import joblib
import scipy.sparse as sp
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("toxic-detector")

MODEL_PATH = Path(__file__).parent / "toxic_debiased.joblib"
THRESHOLD = 0.5  # same cut-off LogisticRegression.predict() uses
MAX_CHARS = 5000

META = {  # cosmetic only; the label list itself always comes from the model
    "toxic": ("☠️", "Toxic"),
    "severe_toxic": ("🚫", "Severe Toxic"),
    "obscene": ("🤬", "Obscene"),
    "threat": ("🚨", "Threat"),
    "insult": ("😡", "Insult"),
    "identity_hate": ("💔", "Identity Hate"),
}
CARD_COLORS = ["#f6cdd0", "#fbdcb8", "#d8d2fb", "#c6efdc", "#cfe6fb", "#fdf0b8"]

st.set_page_config(page_title="Toxic Comment Detector", page_icon="🧪", layout="wide")

# ----------------------------- doodles & pixel art -----------------------------
STAR = '<svg viewBox="0 0 24 24" width="{s}" height="{s}"><path d="M12 2l2.6 7.4L22 12l-7.4 2.6L12 22l-2.6-7.4L2 12l7.4-2.6z" fill="{c}" stroke="#3f3556" stroke-width="1.4" stroke-linejoin="round"/></svg>'
HEART = '<svg viewBox="0 0 24 24" width="{s}" height="{s}"><path d="M12 21s-8-5.2-8-11a4.5 4.5 0 0 1 8-2.8A4.5 4.5 0 0 1 20 10c0 5.800-8 11-8 11z" fill="{c}" stroke="#3f3556" stroke-width="1.4" stroke-linejoin="round"/></svg>'
CLOUD = '<svg viewBox="0 0 48 28" width="{s}" height="{s}"><path d="M12 25a8 8 0 0 1-1-15.900A11 11 0 0 1 32 8a9 9 0 0 1 5 17z" fill="{c}" stroke="#3f3556" stroke-width="1.4" stroke-linejoin="round"/></svg>'

def poison_bottle(h=60) -> str:
    """Cute pastel poison-bottle doodle (cork, skull & crossbones, bubbles)."""
    k = 'stroke="#3f3556" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"'
    return (
        f'<svg viewBox="0 0 60 80" height="{h}" width="{round(h * 0.75)}"><g transform="rotate(-8 30 42)">'
        f'<path d="M22 4h16l-1.500 9h-13z" fill="#f6d9a8" {k}/>'
        '<circle cx="27" cy="7.500" r="1" fill="#d9b27c"/><circle cx="33" cy="9" r="1" fill="#d9b27c"/>'
        f'<rect x="20" y="16" width="20" height="7" rx="3" fill="#ece6ff" {k}/>'
        f'<path d="M22 23V30Q10 34 10 46V66Q10 75 19 75H41Q50 75 50 66V46Q50 34 38 30V23Z" fill="#c6efdc" {k}/>'
        '<rect x="23" y="24" width="14" height="6" fill="#eef3ff"/>'
        '<path d="M15 46V64" stroke="#fff" stroke-width="2.600" stroke-linecap="round" opacity=".85"/>'
        f'<rect x="18" y="43" width="26" height="27" rx="6" fill="#fff" {k}/>'
        f'<ellipse cx="31" cy="51" rx="6.500" ry="5.500" fill="#fff" {k}/>'
        f'<rect x="28" y="55" width="6" height="3.500" rx="1.200" fill="#fff" {k}/>'
        '<circle cx="28.500" cy="51" r="1.500" fill="#3f3556"/><circle cx="33.500" cy="51" r="1.500" fill="#3f3556"/>'
        '<path d="M31 53l-1 1.600h2z" fill="#3f3556"/>'
        '<path d="M23 61L39 67M39 61L23 67" fill="none" stroke="#3f3556" stroke-width="2.200" stroke-linecap="round"/>'
        f'<circle cx="44" cy="12" r="2" fill="#fff" {k}/><circle cx="49" cy="6" r="1.300" fill="#fff" {k}/>'
        '<circle cx="14" cy="38" r="1.300" fill="#fff"/><circle cx="46" cy="40" r="1.100" fill="#fff"/>'
        '</g></svg>')


PIXEL_PAL = {"O": "#f7ae6a", "D": "#4a3b52", "P": "#f59bb5", "B": "#8ecbf5"}


def cat_rects(mood: str) -> str:
    """Pixel-art cat built from a character grid (happy or worried)."""
    happy = mood == "happy"
    rows = ["..O......O..", "..OO....OO..", "..OOOOOOOO..", ".OOOOOOOOOO.", ".OODOOOODOO.",
            ".OPOOOOOOPO." if happy else ".OOBOOOOOOO.", ".OOOOPPOOOO.",
            ".OOODOODOOO." if happy else ".OOOODDOOOO.",
            ".OOOODDOOOO." if happy else ".OOODOODOOO.", "..OOOOOOOO..", "..OO....OO.."]
    return "".join(f'<rect x="{x}" y="{y}" width="1" height="1" fill="{PIXEL_PAL[ch]}"/>'
                   for y, r in enumerate(rows) for x, ch in enumerate(r) if ch != ".")


def pixel_cat(mood="happy", size=96) -> str:
    return (f'<svg viewBox="0 0 12 11" width="{size}" height="{round(size * 11 / 12)}" '
            f'shape-rendering="crispEdges">{cat_rects(mood)}</svg>')


def hero_scene() -> str:
    return (
        '<svg viewBox="0 0 320 210" width="100%" style="max-width:340px">'
        '<clipPath id="sc"><rect width="320" height="210" rx="28"/></clipPath><g clip-path="url(#sc)">'
        '<rect width="320" height="210" fill="#d3ecfb"/><circle cx="264" cy="48" r="22" fill="#ffe2a8"/>'
        '<polygon points="0,155 85,70 170,155" fill="#e4d9ff"/><polygon points="110,155 205,52 305,155" fill="#ffd6e3"/>'
        '<polygon points="205,52 190,72 205,66 218,74" fill="#fff"/>'
        '<g fill="#fff"><ellipse cx="62" cy="44" rx="32" ry="12"/><ellipse cx="84" cy="36" rx="20" ry="12"/>'
        '<ellipse cx="205" cy="26" rx="26" ry="9"/></g>'
        '<rect y="150" width="320" height="60" fill="#cdeed9"/>'
        '<g><circle cx="30" cy="180" r="4" fill="#f6a5c0"/><circle cx="52" cy="194" r="3.5" fill="#ffe08a"/>'
        '<circle cx="270" cy="184" r="4" fill="#f6a5c0"/><circle cx="292" cy="170" r="3.5" fill="#fff"/>'
        '<circle cx="240" cy="196" r="3.5" fill="#ffe08a"/></g></g>'
        f'<svg x="118" y="94" width="88" height="81" viewBox="0 0 12 11" shape-rendering="crispEdges">{cat_rects("happy")}</svg>'
        '</svg>')


# ----------------------------- CSS -----------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Silkscreen:wght@400;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, .stButton button {font-family:'DM Sans',sans-serif;}
.stApp {background-color:#fff8f1; background-image:radial-gradient(#f3dccf 1.3px, transparent 1.4px); background-size:24px 24px;}
#MainMenu, footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:2rem; max-width:1150px;}
h1,h2,h3,h4 {color:#3f3556 !important; letter-spacing:-.02em;}
.px {font-family:'Silkscreen',monospace;}
[data-testid="stSidebar"] {background:#ffeee3; border-right:3px dashed #f6cdd0;}
.side-cat {text-align:center; margin:.3rem 0 .4rem;}
.side-title {font-weight:700; font-size:1.25rem; color:#3f3556; text-align:center;}
.side-sub {font-size:.8rem; color:#8a7a9c; text-align:center; margin-bottom:1.1rem;}
.side-link {display:flex; align-items:center; gap:.6rem; padding:.7rem 1rem; margin:.45rem 0; border-radius:999px; background:#fff;
  color:#3f3556 !important; text-decoration:none !important; font-weight:500; box-shadow:0 3px 0 #f1d3c6; transition:all .15s;}
.side-link:hover {background:#d8d2fb; transform:translateY(-1px);}
.side-foot {margin-top:1.8rem; padding:.8rem; border-radius:20px; background:#c6efdc; font-size:.78rem; color:#3f5f50; text-align:center; font-weight:500;}
.hero {position:relative; display:grid; grid-template-columns:1.3fr 1fr; gap:1.2rem; align-items:center; background:#fde4e6; border-radius:34px;
  padding:2.2rem 2.4rem; margin-bottom:1.4rem; box-shadow:0 8px 0 #f3c9cf;}
.hero h1 {font-size:3.1rem; line-height:1.05; margin:0 0 .6rem 0; font-weight:700;}
.hero p {font-size:1.12rem; color:#6b5b7b; margin:0 0 1.1rem 0;}
.badge {display:inline-block; padding:.45rem 1rem; border-radius:999px; background:#fff; color:#6b55a8; font-size:.84rem; font-weight:700; box-shadow:0 3px 0 #e6d9f5;}
.bottle {display:inline-block; vertical-align:middle; margin-right:.5rem; animation:bob 4s ease-in-out infinite;}
[data-testid="stHeaderActionElements"] {display:none;}
.doodle {position:absolute; animation:bob 4s ease-in-out infinite;}
@keyframes bob {0%,100%{transform:translateY(0) rotate(-4deg);} 50%{transform:translateY(-7px) rotate(4deg);}}
.grid4 {display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.6rem;}
.grid3 {display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:.6rem 0 1.4rem 0;}
.gridcat {display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:1rem; margin:1rem 0;}
@media (max-width:820px){.grid4{grid-template-columns:repeat(2,1fr);} .grid3{grid-template-columns:1fr;} .hero{grid-template-columns:1fr; padding:1.6rem;} .hero h1{font-size:2.2rem;}}
.card {border-radius:26px; padding:1.1rem 1.2rem; box-shadow:0 5px 0 rgba(63,53,86,.08);}
.c-lav{background:#d8d2fb;} .c-pink{background:#f6cdd0;} .c-blue{background:#cfe6fb;} .c-mint{background:#c6efdc;} .c-peach{background:#fbdcb8;} .c-white{background:#fff;}
.tag {display:inline-block; background:#fff; border-radius:999px; padding:.2rem .7rem; font-size:.76rem; font-weight:700; color:#3f3556;}
.card .val {font-size:1.9rem; font-weight:700; color:#3f3556; margin:.7rem 0 .1rem; letter-spacing:-.02em;}
.card .sub {font-size:.82rem; color:#5d5278;}
.result {border-radius:30px; padding:1.4rem 1.7rem; margin:1rem 0; display:flex; gap:1.2rem; align-items:center;}
.result.safe {background:#c6efdc; box-shadow:0 6px 0 #a9dcc4;} .result.toxic {background:#f6cdd0; box-shadow:0 6px 0 #e9aab5;}
.result h3 {margin:0 0 .25rem 0; font-size:1.5rem;} .result p {margin:0; color:#4d4262;}
.device {background:#d9ecb0; border-radius:38px; padding:1.1rem; box-shadow:0 8px 0 #bcd48a; margin:1rem 0;}
.lcd {background:#e8f1cf; border-radius:26px 26px 60px 26px; border:3px solid #c5d6a0; padding:1.1rem 1.3rem; box-shadow:inset 0 4px 10px rgba(110,140,70,.25);
  display:grid; grid-template-columns:auto 1fr; gap:1.4rem; align-items:center; font-family:'Silkscreen',monospace; color:#4a5a35;}
.lcd .pet {text-align:center; padding-right:1.4rem; border-right:3px dotted #b5c78f; font-size:.8rem;}
.lcd .row {display:flex; justify-content:space-between; gap:1rem; margin:.35rem 0; font-size:.95rem;}
.btns {display:flex; justify-content:space-between; align-items:center; padding:.9rem .6rem .1rem;}
.pill-grey {display:inline-block; width:74px; height:16px; border-radius:99px; background:#5c5f55; margin-right:.6rem; box-shadow:inset 0 3px 0 #3d3f38;}
.dot {display:inline-block; width:34px; height:34px; border-radius:50%; background:radial-gradient(circle at 35% 30%,#ffc2dc,#ee86b3); margin-left:.6rem; box-shadow:0 4px 0 #cc6a98;}
@media (max-width:640px){.lcd{grid-template-columns:1fr;} .lcd .pet{border-right:none; border-bottom:3px dotted #b5c78f; padding:0 0 .8rem;}}
.chart {display:flex; align-items:flex-end; justify-content:space-between; gap:.7rem; height:200px; padding-top:1rem;}
.chart .col {flex:1; display:flex; flex-direction:column; justify-content:flex-end; align-items:center; height:100%; font-size:.75rem; color:#5d5278; text-align:center;}
.chart .bar {width:100%; max-width:56px; border-radius:14px; min-height:8px;}
.chart .bar.hit {outline:3px solid #3f3556; outline-offset:2px;}
.chart .p {font-weight:700; color:#3f3556; margin-bottom:.3rem; font-size:.82rem;}
.chart .n {margin-top:.5rem; height:2.2em; line-height:1.1;}
.cat {min-height:150px; display:flex; flex-direction:column; justify-content:space-between;}
.cat .top {display:flex; justify-content:space-between; align-items:center;}
.cat .name {font-size:1.2rem; font-weight:700; color:#3f3556; margin:.9rem 0 .5rem;}
.cat.hit {outline:3px solid #e58aa4; outline-offset:-3px;}
.st {display:inline-block; padding:.2rem .75rem; border-radius:999px; font-size:.76rem; font-weight:700; background:#fff;}
.st.hit {color:#b3416b;} .st.ok {color:#2f8a64;}
.step .num {font-family:'Silkscreen',monospace; font-size:1.8rem; color:#3f3556;}
.step h4 {margin:.2rem 0;} .step p {margin:0; color:#4d4262; font-size:.92rem;}
.section-title {font-size:1.7rem; font-weight:700; color:#3f3556; margin:2rem 0 .3rem 0; letter-spacing:-.02em;}
.stTextArea textarea {border-radius:24px !important; border:3px solid #f1d3c6 !important; background:#fff !important; font-size:1rem; padding:1rem 1.1rem !important; color:#3f3556 !important;}
.stTextArea textarea:focus {border-color:#f4a7c3 !important; box-shadow:0 0 0 4px rgba(244,167,195,.3) !important;}
.stTextArea label p {font-weight:700; font-size:1.05rem; color:#3f3556;}
.stButton button {background:#f4a7c3; color:#4a2f4a; font-weight:700; font-size:1.05rem; border:none; border-radius:999px; padding:.75rem 2.2rem;
  box-shadow:0 6px 0 #d985a8; transition:all .1s;}
.stButton button:hover {background:#f7b6cd; color:#4a2f4a; border:none; transform:translateY(-1px);}
.stButton button:active {transform:translateY(4px); box-shadow:0 2px 0 #d985a8;}
.about p {color:#4d4262; line-height:1.65;}
</style>
"""


def html(block: str) -> None:
    """Render HTML without markdown treating indented lines as code."""
    st.markdown("".join(line.strip() for line in block.splitlines()), unsafe_allow_html=True)


# ----------------------------- model -----------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: str = str(MODEL_PATH)):
    """Load and validate the trained bundle. Returns (bundle, error_message)."""
    try:
        p = Path(path)
        if not p.exists():
            log.error("Model file not found: %s", p)
            return None, "The model file `toxic_debiased.joblib` was not found next to app.py."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn version notice; versions are pinned in requirements.txt
            bundle = joblib.load(p)
        missing = {"word_vec", "char_vec", "models", "labels"} - set(bundle)
        if missing:
            log.error("Model bundle missing keys: %s", missing)
            return None, "The model file does not have the expected structure."
        return bundle, None
    except Exception:
        log.exception("Failed to load model")
        return None, "The model could not be loaded. Check that scikit-learn matches requirements.txt."


def predict(bundle, text: str) -> dict:
    """Return {label: probability} using the same features the model was trained on."""
    features = sp.hstack(
        [bundle["word_vec"].transform([text]), bundle["char_vec"].transform([text])]
    ).tocsr()
    return {lbl: float(bundle["models"][lbl].predict_proba(features)[0, 1]) for lbl in bundle["labels"]}


def pretty(label: str):
    return META.get(label, ("🏷️", label.replace("_", " ").title()))


def validate(text: str):
    text = (text or "").strip()
    if not text:
        return None, "Please enter a comment before analyzing."
    if not any(c.isalnum() for c in text):
        return None, "That input has no readable words. Please enter a real comment."
    if len(text) > MAX_CHARS:
        return None, f"Comments are limited to {MAX_CHARS:,} characters (yours has {len(text):,})."
    return text, None


# ----------------------------- components -----------------------------
def summary_cards(text=None, probs=None, total=6):
    if probs is None:
        vals = [("🔍 Comment Analysis", "—", "Waiting for input", "c-lav"),
                ("🛡️ Safety Status", "—", "Not analyzed yet", "c-mint"),
                ("📊 Toxicity Score", "—", "Highest category score", "c-pink"),
                ("🏷️ Categories Detected", "—", f"Out of {total} categories", "c-peach")]
    else:
        hits = [k for k, v in probs.items() if v >= THRESHOLD]
        vals = [("🔍 Comment Analysis", f"{len(text.split())} words", f"{len(text)} characters", "c-lav"),
                ("🛡️ Safety Status", "Flagged" if hits else "Safe", "Review suggested" if hits else "No harmful content", "c-pink" if hits else "c-mint"),
                ("📊 Toxicity Score", f"{max(probs.values()):.0%}", "Highest category score", "c-blue"),
                ("🏷️ Categories Detected", f"{len(hits)} / {total}", ", ".join(pretty(h)[1] for h in hits) or "None", "c-peach")]
    cards = "".join(f'<div class="card {c}"><span class="tag">{a}</span><div class="val">{b}</div><div class="sub">{s}</div></div>'
                    for a, b, s, c in vals)
    html(f'<div class="grid4">{cards}</div>')


def mood_hearts(top: float) -> str:
    n = 3 if top < 0.2 else 2 if top < 0.5 else 1 if top < 0.8 else 0
    return "💗" * n + "🤍" * (3 - n)


def device(probs=None):
    """Pixel-pet gadget screen that reacts to the analysis result."""
    if probs is None:
        mood, verdict, top, flags, hearts = "happy", "WAITING", "--", "--", "💗💗💗"
    else:
        top_v = max(probs.values())
        hits = [k for k, v in probs.items() if v >= THRESHOLD]
        mood, verdict = ("sad", "TOXIC") if hits else ("happy", "SAFE")
        top, flags, hearts = f"{top_v:.0%}", f"{len(hits)}/{len(probs)}", mood_hearts(top_v)
    html(f'<div class="device"><div class="lcd"><div class="pet">{pixel_cat(mood, 92)}<div>Kiko the Mod</div></div>'
         f'<div><div class="row"><span>Verdict</span><span>{verdict}</span></div>'
         f'<div class="row"><span>Mood</span><span>{hearts}</span></div>'
         f'<div class="row"><span>Toxicity</span><span>{top}</span></div>'
         f'<div class="row"><span>Flags</span><span>{flags}</span></div></div></div>'
         '<div class="btns"><div><span class="pill-grey"></span><span class="pill-grey"></span></div>'
         '<div><span class="dot"></span><span class="dot"></span></div></div></div>')


def show_results(probs: dict):
    hits = [k for k, v in probs.items() if v >= THRESHOLD]
    if hits:
        html(f'<div class="result toxic">{pixel_cat("sad", 70)}<div><h3>⚠️ Toxic content detected</h3>'
             f'<p>The model flagged {len(hits)} categor{"y" if len(hits) == 1 else "ies"}: '
             f'{", ".join(pretty(h)[1] for h in hits)}.</p></div></div>')
    else:
        html(f'<div class="result safe">{pixel_cat("happy", 70)}<div><h3>🟢 Comment appears safe</h3>'
             '<p>🌸 <b>Looks good!</b> This comment does not appear to contain significant toxic content.</p></div></div>')
    device(probs)

    st.markdown('<div class="section-title">📊 Toxicity profile</div>', unsafe_allow_html=True)
    cols = ""
    for i, (lbl, p) in enumerate(probs.items()):
        hit = "hit" if p >= THRESHOLD else ""
        cols += (f'<div class="col"><div class="p">{p:.0%}</div>'
                 f'<div class="bar {hit}" style="height:{max(p * 100, 4):.0f}%;background:{CARD_COLORS[i % 6]}"></div>'
                 f'<div class="n">{pretty(lbl)[1]}</div></div>')
    html(f'<div class="card c-white"><div class="chart">{cols}</div></div>')

    st.markdown('<div class="section-title">🏷️ Category results</div>', unsafe_allow_html=True)
    cards = ""
    for i, (lbl, p) in enumerate(probs.items()):
        emoji, name = pretty(lbl)
        hit = p >= THRESHOLD
        cards += (f'<div class="card cat {"hit" if hit else ""}" style="background:{CARD_COLORS[i % 6]}">'
                  f'<div class="top"><span class="tag">{emoji} Category</span><span class="tag">{p:.0%}</span></div>'
                  f'<div class="name">{name}</div>'
                  f'<div><span class="st {"hit" if hit else "ok"}">{"⚠️ Detected" if hit else "✓ Not detected"}</span></div></div>')
    html(f'<div class="gridcat">{cards}</div>')


# ----------------------------- page -----------------------------
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    html(f'<div class="side-cat">{pixel_cat("happy", 84)}</div>'
         '<div class="side-title">Toxic Comment Detector</div><div class="side-sub">Understand harmful language</div>'
         '<a class="side-link" href="#home">🏠 Home</a><a class="side-link" href="#analyze">🔍 Analyze</a>'
         '<a class="side-link" href="#about">📖 About</a><a class="side-link" href="#how-it-works">ℹ️ How It Works</a>'
         '<div class="side-foot">Powered by NLP + Machine Learning</div>')

html('<div id="home"></div><div class="hero">'
     f'<div class="doodle" style="top:12px;left:46%">{STAR.format(s=26, c="#ffe08a")}</div>'
     f'<div class="doodle" style="bottom:16px;left:40%;animation-delay:1s">{HEART.format(s=24, c="#f9a8c4")}</div>'
     f'<div class="doodle" style="top:14px;right:18px;animation-delay:2s">{CLOUD.format(s=56, c="#fff")}</div>'
     f'<div><h1><span class="bottle">{poison_bottle(62)}</span>Toxic Comment Detector</h1><p>AI-powered NLP system for detecting harmful and toxic online comments</p>'
     f'<span class="badge">NLP • Machine Learning • Text Classification</span></div><div style="text-align:center">{hero_scene()}</div></div>')

bundle, load_error = load_model()
total_labels = len(bundle["labels"]) if bundle else 6
dashboard = st.empty()
with dashboard.container():
    summary_cards(total=total_labels)

html('<div id="analyze"></div>')
st.markdown('<div class="section-title">🔍 Analyze a comment</div>', unsafe_allow_html=True)

if load_error:
    st.error(f"😕 {load_error}")
else:
    comment = st.text_area("Enter a comment to analyze", height=170, placeholder="Type or paste a comment here...")
    if st.button("🔎 Analyze Comment"):
        text, problem = validate(comment)
        if problem:
            st.warning(f"✋ {problem}")
        else:
            try:
                with st.spinner("Analyzing…"):
                    probs = predict(bundle, text)
                with dashboard.container():
                    summary_cards(text, probs, total_labels)
                show_results(probs)
            except Exception:
                log.exception("Prediction failed")
                st.error("😕 Something went wrong while analyzing that comment. Please try a different one.")

html('<div id="about"></div>')
st.markdown('<div class="section-title">📖 About the Project</div>', unsafe_allow_html=True)
html('<div class="grid3 about">'
     '<div class="card c-lav"><h4>What is Toxic Comment Detection?</h4><p>It is a text classification task: reading a comment and deciding whether it contains harmful language such as insults, threats or hate.</p></div>'
     '<div class="card c-blue"><h4>How does it work?</h4><p>The text is turned into numerical word and character patterns (TF‑IDF). A trained machine-learning model then estimates how likely each type of toxicity is.</p></div>'
     '<div class="card c-peach"><h4>Why is it useful?</h4><p>Platforms and communities can use it to flag abusive comments early, helping moderators keep conversations safer and more respectful.</p></div></div>')

html('<div id="how-it-works"></div>')
st.markdown('<div class="section-title">ℹ️ How It Works</div>', unsafe_allow_html=True)
html('<div class="grid3">'
     '<div class="card step c-pink"><div class="num">01</div><h4>Enter Comment</h4><p>Type or paste any comment into the text box.</p></div>'
     '<div class="card step c-lav"><div class="num">02</div><h4>NLP Analysis</h4><p>The trained machine-learning model analyzes the text.</p></div>'
     '<div class="card step c-mint"><div class="num">03</div><h4>Get Results</h4><p>See toxicity categories and confidence scores instantly.</p></div></div>')
html('<p style="text-align:center;color:#8b80a8;font-size:.85rem;margin-top:2rem">Predictions are automated and can be wrong. Use them as guidance, not as a final judgment.</p>')
