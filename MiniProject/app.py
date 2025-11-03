# app.py
# Streamlit app for predicting student risk with your trained .pkl model
# Files: app.py and logistic_regression_model.pkl in the same folder.
# Run:  streamlit run app.py

import os
import io
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="🎓 Student Risk Predictor",
    page_icon="🎓",
    layout="centered",
)

# ---------- Global custom styling ----------
st.markdown(
    """
    <style>
    /* Overall app background */
    .stApp {
        background: radial-gradient(circle at top left, #1f2933 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Center the main block & give it card feel */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    /* Title styling */
    .app-header {
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .app-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        background: linear-gradient(120deg, #38bdf8, #a855f7, #f97316);
        -webkit-background-clip: text;
        color: transparent;
        text-shadow: 0 0 25px rgba(56,189,248,0.25);
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 0.98rem;
        color: #9ca3af;
    }

    /* Pill badges */
    .pill-row {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-top: 0.7rem;
        flex-wrap: wrap;
    }
    .pill {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        border: 1px solid rgba(156, 163, 175, 0.5);
        background: rgba(15, 23, 42, 0.7);
        color: #e5e7eb;
    }

    /* Glass card container */
    .glass-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.84), rgba(17,24,39,0.96));
        border-radius: 1.25rem;
        padding: 1.5rem 1.6rem;
        border: 1px solid rgba(148,163,184,0.32);
        box-shadow:
            0 18px 45px rgba(0,0,0,0.7),
            0 0 0 1px rgba(15,23,42,0.9);
        backdrop-filter: blur(18px);
    }

    /* Section headings */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .section-title span.icon {
        font-size: 1.2rem;
    }
    .section-caption {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.7rem;
    }

    /* Input labels tweak */
    label, .stRadio > label {
        font-size: 0.9rem !important;
        font-weight: 500;
        color: #e5e7eb !important;
    }

    /* Text / number inputs */
    div[data-baseweb="input"] > div {
        border-radius: 0.75rem !important;
        border: 1px solid rgba(148,163,184,0.55) !important;
        background-color: rgba(15,23,42,0.9) !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.08s ease;
    }
    div[data-baseweb="input"] > div:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px rgba(56,189,248,0.35);
        transform: translateY(-1px);
    }
    div[data-baseweb="input"] > div:focus-within {
        border-color: #a855f7 !important;
        box-shadow: 0 0 0 1px rgba(168,85,247,0.7);
    }

    /* Select / radio containers */
    div[data-baseweb="select"] > div {
        border-radius: 0.75rem !important;
        background-color: rgba(15,23,42,0.95) !important;
        border: 1px solid rgba(148,163,184,0.55) !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.08s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 1px rgba(56,189,248,0.35);
        transform: translateY(-1px);
    }

    /* Radio pills horizontal */
    .stRadio > div {
        gap: 0.35rem;
    }
    .stRadio > div > label {
        padding: 0.28rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.4);
        background: rgba(15,23,42,0.75);
        font-size: 0.8rem;
        transition: all 0.16s ease;
    }
    .stRadio > div > label:hover {
        border-color: #38bdf8;
        background: radial-gradient(circle at top, rgba(56,189,248,0.16), rgba(15,23,42,0.9));
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(15,23,42,0.8);
    }

    /* Primary button styling */
    .stButton > button {
        width: 100%;
        border-radius: 999px;
        padding: 0.6rem 1.4rem;
        border: none;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.78rem;
        background-image: linear-gradient(135deg, #22c55e, #16a34a, #22c55e);
        color: #ecfeff;
        box-shadow:
            0 14px 30px rgba(34,197,94,0.28),
            0 0 0 1px rgba(15,23,42,0.95);
        transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow:
            0 18px 40px rgba(34,197,94,0.5),
            0 0 0 1px rgba(34,197,94,0.65);
        filter: brightness(1.03);
        cursor: pointer;
    }
    .stButton > button:active {
        transform: translateY(0px) scale(0.99);
        box-shadow:
            0 8px 20px rgba(22,163,74,0.6),
            0 0 0 1px rgba(22,163,74,0.9);
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.8);
        background: radial-gradient(circle at top left, #0f172a, #020617);
        color: #e5e7eb;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.55rem 1.3rem;
        box-shadow: 0 10px 24px rgba(15,23,42,0.9);
        transition: all 0.15s ease;
    }
    .stDownloadButton > button:hover {
        border-color: #38bdf8;
        box-shadow:
            0 16px 36px rgba(15,23,42,0.95),
            0 0 0 1px rgba(56,189,248,0.45);
        transform: translateY(-1px);
    }

    /* Metric cards */
    .metric-card {
        border-radius: 1rem;
        padding: 0.85rem 1rem;
        background: radial-gradient(circle at top left, rgba(56,189,248,0.13), rgba(15,23,42,0.94));
        border: 1px solid rgba(148,163,184,0.45);
        box-shadow: 0 12px 26px rgba(15,23,42,0.95);
    }
    .metric-label {
        font-size: 0.78rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.1rem;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e5e7eb;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #a5b4fc;
    }

    /* Info strip */
    .info-strip {
        margin-top: 0.4rem;
        padding: 0.7rem 0.85rem;
        border-radius: 0.9rem;
        border: 1px dashed rgba(148,163,184,0.7);
        background: rgba(15,23,42,0.85);
        font-size: 0.78rem;
        color: #cbd5f5;
    }
    .info-strip strong {
        color: #e5e7eb;
    }

    /* Dataframe card */
    .df-wrapper {
        margin-top: 0.9rem;
        padding: 0.7rem 0.9rem 0.9rem;
        border-radius: 1rem;
        border: 1px solid rgba(148,163,184,0.45);
        background: radial-gradient(circle at top right, rgba(129,140,248,0.18), rgba(15,23,42,0.96));
        box-shadow: 0 14px 32px rgba(15,23,42,0.96);
    }
    .df-title {
        font-size: 0.86rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.4rem;
    }

    /* Messages refinement */
    .stAlert {
        border-radius: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Early-Warning — Student Risk Prediction</div>
        <div class="app-subtitle">
            Identify at-risk students early using academic performance & contextual indicators.
        </div>
        <div class="pill-row">
            <span class="pill">Logistic Regression Model</span>
            <span class="pill">Data-Driven Insights</span>
            <span class="pill">PDF Risk Report</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Resolve model path relative to this file
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, "logistic_regression_model.pkl")
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))

# ---------------- Helpers ----------------
@st.cache_resource(show_spinner=True)
def load_model(path: str):
    model = joblib.load(path)
    meta = getattr(model, "meta_", {}) or {}
    feat_names = (
        list(getattr(model, "feature_names_in_", []))
        or list(getattr(model, "feature_names_", []))
        or meta.get("feature_names", [])
    )
    classes = list(getattr(model, "classes_", []))
    return model, meta, feat_names, classes


def positive_index(classes):
    if classes and 1 in classes:
        return list(classes).index(1)
    return 1


def risk_bucket(p):
    if p >= 0.70:
        return "High"
    if p >= 0.40:
        return "Moderate"
    return "Low"


def to_int_0_100(x):
    try:
        v = int(x)
    except Exception:
        v = 0
    return max(0, min(100, v))


def make_pdf_bytes(report: dict) -> bytes:
    # Requires: pip install reportlab
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("🎓 Student Risk Prediction Report", styles["Title"]))
    elems.append(Spacer(1, 10))
    elems.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        )
    )
    elems.append(Spacer(1, 16))

    inputs = report["inputs"]
    rows = [["Field", "Value"]] + [[k, str(inputs[k])] for k in inputs]
    tbl = Table(rows, colWidths=[7 * cm, 8 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    elems.append(tbl)
    elems.append(Spacer(1, 12))

    s = report["scores"]
    tbl2 = Table(
        [
            ["Total Score", str(s["total_score"])],
            ["Average Score", f'{s["average_score"]:.2f}'],
        ],
        colWidths=[7 * cm, 8 * cm],
    )
    tbl2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    elems.append(tbl2)
    elems.append(Spacer(1, 12))

    p = report["prediction"]
    tbl3 = Table(
        [
            ["Predicted Class", p["label"]],
            ["Risk Percentage", f'{p["risk_percent"]:.1f}%'],
            ["Risk Level", p["risk_level"]],
            ["Decision Threshold", f'{p["threshold"]:.2f}'],
        ],
        colWidths=[7 * cm, 8 * cm],
    )
    tbl3.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )
    elems.append(tbl3)

    doc.build(elems)
    return buf.getvalue()


# ---------------- Load model ----------------
try:
    model, meta, feature_names, classes = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Couldn't load model at '{MODEL_PATH}'.\n\n{e}")
    st.stop()

THRESHOLD = float(meta.get("threshold", DEFAULT_THRESHOLD))
pos_idx = positive_index(classes)

# Encoders (adjust if your training used different mappings)
GENDER_MAP = {"Male": 1, "Female": 0}
PARENT_EDU_CATS = meta.get(
    "parental_education_categories",
    [
        "Some high school",
        "High school",
        "Associate's degree",
        "Some college",
        "Bachelor's degree",
        "Master's degree",
    ],
)
PARENT_MAP = {name: i for i, name in enumerate(PARENT_EDU_CATS)}

# ---------------- Input form ----------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="section-title">
        <span class="icon">🧑‍🏫</span>
        <span>Student Profile & Exam Performance</span>
    </div>
    <div class="section-caption">
        Fill in the basic demographics and latest test scores to estimate the student’s current risk level.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("risk_form"):
    # Demographic row
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    with c2:
        parent = st.selectbox(
            "Parental Level of Education",
            PARENT_EDU_CATS,
            index=min(3, len(PARENT_EDU_CATS) - 1),
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Context row
    c3, c4 = st.columns(2)
    with c3:
        lunch_label = st.radio(
            "Lunch",
            ["Had lunch", "Not had lunch"],
            index=0,
            horizontal=True,
        )
    with c4:
        test_prep_label = st.radio(
            "Test Preparation Course",
            ["Completed", "None"],
            index=1,
            horizontal=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Scores row
    st.markdown(
        """
        <div class="section-title" style="margin-top:0.15rem;">
            <span class="icon">📊</span>
            <span>Exam Scores (0 – 100)</span>
        </div>
        <div class="section-caption">
            These scores are combined with the model to estimate the probability of the student being at risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1, s2 = st.columns(2)
    with s1:
        math_score = st.number_input(
            "Maths Score",
            min_value=0,
            max_value=100,
            value=75,
            step=1,
        )
        writing_score = st.number_input(
            "Writing Score",
            min_value=0,
            max_value=100,
            value=75,
            step=1,
        )
    with s2:
        reading_score = st.number_input(
            "Reading Score",
            min_value=0,
            max_value=100,
            value=75,
            step=1,
        )
        science_score = st.number_input(
            "Science Score",
            min_value=0,
            max_value=100,
            value=75,
            step=1,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Predict Risk")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Predict ----------------
if submitted:
    # sanitize
    math_score = to_int_0_100(math_score)
    reading_score = to_int_0_100(reading_score)
    writing_score = to_int_0_100(writing_score)
    science_score = to_int_0_100(science_score)

    total_score = math_score + reading_score + writing_score + science_score
    average_score = total_score / 4.0

    # --- Build X matching model features (robust to one-hot/encoded/raw) ---
    row_values = {}

    for feat in feature_names:
        # ----- Gender -----
        if feat == "gender":
            row_values[feat] = gender
        elif feat == "gender_encoded":
            row_values[feat] = GENDER_MAP.get(gender, 0)
        elif feat.startswith("gender_"):
            cat = feat[len("gender_") :]
            row_values[feat] = 1 if gender == cat else 0

        # ----- Parental Education -----
        elif feat == "parental_level_of_education":
            row_values[feat] = parent
        elif feat == "parental_education_encoded":
            row_values[feat] = PARENT_MAP.get(parent, 0)
        elif feat.startswith("parental_level_of_education_"):
            cat = feat[len("parental_level_of_education_") :]
            row_values[feat] = 1 if parent == cat else 0

        # ----- Lunch -----
        elif feat == "lunch":
            row_values[feat] = 1 if lunch_label == "Had lunch" else 0
        elif feat.startswith("lunch_"):
            cat = feat[len("lunch_") :]
            if cat in ("1", "Had lunch", "had lunch"):
                row_values[feat] = 1 if lunch_label == "Had lunch" else 0
            elif cat in ("0", "Not had lunch", "not had lunch"):
                row_values[feat] = 1 if lunch_label == "Not had lunch" else 0
            else:
                row_values[feat] = 0

        # ----- Test Preparation Course -----
        elif feat == "test_preparation_course":
            row_values[feat] = 1 if test_prep_label == "Completed" else 0
        elif feat.startswith("test_preparation_course_"):
            cat = feat[len("test_preparation_course_") :]
            if cat in ("1", "Completed", "completed", "Yes", "yes"):
                row_values[feat] = 1 if test_prep_label == "Completed" else 0
            elif cat in ("0", "None", "none", "No", "no"):
                row_values[feat] = 1 if test_prep_label == "None" else 0
            else:
                row_values[feat] = 0

        # ----- Numerical Scores -----
        elif feat == "math_score":
            row_values[feat] = math_score
        elif feat == "reading_score":
            row_values[feat] = reading_score
        elif feat == "writing_score":
            row_values[feat] = writing_score
        elif feat == "science_score":
            row_values[feat] = science_score

        else:
            row_values[feat] = 0  # safe default

    X = pd.DataFrame([row_values], columns=feature_names)

    if X.isnull().values.any():
        nan_cols = X.columns[X.isnull().any()].tolist()
        st.error(
            f"Input data contains NaN values for features: {nan_cols}. Please check feature mapping."
        )
        st.stop()
    # --- End build X ---

    # Model probability
    try:
        proba = model.predict_proba(X)[0]
        risk_p = float(proba[pos_idx])
    except Exception as e:
        st.error(f"Prediction failed. Check feature names and encodings.\n\n{e}")
        st.stop()

    # -------- Business rule override --------
    # If average score < 45%, student is AT RISK regardless of model probability
    if average_score < 45:
        pred_label_bin = 1
    else:
        pred_label_bin = 1 if risk_p >= THRESHOLD else 0

    pred_label_txt = (
        "Student AT RISK" if pred_label_bin == 1 else "Student NOT at risk"
    )
    level = risk_bucket(risk_p)

    # ---------------- Display ----------------
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-title">
            <span class="icon">📌</span>
            <span>Prediction Summary</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Score</div>
                <div class="metric-value">{total_score}</div>
                <div class="metric-sub">out of 400</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Score</div>
                <div class="metric-value">{average_score:.2f}</div>
                <div class="metric-sub">percentage</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Risk Probability</div>
                <div class="metric-value">{risk_p*100:.1f}%</div>
                <div class="metric-sub">model estimate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    if pred_label_bin == 1:
        st.error(f"**Prediction:** {pred_label_txt}")
    else:
        st.success(f"**Prediction:** {pred_label_txt}")

    st.markdown(
        f"""
        <div class="info-strip">
            <strong>Risk Level:</strong> {level} &nbsp;&nbsp;•&nbsp;&nbsp;
            <strong>Decision Threshold:</strong> {THRESHOLD:.2f} &nbsp;&nbsp;•&nbsp;&nbsp;
            <strong>Business Rule (Avg &lt; 45% ⇒ At Risk):</strong> {"Yes" if average_score < 45 else "No"}
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_df = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Parental Education": parent,
                "Lunch": lunch_label,
                "Test Prep": test_prep_label,
                "Math": math_score,
                "Reading": reading_score,
                "Writing": writing_score,
                "Science": science_score,
                "Total": total_score,
                "Average": round(average_score, 2),
                "Prediction": pred_label_txt,
                "Risk %": round(risk_p * 100, 1),
                "Risk Level": level,
                "Rule(<45%⇒Risk)": "Yes" if average_score < 45 else "No",
            }
        ]
    )

    st.markdown(
        """
        <div class="df-wrapper">
            <div class="df-title">Snapshot of This Prediction</div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(show_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- PDF download ----------------
    report_payload = {
        "inputs": {
            "Gender": gender,
            "Parental Level of Education": parent,
            "Lunch": lunch_label,
            "Test Preparation Course": test_prep_label,
            "Maths Score": math_score,
            "Reading Score": reading_score,
            "Writing Score": writing_score,
            "Science Score": science_score,
        },
        "scores": {"total_score": total_score, "average_score": average_score},
        "prediction": {
            "label": pred_label_txt,
            "risk_percent": risk_p * 100.0,
            "risk_level": level,
            "threshold": THRESHOLD,
        },
    }

    st.markdown("<br/>", unsafe_allow_html=True)

    try:
        pdf_bytes = make_pdf_bytes(report_payload)
        st.download_button(
            label="⬇️ Download PDF Risk Report",
            data=pdf_bytes,
            file_name=f"student_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
    except ModuleNotFoundError:
        st.warning("PDF export requires `reportlab`. Install it with:  pip install reportlab")
    except Exception as e:
        st.warning(f"PDF export failed: {e}")
