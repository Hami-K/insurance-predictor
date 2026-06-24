import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Cross-Sell Likelihood",
    page_icon=None,
    layout="wide",
)

# ─────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.31.0/tabler-icons.min.css">
<style>
    :root {
        --bg:       #F7F7F5;
        --surface:  #FFFFFF;
        --ink:      #1A1A18;
        --muted:    #6B7280;
        --line:     #E5E7EB;
        --teal:     #0D7A5F;
        --teal-bg:  #ECFBF4;
        --amber:    #92480C;
        --amber-bg: #FEF3E2;
        --red:      #9B1C1C;
        --red-bg:   #FEF2F2;
        --mono:     'IBM Plex Mono', monospace;
    }

    html, body, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    section[data-testid="stSidebar"] { background: var(--bg) !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    
    /* Strict scoping for our layouts */
    .custom-layout, .custom-layout * { font-family: 'Inter', sans-serif !important; }

    /* ── hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* ── page title ── */
    .page-eyebrow {
        font-size: 11px;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 500;
        margin-bottom: 4px;
    }
    .page-title {
        font-size: 22px;
        font-weight: 500;
        color: var(--ink);
        line-height: 1.2;
        margin-bottom: 4px;
    }
    .page-sub {
        font-size: 14px;
        color: var(--muted);
        margin-bottom: 0;
        max-width: 560px;
    }

    /* ── section headers ── */
    .section-head {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        padding-bottom: 8px;
        border-bottom: 0.5px solid var(--line);
        margin-bottom: 14px;
        margin-top: 22px;
    }

    /* ── result panel ── */
    .result-panel {
        background: var(--surface);
        border: 0.5px solid var(--line);
        border-radius: 12px;
        padding: 22px 20px;
    }
    .panel-title {
        font-size: 11px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--muted);
        margin-bottom: 14px;
    }

    /* ── gauge SVG wrapper ── */
    .gauge-wrap { text-align: center; margin-bottom: 4px; }
    .prob-num {
        font-family: var(--mono) !important;
        font-size: 38px;
        font-weight: 500;
        color: var(--ink);
        line-height: 1;
        text-align: center;
    }
    .prob-cap { font-size: 12px; color: var(--muted); text-align: center; margin-top: 4px; margin-bottom: 14px; }

    /* ── verdict badge ── */
    .verdict {
        width: 100%;
        text-align: center;
        font-size: 12px;
        font-weight: 500;
        padding: 6px 0;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .verdict-high { background: var(--teal-bg);  color: var(--teal);  }
    .verdict-mid  { background: var(--amber-bg); color: var(--amber); }
    .verdict-low  { background: var(--red-bg);   color: var(--red);   }

    /* ── driver items ── */
    .driver-list { border-top: 0.5px solid var(--line); padding-top: 10px; }
    .driver-item {
        display: flex;
        gap: 8px;
        align-items: flex-start;
        padding: 7px 0;
        border-bottom: 0.5px solid var(--line);
        font-size: 12px;
        color: var(--ink); 
        line-height: 1.45;
    }
    .driver-item:last-child { border-bottom: none; }
    .di-pos i { color: var(--teal); }
    .di-neg i { color: var(--red);  }
    .di-neu i { color: var(--muted); }

    /* ── factor bars ── */
    .factor-panel {
        background: var(--surface); 
        border: 0.5px solid var(--line);
        border-radius: 8px;
        padding: 16px;
        margin-top: 18px;
    }
    .fp-title { font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); margin-bottom: 12px; }
    .factor-row { display: flex; align-items: center; margin-bottom: 10px; gap: 12px; font-size: 12px; }
    .factor-label { color: var(--ink); width: 150px; flex-shrink: 0; font-weight: 400; }
    .factor-track {
        flex: 1;
        height: 8px;
        background: #E5E7EB; 
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }
    .factor-fill  { height: 100%; border-radius: 999px; }
    .factor-val { font-family: var(--mono) !important; font-size: 11px; color: var(--muted); width: 35px; text-align: right; }

    /* ── FIX: Force Expander Title and internal components to be visible against forced light background ── */
    div[data-testid="stExpander"] details summary p {
        color: var(--ink) !important;
    }
    div[data-testid="stExpander"] [data-testid="stWidgetLabel"] p {
        color: var(--muted) !important;
    }
    
    /* ── streamlit widget overrides ── */
    div[data-testid="stRadio"] > label,
    div[data-testid="stSlider"] > label,
    div[data-testid="stNumberInput"] > label,
    div[data-testid="stSelectbox"] > label {
        font-size: 13px !important;
        font-weight: 400 !important;
        color: var(--muted) !important;
    }
    
    div[role="radiogroup"] label p {
        color: var(--ink) !important;
        font-size: 13px !important;
    }

    hr { border-color: var(--line) !important; margin: 18px 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────
from pathlib import Path

@st.cache_resource
def load_artifacts():
    try:
        BASE_DIR = Path(__file__).parent

        model = joblib.load(BASE_DIR / "exports" / "lgb_model.sav")
        scaler = joblib.load(BASE_DIR / "exports" / "scaler.sav")
        premium_bins = joblib.load(BASE_DIR / "exports" / "premium_bins.sav")

        return model, scaler, premium_bins, True

    except Exception as e:
        st.error(f"Error loading model files: {e}")

        return (
            None,
            None,
            [2630, 24875, 31785, 39526, 56295, 540165],
            False,
        )

model, scaler, premium_bins, MODEL_LOADED = load_artifacts()

MOST_COMMON_REGION_CODE  = 28
MOST_COMMON_CHANNEL_CODE = 26

# ─────────────────────────────────────────
# HEURISTIC SCORE (used when model not loaded)
# ─────────────────────────────────────────
def heuristic_score(age, prem, vin, vehicle_age_enc, has_damage, has_dl, prev_insured, is_male):
    score = 0.18
    if has_damage:       score += 0.30
    if not prev_insured: score += 0.18
    if vehicle_age_enc == 2: score += 0.10
    if vehicle_age_enc == 1: score += 0.06
    if 40 < age < 60: score += 0.06
    if 25 <= age <= 45: score += 0.04
    if has_dl:        score += 0.04
    if is_male:       score += 0.02
    if prem > 50000:  score += 0.05
    if prem < 10000:  score -= 0.04
    if vin > 200:     score += 0.03
    if prev_insured:  score -= 0.20
    return float(np.clip(score, 0.03, 0.97))

# ─────────────────────────────────────────
# GAUGE SVG
# ─────────────────────────────────────────
def gauge_svg(probability: float) -> str:
    TOTAL_ARC = 204.0
    offset = TOTAL_ARC * (1 - probability)
    if probability >= 0.60:
        stroke = "#0D7A5F"
    elif probability >= 0.35:
        stroke = "#92480C"
    else:
        stroke = "#9B1C1C"
    return f"""
    <svg viewBox="0 0 160 90" width="160" height="90" xmlns="http://www.w3.org/2000/svg">
      <path d="M 15 85 A 65 65 0 0 1 145 85" fill="none" stroke="#E5E7EB" stroke-width="10" stroke-linecap="round"/>
      <path d="M 15 85 A 65 65 0 0 1 145 85" fill="none" stroke="{stroke}" stroke-width="10" stroke-linecap="round" stroke-dasharray="204" stroke-dashoffset="{offset:.1f}"/>
    </svg>
    """

# ─────────────────────────────────────────
# FACTOR BARS HTML
# ─────────────────────────────────────────
def factor_bars(dmg_pct, age_pct, veh_pct, prem_pct):
    rows = [
        ("Vehicle damage signal", dmg_pct, "#0D7A5F"),
        ("Age profile", age_pct, "#1D4ED8"),
        ("Vehicle maturity", veh_pct, "#92480C"),
        ("Premium tier", prem_pct, "#6B7280"),
    ]

    # Combined layout wrapper inside a single string variable to prevent raw rendering bugs
    html = '<div class="custom-layout"><div class="factor-panel"><div class="fp-title">Key Input Factors</div>'
    for label, pct, color in rows:
        html += f'<div class="factor-row"><div class="factor-label">{label}</div><div class="factor-track"><div class="factor-fill" style="width:{pct}%; background:{color};"></div></div><div class="factor-val">{pct}%</div></div>'
    html += "</div></div>"
    return html

# ─────────────────────────────────────────
# DRIVER SIGNALS HTML
# ─────────────────────────────────────────
def driver_signals(has_damage, prev_insured, vehicle_age_enc, age):
    items = []
    pos = ("<i class='ti ti-trending-up'></i>", "di-pos")
    neg = ("<i class='ti ti-shield-check'></i>", "di-neg")
    neu = ("<i class='ti ti-minus'></i>",        "di-neu")

    if has_damage and not prev_insured:
        items.append((pos, "Damaged vehicle with no existing coverage — strong cross-sell signal"))
    elif has_damage:
        items.append((pos, "Past vehicle damage increases interest in protection"))
    if prev_insured:
        items.append((neg, "Already insured — less likely to switch or add coverage"))
    if vehicle_age_enc == 2:
        items.append((pos, "Older vehicles tend to raise risk awareness and openness to offers"))
    elif vehicle_age_enc == 0:
        items.append((neg, "New vehicle owners often have coverage handled at point of sale"))
    if 40 < age < 60:
        items.append((pos, "Mid-career age bracket shows historically higher take-up rates"))
    if not has_damage and not prev_insured:
        items.append((neu, "No damage history reduces urgency, but no existing policy leaves an opening"))
    if not items:
        items.append((neu, "Profile is average across most factors — no single signal dominates"))

    rows = "".join(f"<div class='driver-item {cls}'>{icon}<span>{text}</span></div>" for (icon, cls), text in items[:4])
    return f"<div class='driver-list'>{rows}</div>"

# ─────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────
st.markdown('<div class="custom-layout"><div class="page-eyebrow">Cross-sell model</div><div class="page-title">Will this customer respond?</div><div class="page-sub">Fill in the customer\'s details. The model estimates how likely they are to take up a vehicle insurance offer based on patterns from past customers.</div></div>', unsafe_allow_html=True)

if not MODEL_LOADED:
    st.info(
        "Model files not found (lgb_model.sav, scaler.sav, premium_bins.sav). Showing a heuristic estimate.",
        icon="ℹ️",
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LAYOUT: FORM (left) | RESULT PANEL (right)
# ─────────────────────────────────────────
form_col, result_col = st.columns([1.6, 1], gap="large")

with form_col:
    st.markdown('<div class="custom-layout"><div class="section-head">Customer</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    with c2:
        age = st.slider("Age", 18, 85, 35)

    c3, c4 = st.columns(2)
    with c3:
        driving_license = st.radio("Driving license", ["Yes", "No"], horizontal=True)
    with c4:
        previously_insured = st.radio("Previously insured", ["Yes", "No"], index=1, horizontal=True)

    st.markdown('<div class="custom-layout"><div class="section-head">Vehicle</div></div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        vehicle_age = st.selectbox("Vehicle age", ["Under 1 year", "1 to 2 years", "Over 2 years"], index=1)
    with c6:
        vehicle_damage = st.radio("Past damage?", ["Yes", "No"], index=1, horizontal=True)

    st.markdown('<div class="custom-layout"><div class="section-head">Policy & history</div></div>', unsafe_allow_html=True)
    annual_premium = st.slider("Annual premium (INR)", min_value=2_630, max_value=100_000, value=30_000, step=500, format="₹%d")
    vintage = st.slider("Customer tenure (days)", 0, 300, 150)

    with st.expander("Advanced — internal codes (optional)"):
        region_code = st.number_input("Region code (0–52)", min_value=0.0, max_value=52.0, value=float(MOST_COMMON_REGION_CODE), step=1.0)
        policy_sales_channel = st.number_input("Sales channel code", min_value=0.0, value=float(MOST_COMMON_CHANNEL_CODE), step=1.0)

    # ── ENCODING ──────────────────────────
    gender_enc        = 1 if gender == "Male" else 0
    dl_enc            = 1 if driving_license == "Yes" else 0
    pi_enc            = 1 if previously_insured == "Yes" else 0
    dmg_enc           = 1 if vehicle_damage == "Yes" else 0
    veh_age_map       = {"Under 1 year": 0, "1 to 2 years": 1, "Over 2 years": 2}
    veh_age_enc       = veh_age_map[vehicle_age]
    age_dmg_interact  = veh_age_enc * dmg_enc

    premium_quintile = pd.cut([annual_premium], bins=premium_bins, labels=False, include_lowest=True)[0]
    premium_quintile = 0 if pd.isna(premium_quintile) else int(premium_quintile)

    input_df = pd.DataFrame({
        "Gender":               [gender_enc],
        "Age":                  [age],
        "Driving_License":      [dl_enc],
        "Region_Code":          [region_code],
        "Previously_Insured":   [pi_enc],
        "Vehicle_Age":          [veh_age_enc],
        "Vehicle_Damage":       [dmg_enc],
        "Annual_Premium":       [annual_premium],
        "Policy_Sales_Channel": [policy_sales_channel],
        "Vintage":              [vintage],
        "Age_Damage_Interaction": [age_dmg_interact],
        "Premium_Quintile":     [premium_quintile],
    })

    # ── FACTOR BAR PERCENTAGES ─────────────
    has_damage   = vehicle_damage == "Yes"
    prev_insured = previously_insured == "Yes"
    dmg_pct  = int(95 if (has_damage and not prev_insured) else (55 if has_damage else 10))
    age_pct  = int(((min(age, 65) - 18) / 47) * 80 + 10)
    veh_pct  = [20, 55, 85][veh_age_enc]
    prem_pct = int(min((annual_premium / 100_000) * 100, 100))

    # FIXED: Rendered via a clean string format cleanly scoped
    st.markdown(factor_bars(dmg_pct, age_pct, veh_pct, prem_pct), unsafe_allow_html=True)

    with st.expander("🔍 View raw model input"):
        st.dataframe(input_df, use_container_width=True)

with result_col:
    if MODEL_LOADED:
        scaled = input_df.copy()
        num_cols = ["Age", "Annual_Premium", "Vintage", "Region_Code", "Policy_Sales_Channel"]
        scaled[num_cols] = scaler.transform(scaled[num_cols])
        probability = float(model.predict_proba(scaled)[0][1])
    else:
        probability = heuristic_score(
            age, annual_premium, vintage, veh_age_enc,
            has_damage, driving_license == "Yes", prev_insured, gender == "Male"
        )

    pct = int(round(probability * 100))

    if probability >= 0.60:
        verdict_class, verdict_text = "verdict-high", "Strong candidate"
        tip = "High confidence — a strong candidate for a personal call or tailored offer."
    elif probability >= 0.35:
        verdict_class, verdict_text = "verdict-mid", "Moderate interest"
        tip = "Moderate interest — a targeted email or follow-up may help convert them."
    else:
        verdict_class, verdict_text = "verdict-low", "Low priority"
        tip = "Low interest expected — may not be worth heavy sales investment right now."

    panel_html = f"""
    <div class="custom-layout">
        <div class="result-panel">
          <div class="panel-title">Response likelihood</div>
          <div class="gauge-wrap">{gauge_svg(probability)}</div>
          <div class="prob-num">{pct}%</div>
          <div class="prob-cap">probability of response</div>
          <div class="verdict {verdict_class}">{verdict_text}</div>
          <hr style="margin:14px 0 12px !important;">
          <div class="panel-title">Signals driving this score</div>
          {driver_signals(has_damage, prev_insured, veh_age_enc, age)}
        </div>
    </div>
    """
    st.markdown(panel_html, unsafe_allow_html=True)
    st.markdown(f"<span style='color:black;'>💡 {tip}</span>", unsafe_allow_html=True)