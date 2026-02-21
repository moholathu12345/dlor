import streamlit as st
from datetime import datetime
import random

st.set_page_config(page_title="Planter - Plant Rotting Detection", page_icon="🌱", layout="wide")

# ---------- Styles (ported from your HTML) ----------
CSS = """
<style>
  :root {
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-tertiary: #ecfdf5;
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --border-color: #e2e8f0;
    --accent-primary: #166534;
    --accent-primary-hover: #14532d;
    --accent-secondary: #22c55e;
    --accent-light: #dcfce7;
    --warning: #f59e0b;
    --warning-bg: #fffbeb;
    --error: #ef4444;
    --error-bg: #fef2f2;
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
    --radius-xl: 1rem;
    --transition-fast: 150ms ease;
    --transition-normal: 300ms ease;
    --container-max: 1100px;
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  /* Streamlit page background */
  .stApp {
    font-family: var(--font-family);
    background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
    color: var(--text-primary);
  }

  /* Header */
  .planter-header {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
    color: white;
    padding: 1.1rem 0;
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-md);
    margin-bottom: 1.25rem;
  }
  .planter-header-inner {
    max-width: var(--container-max);
    margin: 0 auto;
    padding: 0 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-weight: 700;
    font-size: 1.35rem;
  }
  .logo-icon {
    width: 36px;
    height: 36px;
    background: rgba(255,255,255,0.2);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.15rem;
  }

  /* Hero */
  .hero {
    max-width: var(--container-max);
    margin: 0 auto;
    padding: 0 1.25rem;
    text-align: center;
    margin-bottom: 1.25rem;
  }
  .hero-title {
    font-size: clamp(1.75rem, 4vw, 2.25rem);
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, var(--accent-primary), var(--text-primary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 680px;
    margin: 0 auto;
  }

  /* Cards */
  .card {
    background: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-xl);
    padding: 1.25rem;
    box-shadow: var(--shadow-md);
    transition: box-shadow var(--transition-fast);
  }
  .card:hover { box-shadow: var(--shadow-lg); }

  .card-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent-primary);
    padding-bottom: 0.75rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid var(--accent-light);
  }

  /* Result card accent */
  .result-card {
    border-left: 4px solid var(--accent-secondary);
    background: linear-gradient(135deg, #ffffff 0%, var(--bg-tertiary) 100%);
  }

  .result-label {
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--accent-primary);
    margin: 0.4rem 0 0.6rem 0;
    text-align: center;
  }

  .confidence-wrap { margin: 0.5rem 0 0.25rem 0; }
  .confidence-header {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.35rem;
  }
  .confidence-bar {
    height: 10px;
    background: var(--border-color);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }
  .confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-secondary), var(--accent-primary));
    border-radius: var(--radius-lg);
    width: 0%;
  }

  .warning-box {
    background: var(--warning-bg);
    border-left: 4px solid var(--warning);
    padding: 0.75rem;
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    margin-top: 0.9rem;
    color: #92400e;
    font-weight: 600;
    font-size: 0.95rem;
  }

  .muted {
    color: var(--text-muted);
    font-size: 0.9rem;
  }

  /* Streamlit button styling */
  div.stButton > button {
    background: linear-gradient(135deg, var(--accent-primary), var(--accent-primary-hover));
    color: white;
    border: 0;
    padding: 0.75rem 1.2rem;
    border-radius: var(--radius-md);
    font-weight: 700;
    min-height: 44px;
    box-shadow: var(--shadow-md);
    transition: transform var(--transition-fast);
  }
  div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-lg);
  }

  /* History items */
  .history-item {
    display: flex;
    gap: 0.75rem;
    padding: 0.75rem;
    background: var(--bg-secondary);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
    margin-bottom: 0.6rem;
  }
  .history-thumb {
    width: 56px;
    height: 56px;
    border-radius: 0.375rem;
    object-fit: cover;
    border: 1px solid var(--border-color);
    flex-shrink: 0;
  }
  .history-label {
    font-weight: 800;
    color: var(--text-primary);
    font-size: 0.95rem;
    margin-bottom: 0.1rem;
  }
  .history-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: var(--text-muted);
  }
  .history-confidence {
    font-weight: 800;
    color: var(--accent-primary);
  }

  /* Footer */
  .footer {
    max-width: var(--container-max);
    margin: 1.75rem auto 0 auto;
    padding: 1.25rem;
    text-align: center;
    border-top: 1px solid var(--border-color);
    color: var(--text-muted);
    font-size: 0.9rem;
  }
</style>
"""

# ---------- App state ----------
if "history" not in st.session_state:
    st.session_state.history = []  # newest first, each: {img_bytes, label, conf, ts}

# ---------- Demo analysis (replace with your real model later) ----------
def demo_predict(seed_text: str):
    labels = [
        ("Healthy Leaf", 0.94),
        ("Early Rot Detected", 0.87),
        ("Fungal Infection", 0.79),
        ("Bacterial Blight", 0.72),
        ("Nutrient Deficiency", 0.65),
    ]
    seed = sum(ord(c) for c in seed_text) % len(labels)
    label, base_conf = labels[seed]

    # small deterministic jitter to feel "alive"
    rnd = random.Random(sum(ord(c) for c in seed_text) + 1337)
    conf = max(0.50, min(0.99, base_conf + rnd.uniform(-0.03, 0.03)))
    return label, conf

# ---------- Layout ----------
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="planter-header" role="banner">
      <div class="planter-header-inner">
        <div class="logo">
          <div class="logo-icon">🌱</div>
          <div>Planter</div>
        </div>
        <div style="font-weight:600; opacity:0.95;">History</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-title">Detect Plant Rotting with AI</div>
      <div class="hero-subtitle">
        Upload a leaf photo to instantly analyze for disease, rot, or decay.
        Get confidence scores and actionable insights.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.25, 0.85], gap="large")

# ---------- Left column: Upload + Results ----------
with left:
    st.markdown('<div class="card"><div class="card-title">📤 Upload Leaf Image</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag and drop or click to upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        accept_multiple_files=False,
    )

    st.markdown('<div class="muted">💡 Tip: Use a clear, well-lit photo of a single leaf for best results.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # end card

    # Preview
    if uploaded is not None:
        st.markdown('<div class="card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.image(uploaded, caption=f"{uploaded.name} • {uploaded.size/1024:.1f} KB", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Analyze button + Results
    analyze_disabled = uploaded is None
    if st.button("🔍 Analyze Image", disabled=analyze_disabled, use_container_width=False):
        with st.spinner("Analyzing image with AI model..."):
            label, conf = demo_predict(uploaded.name)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state.history.insert(
                0,
                {
                    "img_bytes": uploaded.getvalue(),
                    "label": label,
                    "conf": conf,
                    "ts": ts,
                    "filename": uploaded.name,
                },
            )
            st.session_state.history = st.session_state.history[:10]

    # Show latest result if available
    if st.session_state.history:
        latest = st.session_state.history[0]
        conf_pct = round(latest["conf"] * 100)

        st.markdown('<div class="card result-card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✅ Analysis Complete</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-label">{latest["label"]}</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="confidence-wrap">
              <div class="confidence-header">
                <span>Confidence</span>
                <span>{conf_pct}%</span>
              </div>
              <div class="confidence-bar" role="progressbar" aria-valuenow="{conf_pct}" aria-valuemin="0" aria-valuemax="100">
                <div class="confidence-fill" style="width:{conf_pct}%;"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if conf_pct < 70:
            st.markdown(
                '<div class="warning-box">⚠️ Low confidence. Retake photo with better lighting for more accurate results.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Right column: History ----------
with right:
    st.markdown('<div class="card"><div class="card-title">📋 Recent Analyses</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown('<div class="muted" style="text-align:center; padding: 1.2rem 0;">No analyses yet. Upload an image to get started.</div>', unsafe_allow_html=True)
    else:
        for item in st.session_state.history:
            conf_pct = round(item["conf"] * 100)
            st.markdown(
                f"""
                <div class="history-item">
                  <img class="history-thumb" src="data:image/png;base64,{st.image(item["img_bytes"], output_format="PNG", width=56, caption=None) if False else ""}">
                </div>
                """,
                unsafe_allow_html=True,
            )

        # The above "img in HTML" trick is messy in Streamlit.
        # So we render history items using Streamlit widgets instead, but styled containers.
        # Clear the messy block by re-rendering properly:
        st.markdown("<div></div>", unsafe_allow_html=True)  # noop

        for item in st.session_state.history:
            conf_pct = round(item["conf"] * 100)
            # Render a compact row with real image rendering
            c1, c2 = st.columns([0.28, 0.72], gap="small")
            with c1:
                st.image(item["img_bytes"], width=56)
            with c2:
                st.markdown(f"**{item['label']}**")
                st.markdown(f"<span class='muted'>{item['ts']}</span> <span class='history-confidence' style='float:right;'>{conf_pct}%</span>", unsafe_allow_html=True)
            st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1], gap="small")
    with col_a:
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # end card

# ---------- Footer ----------
st.markdown(
    """
    <div class="footer" role="contentinfo">
      <div>Planter v1.0 • AI-Powered Plant Health Monitoring</div>
      <div style="margin-top:0.25rem; font-size:0.85rem;">
        For agricultural research use. Not a substitute for professional agronomic advice.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
