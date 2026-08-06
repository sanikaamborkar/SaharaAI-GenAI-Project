"""
app.py

Sahara AI's Streamlit entry point. Replicates a dark glassmorphism
dashboard: chat on the left, live session analytics (safety status,
detected emotion, distress trajectory chart) and system logs on the
right. Session memory lives in st.session_state and is threaded into
every crew.run_pipeline() call as conversation_history.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from crew import run_pipeline

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LOG_FILE = "sahara_ai.log"

# Countries with actual helpline data in helpline_tool.py's CSV — keeping
# this list in sync means Referral's lookup will reliably find a match
# instead of silently falling back to the generic "no data" message.
SUPPORTED_COUNTRIES = [
    "India", "United States", "United Kingdom", "Australia", "Canada",
    "New Zealand", "Ireland", "South Africa", "Singapore", "Philippines",
    "Nigeria", "Pakistan", "Bangladesh", "Kenya", "Malaysia",
]

RISK_SCORE_MAP = {"LOW": 2, "MODERATE": 5, "HIGH": 9}

STATUS_COLOR = {"LOW": "#4ade80", "MODERATE": "#fbbf24", "HIGH": "#f87171"}
STATUS_ICON = {"LOW": "🛡️", "MODERATE": "⚠️", "HIGH": "🚨"}

st.set_page_config(page_title="Sahara AI", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# LOGGING — file is cleared once per browser session, appended to after that
# ---------------------------------------------------------------------------
if "log_initialized" not in st.session_state:
    with open(LOG_FILE, "w") as f:
        f.write("--- Sahara AI System Session Started ---\n")
    st.session_state.log_initialized = True

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # re-applies config across Streamlit reruns
)
logger = logging.getLogger("sahara_ai")


def get_live_logs() -> str:
    """Reads the last ~3000 chars of the log file."""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return f.read()[-3000:]
        except Exception:
            return "Loading logs..."
    return "Initializing system logs..."


def generate_plot(history: list):
    """Matplotlib distress-trajectory chart, dark theme, color-coded by latest score."""
    plt.close("all")
    fig = plt.figure(figsize=(6, 3.2))
    fig.patch.set_alpha(0.0)
    ax = plt.gca()
    ax.set_facecolor("#0f172a")
    ax.patch.set_alpha(0.3)

    if not history:
        plt.text(0.5, 0.5, "Awaiting User Input...", ha="center", va="center",
                  transform=ax.transAxes, color="#64748b", fontsize=10)
        plt.axis("off")
    else:
        x = list(range(1, len(history) + 1))
        y = history
        last_score = history[-1]

        line_color = "#34d399"  # emerald green
        if last_score > 4:
            line_color = "#fbbf24"  # amber
        if last_score > 7:
            line_color = "#f87171"  # red

        plt.plot(x, y, marker="o", linestyle="-", color=line_color,
                  linewidth=2, markersize=5)
        plt.fill_between(x, y, color=line_color, alpha=0.1)
        plt.title("Real-Time Distress Tracking", color="#e2e8f0",
                   fontsize=10, fontweight="600", pad=10)
        plt.ylim(0, 10.5)
        plt.grid(True, linestyle="--", alpha=0.1, color="white")
        ax.tick_params(axis="x", colors="#64748b", labelsize=8)
        ax.tick_params(axis="y", colors="#64748b", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SESSION STATE — this IS the memory. Persists across reruns within one
# browser session; resets if the browser tab is closed or "Reset" is clicked.
# ---------------------------------------------------------------------------
_defaults = {
    "messages": [],            # [{"role": "user"/"assistant", "content": "..."}]
    "distress_history": [],
    "msg_count": 0,
    "current_risk": "LOW",
    "last_emotion": "Neutral",
    "max_distress": 0,
    "country": "India",
}
for _key, _val in _defaults.items():
    st.session_state.setdefault(_key, _val)

# ---------------------------------------------------------------------------
# CSS — dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400&display=swap');

.stApp {
    background-color: #020617 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}
.stApp, .stApp p, .stApp span, .stApp label, .stApp div {
    color: #e2e8f0;
}
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
}
section[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #0f172a !important;  /* keep dropdown's own text dark, it sits on a white control */
}
[data-testid="stCaptionContainer"], .stCaption {
    color: #94a3b8 !important;
}

/* Chat input box — text set to dark since the input renders on a light
   background regardless of container styling; dark text stays visible. */
[data-testid="stChatInput"] textarea {
    background-color: #ffffff !important;
    color: #0f172a !important;
    caret-color: #0f172a !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #64748b !important;
}

.header-container {
    text-align: center;
    padding: 30px 20px 20px 20px;
    background: radial-gradient(circle at 50% -20%, #1e293b 0%, #020617 80%);
    border-bottom: 1px solid #1e293b;
    margin-bottom: 20px;
}
.brand-title {
    font-size: 3rem;
    font-weight: 800;
    color: #38bdf8 !important;
    text-shadow: 0 0 30px rgba(56, 189, 248, 0.5);
    margin-bottom: 6px;
}
.brand-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 15px;
}
.stat-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 15px 10px;
    text-align: center;
    backdrop-filter: blur(5px);
}
.stat-label { font-size: 0.7rem; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 5px; }
.stat-value { font-size: 1.1rem; font-weight: 700; color: #f8fafc; }

[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid #1e293b !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px);
}

.footer-section {
    margin-top: 40px;
    padding: 20px 0;
    border-top: 1px solid #1e293b;
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="brand-title">Sahara AI</div>
    <div class="brand-subtitle">Advanced Mental Health &amp; Safety AI System</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.session_state.country = st.selectbox(
        "Your country (for crisis helplines)",
        SUPPORTED_COUNTRIES,
        index=SUPPORTED_COUNTRIES.index(st.session_state.country),
    )
    st.caption(
        f"Currently set to **{st.session_state.country}**. Only used if a "
        f"message is classified HIGH risk, to show the correct regional "
        f"helpline — change this anytime before that happens."
    )
    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        for _key, _val in _defaults.items():
            st.session_state[_key] = _val
        st.rerun()

# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------
chat_col, dash_col = st.columns([3, 2], gap="large")

with chat_col:
    st.markdown("#### 💬 Chatbot")

    chat_container = st.container(height=500)
    with chat_container:
        for msg in st.session_state.messages:
            avatar = "🧑" if msg["role"] == "user" else "🛡️"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    prompt = st.chat_input("Type your message...")

    if prompt:
        # History BEFORE this turn — what Planner/Worker should see as context
        history_before = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Sahara AI is thinking..."):
            result = run_pipeline(
                prompt,
                conversation_history=history_before,
                country=st.session_state.country,
            )

        reply = result["reply"]
        blocked = result["blocked"]
        risk = result["risk_level"]
        debug = result.get("debug", {})
        emotion = debug.get("primary_emotion")

        prefix = ""
        if blocked:
            prefix = "🛡️ **Input Guardrail:** "
        elif risk == "HIGH":
            prefix = "🚨 "

        final_reply = prefix + reply
        st.session_state.messages.append({"role": "assistant", "content": final_reply})

        score = 0 if blocked else RISK_SCORE_MAP.get(risk, 0)
        if score > 0:
            st.session_state.distress_history.append(score)
        st.session_state.msg_count += 1
        if not blocked:
            st.session_state.current_risk = risk
            st.session_state.last_emotion = emotion or st.session_state.last_emotion
        st.session_state.max_distress = max(st.session_state.max_distress, score)

        logger.info(f"User input processed. Risk: {risk} | Blocked: {blocked}")

        st.rerun()

with dash_col:
    tab1, tab2 = st.tabs(["📊 Session Analytics", "⚙️ System Logs"])

    with tab1:
        risk = st.session_state.current_risk
        color = STATUS_COLOR.get(risk, "#4ade80")
        icon = STATUS_ICON.get(risk, "🛡️")

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">SAFETY STATUS</div>
                <div class="stat-value" style="color:{color};">{icon} {risk}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">DETECTED EMOTION</div>
                <div class="stat-value" style="color:#bfdbfe;">{st.session_state.last_emotion.title()}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">PEAK DISTRESS</div>
                <div class="stat-value">{st.session_state.max_distress}<span style="font-size:0.6em;color:#64748b;">/10</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">INTERACTIONS</div>
                <div class="stat-value">{st.session_state.msg_count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Emotional Trajectory")
        fig = generate_plot(st.session_state.distress_history)
        st.pyplot(fig)

    with tab2:
        st.markdown("##### Agent Internal Monologue")
        st.code(get_live_logs(), language="log")
        st.caption("Refreshes on each new message.")

st.markdown("""
<div class="footer-section">
    <strong>Sahara AI</strong><br>
    <em>Provides emotional support — not a replacement for professional medical care.</em>
</div>
""", unsafe_allow_html=True)