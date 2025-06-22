import os
import streamlit as st # type: ignore
import requests # type: ignore
from dotenv import load_dotenv # type: ignore

# -- Configurations --
load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

QUESTIONS = [
    "How am I feeling right now?",
    "What was the best part of my day?",
    "What obstacles did I face, and how did I respond (or how might I respond next time)?",
    "List three things I’m grateful for today, and why.",
    "What did I learn today—about myself, others, or the world?",
    "How did I take care of my needs (physical, mental, social) today?",
    "Who made a positive impact on my day (even in a small way)?",
    "What frustrated or stressed me today, and what helped me cope?",
    "What surprised me or felt unexpected, and how did it affect me?",
    "What’s one intention or hope I have for tomorrow?"
]

# -- Streamlit UI --
st.set_page_config(page_title="Memory Capsule", page_icon=":memo:", layout="wide")

# Theme Toggle
if "theme" not in st.session_state:
    st.session_state.theme = "Day"
st.session_state.theme = st.sidebar.radio(
    "Theme",
    ["Day", "Night"],
    index=0,
    help="Switch between light and dark mode"
)

# Day mode CSS
DAY_CSS = """
<style>
  /* Page background */
  .main * {
    background-color: #f5f5f5 !important;
    color: #000000 !important;
  }
  .stButton>button { 
    border-radius: 0.5rem; 
    background-color: #006699 !important;
    color: white !important;
  }
  .stTextInput>div>div>input,
  .stTextArea>div>div>textarea {
    font-size: 1rem !important;
  }
</style>
"""

# Night mode CSS
NIGHT_CSS = """
<style>
/* ─── Global text ───────────────────────────────────────────────────────────── */
/* Make every bit of text in app & sidebar white */
[data-testid="stAppViewContainer"] * {
  color: #EEEEEE !important;
}
[data-testid="stSidebar"] * {
  color: #EEEEEE !important;
}

/* ─── Backgrounds ──────────────────────────────────────────────────────────── */
/* Main background */
[data-testid="stAppViewContainer"] {
  background-color: #111111 !important;
}
/* Section “cards” */
[data-testid="stBlock"] {
  background-color: #111111 !important;
}
/* Sidebar background */
[data-testid="stSidebar"] {
  background-color: #222222 !important;
}
/* Top toolbar (if present) */
[data-testid="stToolbar"] {
  background-color: #222222 !important;
}

/* ─── Buttons ──────────────────────────────────────────────────────────────── */
/* Style all standard and form-submit buttons the same way */
.stButton>button,
.stFormSubmitButton>button {
  background-color: #006699 !important;
  color: white !important;
  border-radius: 0.5rem !important;
}

/* ─── Inputs & TextAreas ───────────────────────────────────────────────────── */
/* Keep text inputs and areas dark with light text */
.stTextInput>div>div>input,
.stTextArea>div>div>textarea {
  background-color: #222222 !important;
  color: #EEEEEE !important;
  border: 1px solid #444444 !important;
}

/* ─── Smooth transitions ───────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"] {
  transition: background-color 0.3s, color 0.3s;
}
</style>
"""

# Inject the chosen theme’s CSS
if st.session_state.theme == "Day":
    st.markdown(DAY_CSS, unsafe_allow_html=True)
else:
    st.markdown(NIGHT_CSS, unsafe_allow_html=True)


st.title("Memory Capsule")

# Initialize session state
if "mode" not in st.session_state:
    st.session_state.mode = "manual"
if "content" not in st.session_state:
    st.session_state.content = ""
if "answers" not in st.session_state:
    st.session_state.answers = {q: "" for q in QUESTIONS}

# Sidebar for mode and user_id
with st.sidebar:
    st.header("Settings")
    st.session_state.mode = st.radio("Mode", ["Hand-written", "AI-assisted"])
    st.session_state.user_id = st.text_input("User ID", value="testuser")

st.header("New Entry")

if st.session_state.mode == "Hand-written":
    st.session_state.content = st.text_area(
        "Write your journal entry:", 
        value=st.session_state.content, 
        height=200
    )
else:
    st.write("Answer the following prompts:")
    cols = st.columns(2)
    for i, q in enumerate(QUESTIONS):
        col = cols[i % 2]
        st.session_state.answers[q] = col.text_input(
            q, value=st.session_state.answers[q]
        )

    
# Submit button
if st.button("Save Entry"):
    payload = {"mode": st.session_state.mode, "user_id": st.session_state.user_id}
    if st.session_state.mode == "Hand-written":
        text = st.session_state.content.strip()
        if not text:
            st.error("Please write something before saving.")
            st.stop()
        payload["content"] = text
    else:
        ans_list = [st.session_state.answers[q] for q in QUESTIONS]
        if not any(a.strip() for a in ans_list):
            st.error("Please answer at least one question.")
            st.stop()
        payload["answers"] = ans_list
    
    with st.spinner("Saving your entry..."):
        try:
            resp = requests.post(f"{API_URL}/entry", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            st.success(f"Entry saved! ID: {data['entry_id']}")
            st.markdown("### Your Entry")
            st.write(data["text"])
            st.session_state.content = ""
            for q in QUESTIONS:
                st.session_state.answers[q] = ""
        except Exception as e:
            st.error(f"Failed to save entry: {e}")
        
# -- Flashback Section --
st.markdown("---")
st.header("Flashbacks")

with st.form("flashback_form"):
    flash_q = st.text_input("Enter a flashback query (e.g. gratitude):")
    k = st.slider("How many results?", 1, 10, 3)
    submitted = st.form_submit_button("Show Flashbacks")
    if submitted:
        if not flash_q.strip():
            st.error("Please enter a query.")
        else:
            try:
                r2 = requests.get(
                    f"{API_URL}/flashback/{st.session_state.user_id}",
                    params={"q": flash_q, "k": k},
                    timeout=10
                )
                r2.raise_for_status()
                entries = r2.json()
                if not entries:
                    st.info("No flashbacks found.")
                else:
                    for e in entries:
                        st.markdown(f"**{e['entry_id']}**")
                        st.write(e["content"])
            except Exception as e:
                st.error(f"Flashback failed: {e}")

# -- Stats --
st.markdown("---")
if st.button("Show Stats"):
    try:
        r3 = requests.get(f"{API_URL}/stats/{st.session_state.user_id}", timeout=5)
        r3.raise_for_status()
        stats = r3.json()
        st.metric("Total Entries", stats["total_entries"])
        st.metric("Current Streak", stats["streak"])
        st.write("🏅 Badges:", ", ".join(stats["badges"]) or "None")
    except Exception as e:
        st.error(f"Stats fetch failed: {e}")
