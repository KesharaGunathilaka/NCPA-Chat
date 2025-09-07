import streamlit as st
from rag import generate_answer
import re

st.set_page_config(page_title="NCPA Chat", layout="wide")
st.title("NCPA — Ask about child protection")

lang = st.selectbox(
    "Language", ["English", "සිංහල (Sinhala)", "தமிழ் (Tamil)"], index=0)
lang_map = {"English": "en", "සිංහල (Sinhala)": "si", "தமிழ் (Tamil)": "ta"}

if "history" not in st.session_state:
    st.session_state.history = []


def detect_urgent(q):
    # simple keyword-based detection; extend with model classifier
    urgent_keywords = ["abuse", "sexual", "rape", "emergency",
                       "help me", "abused", "child beaten", "immediate danger"]
    qlow = q.lower()
    return any(k in qlow for k in urgent_keywords)


with st.form("qform"):
    q = st.text_area("Ask a question", height=120)
    submitted = st.form_submit_button("Send")
if submitted and q.strip():
    is_urgent = detect_urgent(q)
    if is_urgent:
        # show hotline banner
        st.error("If this is an emergency or ongoing abuse, call the 1929 helpline immediately or local emergency services. 1929 is the NCPA helpline. See Contact Us on the official site.")
        # store event in session (do not log sensitive PII)
    answer = generate_answer(q, language=lang_map[lang])
    st.session_state.history.append((q, answer))
for q, a in reversed(st.session_state.history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    st.markdown("---")
