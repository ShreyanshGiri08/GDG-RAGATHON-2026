import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import json
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from llm_extractor import extract_profile
from regression_model import predict, train_and_save
from rag_engine import (
    build_vector_store,
    retrieve_experiences,
    smart_experience_matcher,
    generate_rag_response,
)
from resume_parser import extract_text_from_bytes

# ────────────────────────────────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PlacementAI — Predictor & Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

  .main { background: #0d0f14; color: #e8e8f0; }
  .block-container { padding: 2rem 3rem; }

  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #7c6af7, #48c0c0, #f7c96a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 0.3rem;
  }
  .hero-sub {
    color: #888;
    font-size: 1.05rem;
    margin-bottom: 2rem;
  }

  .score-card {
    background: linear-gradient(135deg, #1a1d2e, #212438);
    border: 1px solid #2e3260;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
  }
  .score-number {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    color: #7c6af7;
    line-height: 1;
  }
  .score-label { color: #aaa; font-size: 0.9rem; margin-top: 0.3rem; }

  .exp-card {
    background: #141620;
    border: 1px solid #252840;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.6rem 0;
    transition: border-color 0.2s;
  }
  .exp-card:hover { border-color: #7c6af7; }
  .exp-rank {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    color: #7c6af7;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.2rem;
  }
  .exp-company {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8e8f0;
  }
  .exp-role { color: #888; font-size: 0.9rem; margin-bottom: 0.6rem; }
  .exp-text { color: #ccc; font-size: 0.88rem; line-height: 1.6; }
  .badge {
    display: inline-block;
    background: #1e2040;
    border: 1px solid #3a3e6e;
    border-radius: 100px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #9090c0;
    margin: 2px;
  }
  .badge-hard { border-color: #6e1e1e; color: #c07070; background: #2a1414; }
  .badge-medium { border-color: #6e5a1e; color: #c0a870; background: #2a2214; }

  .chat-bubble-user {
    background: #1e2040;
    border: 1px solid #3a3e6e;
    border-radius: 14px 14px 4px 14px;
    padding: 0.8rem 1.2rem;
    margin: 0.4rem 0;
    color: #e8e8f0;
    max-width: 85%;
    margin-left: auto;
  }
  .chat-bubble-bot {
    background: #141820;
    border: 1px solid #252840;
    border-radius: 14px 14px 14px 4px;
    padding: 0.8rem 1.2rem;
    margin: 0.4rem 0;
    color: #ccc;
    max-width: 85%;
  }

  .stButton > button {
    background: linear-gradient(135deg, #7c6af7, #5a4fd4);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e8e8f0;
    border-left: 3px solid #7c6af7;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
  }

  .stTextArea textarea {
    background: #141620 !important;
    color: #e8e8f0 !important;
    border: 1px solid #2e3260 !important;
    border-radius: 10px !important;
  }
  .stFileUploader {
    border: 2px dashed #2e3260 !important;
    border-radius: 12px !important;
    background: #0d0f14 !important;
  }
  div[data-testid="stSidebar"] {
    background: #0a0c12 !important;
    border-right: 1px solid #1a1d2e;
  }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Session state init
# ────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "profile" not in st.session_state:
    st.session_state.profile = None
if "score" not in st.session_state:
    st.session_state.score = None
if "experiences" not in st.session_state:
    st.session_state.experiences = []
if "mentor_advice" not in st.session_state:
    st.session_state.mentor_advice = ""
if "vector_store_built" not in st.session_state:
    st.session_state.vector_store_built = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False


# ────────────────────────────────────────────────────────────────────────────
# Init resources once
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def initialize_system():
    metrics = train_and_save()
    build_vector_store()
    return metrics


with st.spinner("🚀 Initializing AI systems…"):
    metrics = initialize_system()

# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.5rem">🎓 PlacementAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="font-size:0.8rem">Predictor & Mentor · RAGATHON 2026</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model Performance**")
    st.metric("R² Score", f"{metrics['r2']:.4f}", help="Gradient Boosting Regressor")
    st.metric("MSE", f"{metrics['mse']:.4f}")

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("```\nChat/Resume Input\n    ↓\nLLM Extraction (Claude)\n    ↓\nRegression Model\n    ↓\nRAG + Cosine Matcher\n    ↓\nMentor Advice\n```")

    st.markdown("---")
    if st.button("🔄 Reset Session"):
        for key in ["messages", "profile", "score", "experiences", "mentor_advice"]:
            st.session_state[key] = [] if key in ["messages", "experiences"] else None if key != "mentor_advice" else ""
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# Main content
# ────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Placement Predictor & Mentor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered readiness scoring + personalized senior interview experience recommendations</div>', unsafe_allow_html=True)

# ── Input Mode Tabs ──────────────────────────────────────────────────────────
tab_chat, tab_resume, tab_form = st.tabs(["💬 Chat Mode", "📄 Resume Upload (Bonus)", "✏️ Quick Form"])

collected_text = None

# Tab 1: Chat
with tab_chat:
    st.markdown('<div class="section-header">Chat with the AI Profiler</div>', unsafe_allow_html=True)
    st.caption("Tell the bot about your CGPA, tech stack, projects, internships, open-source work, and communication skills.")

    chat_container = st.container(height=300)
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_msg = st.text_input("Your message", placeholder="e.g. My CGPA is 8.5, I know Python and AWS…", key="chat_input", label_visibility="collapsed")
    with col_btn:
        send_btn = st.button("Send", key="send_chat")

    if send_btn and user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Got it! Tell me more — or click **Analyze Profile** below when you're ready.",
        })
        st.rerun()

    if st.session_state.messages:
        combined = "\n".join(m["content"] for m in st.session_state.messages if m["role"] == "user")
        if st.button("🔬 Analyze Profile from Chat", key="analyze_chat"):
            collected_text = combined

# Tab 2: Resume Upload (BONUS)
with tab_resume:
    st.markdown('<div class="section-header">📄 Resume Parser <span style="color:#f7c96a;font-size:0.8rem">+15 Bonus Points</span></div>', unsafe_allow_html=True)
    st.caption("Upload your resume (PDF or DOCX) and we'll extract your profile automatically.")

    uploaded_file = st.file_uploader(
        "Drop your resume here",
        type=["pdf", "docx"],
        help="PDF or DOCX files only",
    )

    if uploaded_file:
        st.success(f"✅ Uploaded: **{uploaded_file.name}** ({uploaded_file.size // 1024} KB)")
        if st.button("🔬 Analyze Resume", key="analyze_resume"):
            with st.spinner("Extracting text from resume…"):
                try:
                    file_bytes = uploaded_file.read()
                    resume_text = extract_text_from_bytes(file_bytes, uploaded_file.name)
                    collected_text = resume_text
                    st.success(f"Extracted {len(resume_text)} characters from resume.")
                except Exception as e:
                    st.error(f"Error parsing resume: {e}")

# Tab 3: Quick Form
with tab_form:
    st.markdown('<div class="section-header">Quick Profile Form</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cgpa = st.slider("CGPA (10-point scale)", 5.0, 10.0, 7.5, 0.1)
        dsa = st.slider("DSA Skill (1–10)", 1, 10, 6)
        projects = st.slider("Project Quality (1–10)", 1, 10, 6)
        experience = st.slider("Internship Experience (1–10)", 1, 10, 4)
    with col2:
        opensource = st.radio("Open-Source Contributions?", ["Yes", "No"])
        soft_skills = st.slider("Soft Skills / Communication (1–10)", 1, 10, 7)
        tech_stack_raw = st.text_input("Your Tech Stack (comma-separated)", placeholder="Python, Java, AWS, React")

    if st.button("🔬 Predict from Form", key="analyze_form"):
        techs = [t.strip() for t in tech_stack_raw.split(",") if t.strip()]
        tech_score = min(10, max(1, len(techs) * 1.5)) if techs else 5
        manual_profile = {
            "Academic_Score": cgpa,
            "DSA_Skill": dsa,
            "Project_Quality": projects,
            "Experience_Score": experience,
            "OpenSource_Value": 10 if opensource == "Yes" else 1,
            "Soft_Skills": soft_skills,
            "Tech_Stack_Score": int(tech_score),
            "raw_tech_stack": techs,
        }
        st.session_state.profile = manual_profile
        with st.spinner("Computing readiness score…"):
            st.session_state.score = predict(manual_profile)
        with st.spinner("Retrieving relevant experiences…"):
            st.session_state.experiences = smart_experience_matcher(techs, top_k=3)
        with st.spinner("Generating mentor advice…"):
            st.session_state.mentor_advice = generate_rag_response(
                manual_profile, st.session_state.experiences
            )
        st.rerun()

# ────────────────────────────────────────────────────────────────────────────
# Run full pipeline when text collected from chat or resume
# ────────────────────────────────────────────────────────────────────────────
if collected_text:
    with st.spinner("🧠 Extracting profile with Claude…"):
        st.session_state.profile = extract_profile(collected_text)

    with st.spinner("📊 Computing Readiness Score…"):
        st.session_state.score = predict(st.session_state.profile)

    tech_stack = st.session_state.profile.get("raw_tech_stack", [])

    with st.spinner("🔍 Running RAG + Cosine Matcher…"):
        st.session_state.experiences = smart_experience_matcher(tech_stack, top_k=3)

    with st.spinner("🧑‍🏫 Generating mentor advice…"):
        st.session_state.mentor_advice = generate_rag_response(
            st.session_state.profile,
            st.session_state.experiences,
        )

# ────────────────────────────────────────────────────────────────────────────
# Results Section
# ────────────────────────────────────────────────────────────────────────────
if st.session_state.score is not None:
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Your Placement Analysis</div>', unsafe_allow_html=True)

    col_score, col_radar = st.columns([1, 2])

    with col_score:
        score = st.session_state.score
        # Color based on score
        if score >= 75:
            color, label = "#48c0c0", "🟢 High Readiness"
        elif score >= 55:
            color, label = "#f7c96a", "🟡 Moderate"
        else:
            color, label = "#e07070", "🔴 Needs Work"

        st.markdown(f"""
        <div class="score-card">
          <div class="score-label">Placement Readiness Score</div>
          <div class="score-number" style="color:{color}">{score}</div>
          <div class="score-label" style="font-size:1rem;margin-top:0.5rem">{label}</div>
          <div class="score-label" style="margin-top:1rem">out of 100 · GBR Model (R²={metrics["r2"]})</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100], "tickfont": {"color": "#888"}},
                "bar": {"color": color},
                "bgcolor": "#1a1d2e",
                "bordercolor": "#2e3260",
                "steps": [
                    {"range": [0, 55], "color": "#1a0f0f"},
                    {"range": [55, 75], "color": "#1a1a0f"},
                    {"range": [75, 100], "color": "#0f1a1a"},
                ],
                "threshold": {
                    "line": {"color": "#7c6af7", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
            number={"font": {"color": color, "family": "Syne", "size": 36}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#888"},
            height=220,
            margin=dict(t=20, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_radar:
        profile = st.session_state.profile
        radar_fields = {
            "Academic": profile.get("Academic_Score", 5),
            "DSA": profile.get("DSA_Skill", 5),
            "Projects": profile.get("Project_Quality", 5),
            "Experience": profile.get("Experience_Score", 5),
            "Open Source": profile.get("OpenSource_Value", 1) / 10 * 10,  # normalize to 10
            "Soft Skills": profile.get("Soft_Skills", 5),
            "Tech Stack": profile.get("Tech_Stack_Score", 5),
        }
        categories = list(radar_fields.keys())
        values = list(radar_fields.values())
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(124,106,247,0.15)",
            line=dict(color="#7c6af7", width=2),
            name="Your Profile",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(color="#666"), gridcolor="#1e2040"),
                angularaxis=dict(tickfont=dict(color="#aaa"), gridcolor="#1e2040"),
                bgcolor="rgba(0,0,0,0)",
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=320,
            margin=dict(t=30, b=30, l=50, r=50),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Profile breakdown bar chart
    with st.expander("📈 Profile Breakdown"):
        df_profile = pd.DataFrame({
            "Attribute": list(radar_fields.keys()),
            "Score": list(radar_fields.values()),
            "Max": [10] * len(radar_fields),
        })
        fig_bar = px.bar(
            df_profile, x="Attribute", y="Score",
            color="Score",
            color_continuous_scale=["#e07070", "#f7c96a", "#48c0c0"],
            range_y=[0, 10],
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aaa"),
            coloraxis_showscale=False,
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Recommended Experiences ──────────────────────────────────────────────
    st.markdown('<div class="section-header">🎯 Top Matched Interview Experiences <span style="color:#f7c96a;font-size:0.8rem">Cosine Similarity · Bonus</span></div>', unsafe_allow_html=True)

    for exp in st.session_state.experiences:
        difficulty_class = "badge-hard" if exp["difficulty"] == "Hard" else "badge-medium"
        skills_html = " ".join(f'<span class="badge">{s}</span>' for s in exp["skills"].split(", "))
        st.markdown(f"""
        <div class="exp-card">
          <div class="exp-rank">#{exp['rank']} · Cosine Similarity: {exp['cosine_similarity']:.3f}</div>
          <div class="exp-company">{exp['company']} <span style="color:#7c6af7">·</span> {exp['role']}</div>
          <div class="exp-role">{exp['level']} &nbsp;|&nbsp; <span class="badge {difficulty_class}">{exp['difficulty']}</span></div>
          <div style="margin:0.4rem 0">{skills_html}</div>
          <div class="exp-text">{exp['experience']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Mentor Advice ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🧑‍🏫 Your Personalized Mentor Advice</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="score-card" style="font-size:0.92rem;line-height:1.7;color:#ddd">{st.session_state.mentor_advice.replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )

    # ── Extracted Profile JSON ───────────────────────────────────────────────
    with st.expander("🔍 Extracted Profile JSON"):
        st.json(st.session_state.profile)