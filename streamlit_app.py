"""
Streamlit Web Application for Multi-Agent AI Interview Panel Simulator
Wraps process_candidate logic to provide an interactive dashboard UI.
"""
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from google import genai

from utils.pdf_reader import extract_text_from_pdf
from main import process_candidate

# Page configuration
st.set_page_config(
    page_title="Multi-Agent AI Interview Panel Simulator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">🤖 Multi-Agent AI Interview Panel Simulator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Simulates an executive hiring panel with 4 independent AI personas, turn-based cross-examination debate, and evidence-weighing judicial adjudication.</div>',
    unsafe_allow_html=True,
)

# Load environment
load_dotenv()
env_api_key = os.environ.get("GEMINI_API_KEY", "")

# Sidebar Controls
st.sidebar.title("⚙️ Simulation Settings")
api_key_input = st.sidebar.text_input("Gemini API Key", value=env_api_key, type="password")

candidate_selection = st.sidebar.radio(
    "Select Candidate(s) to Evaluate:",
    options=["Candidate A", "Candidate B", "Both Candidates"],
    index=0,
)

model_choice = st.sidebar.selectbox(
    "Gemini Model:",
    options=["gemini-3.6-flash"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Pipeline Workflow:**\n"
    "1. **Profile Builder** (Facts & Claims)\n"
    "2. **4 Independent Agents** (Technical, HR, Manager, Skeptic)\n"
    "3. **Turn-Based Debate** (Score shifts logged)\n"
    "4. **Judicial Arbitrator** (Step-by-step evidence weighing)\n"
    "5. **Markdown Report**"
)

# Paths setup
base_dir = Path(__file__).parent.resolve()
data_dir = base_dir / "data"
output_dir = base_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# Candidate Data Config
candidate_configs = {
    "Candidate A": {
        "id": "Candidate A",
        "resume": data_dir / "03_Resume_A.pdf",
        "transcript": data_dir / "05_Transcript_A.pdf",
    },
    "Candidate B": {
        "id": "Candidate B",
        "resume": data_dir / "04_Resume_B.pdf",
        "transcript": data_dir / "06_Transcript_B.pdf",
    },
}

# Run Button
run_button = st.button("🚀 Run Evaluation")

if run_button:
    api_key = api_key_input.strip()
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY is required. Please enter your API key in the sidebar or set it in `.env`.")
        st.stop()

    client = genai.Client(api_key=api_key)
    jd_path = data_dir / "02_Job_Description.pdf"

    if not jd_path.exists():
        st.error(f"Job Description file missing at {jd_path}")
        st.stop()

    with st.spinner("Extracting Job Description..."):
        job_description_text = extract_text_from_pdf(jd_path)

    targets = []
    if candidate_selection == "Both Candidates":
        targets = ["Candidate A", "Candidate B"]
    else:
        targets = [candidate_selection]

    reports_data = {}

    for cand_name in targets:
        cfg = candidate_configs[cand_name]

        if not cfg["resume"].exists() or not cfg["transcript"].exists():
            st.warning(f"Files missing for {cand_name}. Skipping.")
            continue

        status_text = st.empty()
        status_text.info(f"⏳ Processing **{cand_name}** through 5-stage multi-agent pipeline...")

        with st.spinner(f"Evaluating {cand_name} (Profile → 4 Agents → Debate → Judge)..."):
            res = process_candidate(
                candidate_id=cfg["id"],
                resume_path=cfg["resume"],
                transcript_path=cfg["transcript"],
                job_description_text=job_description_text,
                output_dir=output_dir,
                client=client,
                model=model_choice,
            )

            # Read generated report
            report_path = res["report_path"]
            if Path(report_path).exists():
                report_content = Path(report_path).read_text(encoding="utf-8")
                reports_data[cand_name] = report_content

        status_text.success(f"✅ Evaluation complete for **{cand_name}**!")

    # Display Reports
    if reports_data:
        st.markdown("---")
        st.subheader("📊 Evaluation Reports")

        if len(reports_data) == 1:
            cand_key = list(reports_data.keys())[0]
            st.markdown(reports_data[cand_key])
        else:
            tabs = st.tabs(list(reports_data.keys()))
            for idx, (cand_key, content) in enumerate(reports_data.items()):
                with tabs[idx]:
                    st.markdown(content)
