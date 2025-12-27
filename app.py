import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import os

# --- Page Config ---
st.set_page_config(page_title="Elite Research Syndicate", layout="wide", page_icon="🎓")

# --- CSS for "Bestest" UI ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button { background-color: #FF4B4B; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- Robust LLM Factory ---
def get_llm(provider, api_key, model_hint=None):
    """
    Universal LLM factory for both Workers and Managers.
    """
    if not api_key:
        return None
    try:
        if provider == "Google Gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro-latest",
                verbose=True,
                google_api_key=api_key
            )
        elif provider == "Groq":
            return ChatGroq(
                temperature=0,
                model_name=model_hint or "llama3-70b-8192",
                groq_api_key=api_key  # FIXED: Was 'api_key'
            )
        elif provider == "OpenRouter":
            return ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                openai_api_key=api_key,  # FIXED: Was 'api_key'
                model=model_hint or "openai/gpt-4o"
            )
    except Exception as e:
        st.error(f"LLM Connection Failed: {e}")
        return None

# --- Sidebar: Command Center ---
with st.sidebar:
    st.header("🧠 Syndicate Controls")
    
    # 1. Manager Configuration
    st.subheader("1. Manager LLM (Orchestrator)")
    st.info("The Manager needs a high-intelligence model (e.g., GPT-4, Gemini 1.5 Pro) to delegate effectively.")
    manager_provider = st.selectbox("Manager Provider", ["Google Gemini", "OpenRouter"], key="mgr_prov")
    manager_key = st.text_input(f"{manager_provider} Key", type="password", key="mgr_key")

    # 2. Worker Configuration
    st.subheader("2. Worker Crew LLM (Execution)")
    worker_provider = st.selectbox("Worker Provider", ["Groq", "Google Gemini", "OpenRouter"], key="wrk_prov")
    worker_key = st.text_input(f"{worker_provider} Key", type="password", key="wrk_key")
    
    worker_model = None
    if worker_provider == "Groq":
        worker_model = st.selectbox("Worker Model", ["llama3-70b-8192", "mixtral-8x7b-32768"])

# --- Main Interface ---
st.title("🏛️ Elite Research Syndicate")
st.markdown("**Status:** Waiting for mission parameters.")

topic = st.text_input("Mission Objective (Research Topic)", placeholder="e.g., The socioeconomic impact of fusion energy in 2050")

if st.button("🚀 Deploy Syndicate"):
    if not topic or not manager_key or not worker_key:
        st.error("❌ Mission Aborted: Missing API Keys or Topic.")
        st.stop()

    # Initialize LLMs
    manager_llm = get_llm(manager_provider, manager_key)
    worker_llm = get_llm(worker_provider, worker_key, worker_model)

    if manager_llm and worker_llm:
        with st.status("⚙️ Mobilizing Agents...", expanded=True) as status:
            # --- 1. The Agents ---
            lead_researcher = Agent(
                role='Principal Investigator',
                goal=f'Conduct a deep-dive forensic investigation into {topic}',
                backstory="""You are a world-renowned investigator with a Nobel-level ability to synthesize disparate information sources. You never settle for surface-level facts.""",
                verbose=True,
                allow_delegation=False,
                llm=worker_llm
            )

            data_analyst = Agent(
                role='Senior Data Statistician',
                goal='Rigorously analyze data points and verify statistical claims',
                backstory="""You are a cynical statistician who demands proof. You look for trends, outliers, and data integrity issues in the research provided.""",
                verbose=True,
                allow_delegation=False,
                llm=worker_llm
            )

            writer = Agent(
                role='Distinguished Academic Stylist',
                goal='Synthesize findings into a Nature-journal caliber paper',
                backstory="""You are a legendary science communicator. You write with absolute clarity, authority, and structural elegance.""",
                verbose=True,
                allow_delegation=False,
                llm=worker_llm
            )

            critic = Agent(
                role='Research Integrity Officer',
                goal='Mercilessly review the draft for bias, fallacies, and gaps',
                backstory="""You are the final gatekeeper. Nothing gets published unless it is factually bulletproof and ethically sound.""",
                verbose=True,
                allow_delegation=False,
                llm=worker_llm
            )

            # --- 2. The Tasks ---
            task_investigate = Task(
                description=f"Compile a comprehensive dossier on {topic}. Focus on recent breakthroughs (2024-2025) and raw data.",
                expected_output="A raw research dossier containing key findings, statistics, and expert quotes.",
                agent=lead_researcher
            )

            task_analyze = Task(
                description="Analyze the research dossier. Extract quantitative trends and validate the statistical methodology.",
                expected_output="A data validation report highlighting confirmed trends and potential fallacies.",
                agent=data_analyst,
                context=[task_investigate]
            )

            task_write = Task(
                description="Write the final academic paper. Integrate the research and the data analysis into a cohesive narrative.",
                expected_output="A polished, markdown-formatted academic paper with Abstract, Methods, and Results.",
                agent=writer,
                context=[task_investigate, task_analyze]
            )

            task_review = Task(
                description="Critique the paper. If necessary, request revisions. Ensure tone is objective and authoritative.",
                expected_output="The final signed-off manuscript ready for publication.",
                agent=critic,
                context=[task_write]
            )

            # --- 3. The Crew ---
            syndicate = Crew(
                agents=[lead_researcher, data_analyst, writer, critic],
                tasks=[task_investigate, task_analyze, task_write, task_review],
                process=Process.hierarchical,
                manager_llm=manager_llm,
                memory=True,
                planning=True,
                verbose=True
            )

            status.write("🧠 Syndicate Assembled. Planning Phase Initiated...")
            result = syndicate.kickoff()
            status.update(label="✅ Mission Complete", state="complete", expanded=False)

        # --- Output Display ---
        st.divider()
        st.subheader("📄 Final Publication")
        st.markdown(result)

        st.download_button(
            label="📥 Download Manuscript",
            data=str(result),
            file_name="elite_publication.md",
            mime="text/markdown"
        )
