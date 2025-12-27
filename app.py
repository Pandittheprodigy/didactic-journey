import streamlit as st
import os

# Removed OpenAI environment variable setting to eliminate any OpenAI dependency.

from crewai import Agent, Task, Crew, Process, LLM

# --- Page Config ---
st.set_page_config(page_title="Elite Research Syndicate", layout="wide", page_icon="🎓")

# --- CSS for UI ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .stButton>button { background-color: #FF4B4B; color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- Robust LLM Factory ---
def get_llm(provider, api_key, model_hint=None):
    """
    Universal LLM factory using CrewAI's native LLM class for consistency.
    All models are non-OpenAI to fully remove OpenAI dependencies.
    """
    if not api_key:
        return None
    
    try:
        if provider == "Google Gemini":
            return LLM(
                model="gemini/gemini-1.5-pro-latest",
                api_key=api_key
            )
        elif provider == "Groq":
            return LLM(
                model=f"groq/{model_hint or 'llama3-70b-8192'}",
                api_key=api_key
            )
        elif provider == "OpenRouter":
            # Changed default to a non-OpenAI model (Anthropic Claude) to remove OpenAI completely.
            return LLM(
                model=f"openrouter/{model_hint or 'anthropic/claude-3-haiku'}",
                api_key=api_key
            )
    except Exception as e:
        st.error(f"LLM Connection Failed: {e}")
        return None

# --- Sidebar: Command Center ---
with st.sidebar:
    st.header("🧠 Syndicate Controls")
    
    # 1. Manager Configuration
    st.subheader("1. Manager LLM (Orchestrator)")
    manager_provider = st.selectbox("Manager Provider", ["Google Gemini", "OpenRouter"], key="mgr_prov")
    manager_key = st.text_input(f"{manager_provider} Key", type="password", key="mgr_key")

    # 2. Worker Configuration
    st.subheader("2. Worker Crew LLM (Execution)")
    worker_provider = st.selectbox("Worker Provider", ["Groq", "Google Gemini", "OpenRouter"], key="wrk_prov")
    worker_key = st.text_input(f"{worker_provider} Key", type="password", key="wrk_key")
    
    worker_model = None
    if worker_provider == "Groq":
        worker_model = st.selectbox("Worker Model", ["llama3-70b-8192", "mixtral-8x7b-32768"])
    elif worker_provider == "OpenRouter":
        # Updated default to non-OpenAI model.
        worker_model = st.text_input("OpenRouter Model ID", value="anthropic/claude-3-haiku")

# --- Main Interface ---
st.title("🏛️ Elite Research Syndicate")
st.markdown("**Status:** Waiting for mission parameters.")

topic = st.text_input("Mission Objective", placeholder="e.g., The socioeconomic impact of fusion energy in 2050")

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
                backstory="You are a world-renowned investigator with a Nobel-level ability to synthesize disparate information sources.",
                verbose=True,
                allow_delegation=False,
                llm=worker_llm
            )

            data_analyst = Agent(
                role='Senior Data Statistician',
                goal='Rigorously analyze data points and verify statistical claims',
                backstory="You are a cynical statistician who demands proof. You look for trends, outliers, and data integrity issues.",
                verbose=True,
                allow_delegation=False,
                llm=worker_llm
            )

            writer = Agent(
                role='Distinguished Academic Stylist',
                goal='Synthesize findings into a Nature-journal caliber paper',
                backstory="You are a legendary science communicator. You write with absolute clarity and authority.",
                verbose=True,
                allow_delegation=False,
                llm=worker_llm
            )

            critic = Agent(
                role='Research Integrity Officer',
                goal='Mercilessly review the draft for bias, fallacies, and gaps',
                backstory="You are the final gatekeeper. Nothing gets published unless it is factually bulletproof.",
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
                memory=False,  # DISABLED: Memory requires embeddings, which often default to OpenAI. Keep false for pure non-OpenAI setup.
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
