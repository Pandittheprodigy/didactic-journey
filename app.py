import streamlit as st
import os
from textwrap import dedent

# WRAPPER: Try/Except to handle import errors gracefully
try:
    from crewai import Agent, Task, Crew, Process, LLM
except ImportError as e:
    st.error(f"⚠️ Library Error: {e}")
    st.warning("Run: pip install --upgrade crewai crewai-tools")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="Harvard Research Crew", layout="wide")

# --- Sidebar: Credentials ---
with st.sidebar:
    st.header("🔐 Auth & Config")
    
    # Secure Key Inputs
    keys = {}
    for provider in ["GEMINI", "GROQ", "OPENROUTER"]:
        key = st.text_input(f"{provider} API Key", type="password")
        if key: keys[provider] = key

    st.divider()
    st.markdown("**Architecture:**")
    st.code("""Researcher -> Groq (Llama3)
Writer     -> Gemini (1.5 Pro)
Editor     -> OpenRouter (Claude)""", language="text")

# --- Core Application ---
st.title("🎓 Harvard Research Crew")
topic = st.text_input("Research Topic", "The Impact of Quantum Computing on Cybersecurity")

def create_crew(topic, api_keys):
    """Constructs the CrewAI assembly with specific LLM bindings."""
    
    # 1. Define LLMs (The Modern 'LiteLLM' Pattern)
    # Note: We pass specific environment variables or api_keys directly
    
    llm_groq = LLM(
        model="groq/llama3-70b-8192",
        api_key=api_keys["GROQ"],
        temperature=0.5
    )

    llm_gemini = LLM(
        model="gemini/gemini-1.5-pro",
        api_key=api_keys["GEMINI"],
        temperature=0.7
    )

    llm_editor = LLM(
        model="openrouter/anthropic/claude-3.5-sonnet",
        api_key=api_keys["OPENROUTER"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2
    )

    # 2. Agents
    researcher = Agent(
        role="Senior Researcher",
        goal=f"Uncover empirical data and primary sources about: {topic}",
        backstory="You are a dogged academic researcher who trusts only peer-reviewed data.",
        llm=llm_groq,
        verbose=True
    )

    writer = Agent(
        role="Lead Author",
        goal="Synthesize research into a coherent, argument-driven academic paper.",
        backstory="You are a professor known for clear, dense, and authoritative writing.",
        llm=llm_gemini,
        verbose=True
    )

    editor = Agent(
        role="Harvard Style Editor",
        goal="Enforce strict Harvard referencing and academic tone.",
        backstory="You are a ruthless editor. You ensure every claim has a (Author, Year) citation.",
        llm=llm_editor,
        verbose=True
    )

    # 3. Tasks
    task_research = Task(
        description=f"Find 5 key papers/stats on '{topic}'. Focus on 2023-2025 data.",
        expected_output="Bulleted list of findings with authors and dates.",
        agent=researcher
    )

    task_write = Task(
        description="Write a 5-section paper (Abstract, Intro, Methods, Results, Discussion).",
        expected_output="Markdown draft with inline citations.",
        agent=writer
    )

    task_edit = Task(
        description="Refine the draft. Add a 'References' list at the end in strict Harvard style.",
        expected_output="Final Publication-Ready Markdown.",
        agent=editor
    )

    return Crew(
        agents=[researcher, writer, editor],
        tasks=[task_research, task_write, task_edit],
        process=Process.sequential
    )

# --- Execution ---
if st.button("🚀 Start Research Process"):
    if len(keys) < 3:
        st.error("❌ Missing API Keys. Please check the sidebar.")
    else:
        with st.spinner("🤖 Agents are working... (View terminal for live logs)"):
            try:
                crew = create_crew(topic, keys)
                result = crew.kickoff()
                
                st.success("Process Complete!")
                st.markdown("### 📄 Final Paper")
                st.markdown(result)
                
                st.download_button(
                    label="Download .md",
                    data=str(result),
                    file_name="research_paper.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"An error occurred during execution: {e}")
