# Harvard Research Paper Publication Crew
# This code implements a CrewAI-based system for generating, reviewing, and publishing research papers.
# It uses a crew of expert agents, each powered by different LLMs (Gemini, OpenRouter, Groq) via their APIs.
# The interface is built with Streamlit for user interaction.
# Note: Requires API keys for Gemini, OpenRouter, and Groq. Set them as environment variables.

import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import tool  # Assuming crewai_tools for custom tools; adjust if needed
from google.generativeai import GenerativeModel  # For Gemini API
import openai  # For OpenRouter (as it uses OpenAI-compatible interface)
import groq  # For Groq API

# Set API keys from environment variables (secure practice)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize API clients
gemini_model = GenerativeModel("gemini-1.5-flash", api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
openai.api_key = OPENROUTER_API_KEY  # OpenRouter uses OpenAI client
groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Custom tool for paper generation (example; can be expanded)
@tool
def research_tool(query: str) -> str:
    """Simulates research by querying an LLM. In production, integrate with real databases."""
    if gemini_model:
        response = gemini_model.generate_content(f"Research on: {query}")
        return response.text
    return "Gemini API not available."

# Define Expert Agents
researcher = Agent(
    role="Researcher",
    goal="Conduct thorough research on given topics.",
    backstory="An expert researcher with access to vast knowledge bases.",
    tools=[research_tool],
    llm="gemini-1.5-flash" if gemini_model else None,  # Uses Gemini
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Draft high-quality research papers based on research.",
    backstory="A skilled academic writer specializing in clear, impactful prose.",
    llm="openai/gpt-4" if OPENROUTER_API_KEY else None,  # Uses OpenRouter (via OpenAI client)
    verbose=True
)

reviewer = Agent(
    role="Reviewer",
    goal="Review and refine papers for accuracy, coherence, and Harvard standards.",
    backstory="A peer reviewer with expertise in academic publishing.",
    llm="groq/llama3-70b-8192" if groq_client else None,  # Uses Groq
    verbose=True
)

publisher = Agent(
    role="Publisher",
    goal="Prepare and simulate publication of the paper.",
    backstory="An expert in academic publishing workflows.",
    llm="gemini-1.5-flash" if gemini_model else None,  # Can reuse Gemini or cycle
    verbose=True
)

# Define Tasks
research_task = Task(
    description="Research the topic: {topic}. Provide key findings and sources.",
    expected_output="A summary of research findings with references.",
    agent=researcher
)

write_task = Task(
    description="Using the research, draft a full research paper on {topic}.",
    expected_output="A complete research paper in academic format.",
    agent=writer,
    context=[research_task]
)

review_task = Task(
    description="Review the drafted paper for errors, coherence, and adherence to Harvard style.",
    expected_output="A reviewed and revised version of the paper.",
    agent=reviewer,
    context=[write_task]
)

publish_task = Task(
    description="Prepare the final paper for publication, including formatting and metadata.",
    expected_output="A publication-ready paper with submission details.",
    agent=publisher,
    context=[review_task]
)

# Create the Crew
crew = Crew(
    agents=[researcher, writer, reviewer, publisher],
    tasks=[research_task, write_task, review_task, publish_task],
    process=Process.sequential,  # Tasks executed in sequence
    verbose=True
)

# Streamlit App
def main():
    st.title("Harvard Research Paper Publication Crew")
    st.write("Generate, review, and publish research papers using AI experts.")
    
    topic = st.text_input("Enter the research topic:", "The Impact of AI on Academic Publishing")
    
    if st.button("Generate Paper"):
        if not all([GEMINI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY]):
            st.error("Please set all API keys as environment variables.")
            return
        
        with st.spinner("Processing..."):
            try:
                result = crew.kickoff(inputs={"topic": topic})
                st.success("Paper generated successfully!")
                st.text_area("Final Output:", value=result, height=400)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
