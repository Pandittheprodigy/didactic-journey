# Harvard Research Paper Publication Crew
# This code implements a CrewAI-based system for generating, reviewing, and publishing research papers.
# It uses a crew of expert agents, each powered by different LLMs (Gemini, OpenRouter, Groq) via their APIs.
# The interface is built with Streamlit for user interaction.
# Note: Requires API keys for Gemini, OpenRouter, and Groq. Set them as environment variables.
# No OpenAI dependencies are used; OpenRouter is accessed directly via its API.

import os
import requests
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai.llms import BaseLLM  # For custom LLM
from google.generativeai import GenerativeModel  # For Gemini API
import groq  # For Groq API

# Set API keys from environment variables (secure practice)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize API clients
gemini_model = GenerativeModel("gemini-1.5-flash", api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Custom LLM for OpenRouter (no OpenAI)
class OpenRouterLLM(BaseLLM):
    model_name: str = "openai/gpt-4"  # Default model; can be changed
    api_key: str = OPENROUTER_API_KEY

    def _call(self, prompt: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000  # Adjust as needed
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenRouter API error: {response.text}")

openrouter_llm = OpenRouterLLM() if OPENROUTER_API_KEY else None

# Custom tool for paper generation (example; can be expanded)
class ResearchTool(BaseTool):
    name: str = "Research Tool"
    description: str = "Conducts research on a given query using an LLM."

    def _run(self, query: str) -> str:
        """Simulates research by querying an LLM. In production, integrate with real databases."""
        if gemini_model:
            response = gemini_model.generate_content(f"Research on: {query}")
            return response.text
        return "Gemini API not available."

research_tool = ResearchTool()

# Define Expert Agents
researcher = Agent(
    role="Researcher",
    goal="Conduct thorough research on given topics.",
    backstory="An expert researcher with access to vast knowledge bases.",
    tools=[research_tool],
    llm=gemini_model if gemini_model else None,  # Uses Gemini
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Draft high-quality research papers based on research.",
    backstory="A skilled academic writer specializing in clear, impactful prose.",
    llm=openrouter_llm,  # Uses OpenRouter
    verbose=True
)

reviewer = Agent(
    role="Reviewer",
    goal="Review and refine papers for accuracy, coherence, and Harvard standards.",
    backstory="A peer reviewer with expertise in academic publishing.",
    llm=groq_client if groq_client else None,  # Uses Groq
    verbose=True
)

publisher = Agent(
    role="Publisher",
    goal="Prepare and simulate publication of the paper.",
    backstory="An expert in academic publishing workflows.",
    llm=gemini_model if gemini_model else None,  # Can reuse Gemini
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
