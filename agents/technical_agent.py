"""
Technical Agent Module
Evaluates candidate technical skill depth, system design, and hands-on domain knowledge.
"""
from typing import Optional
from google import genai
from pydantic import BaseModel
from .base_agent import BaseAgent, AgentEvaluation


class TechnicalAgent(BaseAgent):
    """Evaluates technical depth, architecture understanding, and coding/engineering competence."""

    def __init__(self, model: str = "gemini-3.6-flash"):
        instructions = (
            "PERSONA: Technical Lead / Senior Architect.\n"
            "YOUR TASK: Evaluate the candidate's technical skills, engineering depth, architectural reasoning, "
            "and hands-on problem-solving. Pay attention to how clearly they explain complex technical decisions, "
            "frameworks, algorithms, and agentic/system design.\n"
            "DETERMINE: Are their technical skills sufficient for the target role? Did they display genuine deep understanding "
            "or surface-level buzzwords?"
        )
        super().__init__(
            name="Dr. Alex Vance",
            role_title="Technical Lead Agent",
            persona_instructions=instructions,
            model=model,
        )


def evaluate_technical(
    candidate_name: str,
    candidate_profile: dict | BaseModel,
    resume_text: str,
    transcript_text: str,
    job_description_text: str,
    client: Optional[genai.Client] = None,
    model: str = "gemini-3.6-flash"
) -> AgentEvaluation:
    """Convenience function for independent technical evaluation."""
    agent = TechnicalAgent(model=model)
    return agent.evaluate(
        candidate_name=candidate_name,
        candidate_profile=candidate_profile,
        resume_text=resume_text,
        transcript_text=transcript_text,
        job_description_text=job_description_text,
        client=client,
    )
