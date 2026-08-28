"""
Skeptic Agent Module
Evaluates candidate claims for contradictions, exaggerations, red flags, and lack of depth.
"""
from typing import Optional
from google import genai
from pydantic import BaseModel
from .base_agent import BaseAgent, AgentEvaluation


class SkepticAgent(BaseAgent):
    """Evaluates contradictions, exaggerated metrics, red flags, and integrity issues."""

    def __init__(self, model: str = "gemini-3.6-flash"):
        instructions = (
            "PERSONA: Skeptical Auditor / Red Flag Investigator.\n"
            "YOUR TASK: Actively look for inconsistencies, exaggerations, unverified claims, or red flags.\n"
            "Compare what is written on the resume vs how the candidate answered during the interview.\n"
            "Check if metrics seem unrealistic, if they dodged questions, if they claimed credit for team work as solo work, "
            "or if their theoretical explanation broke down when probed.\n"
            "DETERMINE: Are there hidden risks, unproven claims, or red flags that other interviewers might miss?"
        )
        super().__init__(
            name="Victor Vance",
            role_title="Skeptic Agent",
            persona_instructions=instructions,
            model=model,
        )


def evaluate_skeptic(
    candidate_name: str,
    candidate_profile: dict | BaseModel,
    resume_text: str,
    transcript_text: str,
    job_description_text: str,
    client: Optional[genai.Client] = None,
    model: str = "gemini-3.6-flash"
) -> AgentEvaluation:
    """Convenience function for independent Skeptic evaluation."""
    agent = SkepticAgent(model=model)
    return agent.evaluate(
        candidate_name=candidate_name,
        candidate_profile=candidate_profile,
        resume_text=resume_text,
        transcript_text=transcript_text,
        job_description_text=job_description_text,
        client=client,
    )
