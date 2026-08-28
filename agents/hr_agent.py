"""
HR Agent Module
Evaluates candidate communication, teamwork, transparency, and cultural fit.
"""
from typing import Optional
from google import genai
from pydantic import BaseModel
from .base_agent import BaseAgent, AgentEvaluation


class HRAgent(BaseAgent):
    """Evaluates communication skills, teamwork, adaptability, and cultural alignment."""

    def __init__(self, model: str = "gemini-3.6-flash"):
        instructions = (
            "PERSONA: HR & People Operations Lead.\n"
            "YOUR TASK: Evaluate the candidate's communication clarity, emotional intelligence, teamwork attitude, "
            "collaboration style, responsiveness to questions, and honesty/transparency during the interview.\n"
            "DETERMINE: Will this candidate be a constructive team member? Do they communicate effectively under probing?"
        )
        super().__init__(
            name="Sarah Jenkins",
            role_title="HR Culture Agent",
            persona_instructions=instructions,
            model=model,
        )


def evaluate_hr(
    candidate_name: str,
    candidate_profile: dict | BaseModel,
    resume_text: str,
    transcript_text: str,
    job_description_text: str,
    client: Optional[genai.Client] = None,
    model: str = "gemini-3.6-flash"
) -> AgentEvaluation:
    """Convenience function for independent HR evaluation."""
    agent = HRAgent(model=model)
    return agent.evaluate(
        candidate_name=candidate_name,
        candidate_profile=candidate_profile,
        resume_text=resume_text,
        transcript_text=transcript_text,
        job_description_text=job_description_text,
        client=client,
    )
