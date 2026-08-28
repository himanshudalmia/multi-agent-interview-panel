"""
Hiring Manager Agent Module
Evaluates candidate overall hire-worthiness, business impact, and role alignment.
"""
from typing import Optional
from google import genai
from pydantic import BaseModel
from .base_agent import BaseAgent, AgentEvaluation


class HiringManagerAgent(BaseAgent):
    """Evaluates business alignment, execution capability, and overall hiring ROI for the target role."""

    def __init__(self, model: str = "gemini-3.6-flash"):
        instructions = (
            "PERSONA: Hiring Manager / VP of Engineering.\n"
            "YOUR TASK: Evaluate the candidate's overall hire-worthiness specifically for the target role in the Job Description. "
            "Focus on execution ability, ownership, speed of onboarding, problem-solving mindset, and overall business value.\n"
            "DETERMINE: Should we extend an offer to this candidate for THIS specific role? Can they deliver on project outcomes?"
        )
        super().__init__(
            name="Marcus Brody",
            role_title="Hiring Manager Agent",
            persona_instructions=instructions,
            model=model,
        )


def evaluate_hiring_manager(
    candidate_name: str,
    candidate_profile: dict | BaseModel,
    resume_text: str,
    transcript_text: str,
    job_description_text: str,
    client: Optional[genai.Client] = None,
    model: str = "gemini-3.6-flash"
) -> AgentEvaluation:
    """Convenience function for independent Hiring Manager evaluation."""
    agent = HiringManagerAgent(model=model)
    return agent.evaluate(
        candidate_name=candidate_name,
        candidate_profile=candidate_profile,
        resume_text=resume_text,
        transcript_text=transcript_text,
        job_description_text=job_description_text,
        client=client,
    )
