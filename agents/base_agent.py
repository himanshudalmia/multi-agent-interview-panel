"""
Base Agent Module
Provides standard Pydantic response schema and base evaluation functionality for panel agents.
"""
import os
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class EvidenceQuote(BaseModel):
    quote: str = Field(description="Exact quote or verbatim snippet from resume or transcript supporting the point")
    source: Literal["resume", "transcript"] = Field(description="Source of the verbatim evidence: 'resume' or 'transcript'")


class AgentEvaluation(BaseModel):
    agent_name: str = Field(description="Name or role title of the persona")
    opinion: str = Field(description="Comprehensive evaluation opinion explaining reasoning, strengths, and concerns")
    score: int = Field(description="Evaluation score integer from 1 (unqualified/severe red flags) to 10 (outstanding fit)")
    evidence_quotes: List[EvidenceQuote] = Field(description="List of direct evidence quotes supporting opinion")
    confidence: Literal["low", "medium", "high"] = Field(description="Confidence level based on available evidence")
    insufficient_info_flags: List[str] = Field(
        default_factory=list,
        description="Explicit flags for unverified claims or missing/unclear information in the interview/resume"
    )


class BaseAgent:
    """Base agent wrapper for running independent persona evaluations."""

    def __init__(self, name: str, role_title: str, persona_instructions: str, model: str = "gemini-3.6-flash"):
        self.name = name
        self.role_title = role_title
        self.persona_instructions = persona_instructions
        self.model = model

    def evaluate(
        self,
        candidate_name: str,
        candidate_profile: dict | BaseModel,
        resume_text: str,
        transcript_text: str,
        job_description_text: str,
        client: Optional[genai.Client] = None
    ) -> AgentEvaluation:
        """Runs an independent evaluation of candidate without any visibility into other agents' outputs.

        Args:
            candidate_name: Name/ID of candidate (e.g. Candidate A)
            candidate_profile: Profile dictionary or Pydantic model
            resume_text: Raw resume text
            transcript_text: Raw interview transcript text
            job_description_text: Job description text
            client: Optional genai.Client instance

        Returns:
            Structured AgentEvaluation object.
        """
        if client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is missing.")
            client = genai.Client(api_key=api_key)

        profile_summary = (
            candidate_profile.model_dump_json(indent=2)
            if isinstance(candidate_profile, BaseModel)
            else str(candidate_profile)
        )

        prompt = f"""
You are acting as the {self.role_title} ({self.name}) on an interview panel.

{self.persona_instructions}

CRITICAL RULES:
1. Base your judgment strictly on the candidate's Resume, Transcript, Profile, and Job Description below.
2. You MUST cite exact verbatim quotes for every claim you make using evidence_quotes. Specify source as either "resume" or "transcript".
3. Assign a score from 1 to 10.
4. If candidate info is insufficient or missing for a key area in your domain, flag it explicitly in insufficient_info_flags and adjust confidence accordingly.

--- CANDIDATE ID ---
{candidate_name}

--- JOB DESCRIPTION ---
{job_description_text}

--- CANDIDATE PROFILE ---
{profile_summary}

--- RESUME TEXT ---
{resume_text}

--- INTERVIEW TRANSCRIPT ---
{transcript_text}

Provide your independent evaluation JSON strictly following the required schema.
"""

        from utils.retry_helper import generate_content_with_retry
        response = generate_content_with_retry(
            client=client,
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AgentEvaluation,
                temperature=0.3,
            ),
        )

        if hasattr(response, "parsed") and response.parsed is not None:
            result = response.parsed
            result.agent_name = self.role_title
            return result

        result = AgentEvaluation.model_validate_json(response.text)
        result.agent_name = self.role_title
        return result
