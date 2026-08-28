"""
Profile Builder Module
Uses Gemini API (google-genai SDK) to extract a structured JSON profile from candidate resume and transcript.
"""
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class ProjectHighlight(BaseModel):
    name: str = Field(description="Name or title of the project")
    description: str = Field(description="Brief summary of the project and deliverables")
    technologies: List[str] = Field(default_factory=list, description="Technologies, tools, or frameworks used")


class CandidateProfile(BaseModel):
    candidate_name: str = Field(default="Candidate", description="Full name or identifier of candidate")
    skills: List[str] = Field(description="Key technical, analytical, and professional skills")
    years_experience: str = Field(description="Total years of relevant work experience")
    projects: List[ProjectHighlight] = Field(description="Key projects highlighted in resume or interview")
    claims: List[str] = Field(description="Major professional claims or achievements stated by candidate")
    summary: str = Field(description="Synthesized summary of candidate background relative to job description")


def build_candidate_profile(
    candidate_name: str,
    resume_text: str,
    transcript_text: str,
    job_description_text: str,
    client: Optional[genai.Client] = None,
    model: str = "gemini-3.6-flash"
) -> CandidateProfile:
    """Extracts a structured CandidateProfile from resume and transcript text via Gemini.

    Args:
        candidate_name: Identifier for the candidate (e.g., 'Candidate A')
        resume_text: Raw text of the resume
        transcript_text: Raw text of the interview transcript
        job_description_text: Target job description text
        client: Optional existing genai.Client instance
        model: Gemini model ID to use

    Returns:
        Structured CandidateProfile pydantic model instance.
    """
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is missing. "
                "Please set it in your .env file or environment."
            )
        client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert HR data analyst.
Extract a structured candidate profile for '{candidate_name}' based on their Resume and Interview Transcript, evaluated against the target Job Description.

--- JOB DESCRIPTION ---
{job_description_text}

--- RESUME TEXT ---
{resume_text}

--- INTERVIEW TRANSCRIPT ---
{transcript_text}

Extract all facts objectively. Identify candidate claims, technical skills, years of experience, notable projects, and a balanced executive summary.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CandidateProfile,
            temperature=0.2,
        ),
    )

    if hasattr(response, "parsed") and response.parsed is not None:
        return response.parsed

    # Fallback parsing if needed
    return CandidateProfile.model_validate_json(response.text)
