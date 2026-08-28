"""
Judge Module
Takes initial agent evaluations, evidence quotes, and full debate transcript to make a step-by-step reasoning decision.
Crucially: Does NOT average scores, but weighs evidence strength, transcript proof, and red flags.
"""
import os
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from agents.base_agent import AgentEvaluation
from debate import DebateResult


class JudgeDecision(BaseModel):
    candidate_name: str = Field(description="Name/ID of candidate")
    final_recommendation: Literal["Strong Hire", "Hire", "Weak Hire", "No Hire"] = Field(
        description="Final hiring recommendation"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Overall confidence level in the judicial verdict"
    )
    key_evidence: List[str] = Field(
        description="Crucial verbatim quotes and facts that decided the verdict"
    )
    reasoning: str = Field(
        description="Detailed step-by-step judicial reasoning weighing evidence strength, transcript facts, and agent debate points (NOT simple score averaging)"
    )
    unresolved_disagreements: List[str] = Field(
        default_factory=list,
        description="Any remaining disputes between panel agents that were not reconciled in debate"
    )


def adjudicate(
    candidate_name: str,
    candidate_profile: dict | BaseModel,
    initial_evaluations: Dict[str, AgentEvaluation],
    debate_result: DebateResult,
    resume_text: str,
    transcript_text: str,
    job_description_text: str,
    client: Optional[genai.Client] = None,
    model: str = "gemini-3.6-flash",
) -> JudgeDecision:
    """Makes a final step-by-step judicial hiring decision based on evidence and debate transcript.

    Args:
        candidate_name: Name of candidate
        candidate_profile: Candidate profile
        initial_evaluations: Dictionary of initial agent evaluations
        debate_result: DebateResult object containing debate logs and final agent scores
        resume_text: Resume text
        transcript_text: Interview transcript text
        job_description_text: Job description text
        client: Optional genai.Client
        model: Gemini model ID

    Returns:
        JudgeDecision object.
    """
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        client = genai.Client(api_key=api_key)

    # Format initial agent opinions & quotes
    agent_opinions_formatted = []
    all_evidence_quotes = []
    for role, eval_obj in initial_evaluations.items():
        agent_opinions_formatted.append(
            f"=== {role} ===\n"
            f"Initial Score: {eval_obj.score}/10 | Confidence: {eval_obj.confidence}\n"
            f"Opinion: {eval_obj.opinion}\n"
            f"Insufficient Info Flags: {eval_obj.insufficient_info_flags}\n"
            f"Evidence Quotes: {[q.model_dump() for q in eval_obj.evidence_quotes]}\n"
        )
        for q in eval_obj.evidence_quotes:
            all_evidence_quotes.append(f"[{q.source.upper()}] \"{q.quote}\" (cited by {role})")

    opinions_text = "\n".join(agent_opinions_formatted)
    evidence_text = "\n".join(all_evidence_quotes)

    # Format debate turns
    debate_turns_text = "\n".join(
        f"Turn {t.turn_number} [{t.agent} addressing {t.target_agent} on '{t.topic}']:\n"
        f"  Message: {t.message}\n"
        f"  Opinion Changed: {t.changed_opinion} (Score: {t.previous_score} -> {t.new_score}, Confidence: {t.revised_confidence})"
        for t in debate_result.transcript_logs
    )

    profile_str = (
        candidate_profile.model_dump_json(indent=2)
        if isinstance(candidate_profile, BaseModel)
        else str(candidate_profile)
    )

    prompt = f"""
You are the Chief Judicial Arbitrator of an executive AI interview panel.

CRITICAL INSTRUCTIONS:
1. You MUST NOT simply average the 4 agents' numerical scores to reach your recommendation.
2. Perform step-by-step evidence weighing. Evaluate which evidence quotes are strongest, whether skeptic red flags were verified or debunked in debate, and how well candidate qualifications match the Job Description.
3. Determine final recommendation strictly as one of: "Strong Hire", "Hire", "Weak Hire", or "No Hire".
4. List key evidence quotes and identify any remaining unresolved disagreements between agents.

--- CANDIDATE ID ---
{candidate_name}

--- JOB DESCRIPTION ---
{job_description_text}

--- CANDIDATE PROFILE ---
{profile_str}

--- ALL AGENTS' INITIAL INDEPENDENT EVALUATIONS & QUOTES ---
{opinions_text}

--- ALL CITED EVIDENCE QUOTES ---
{evidence_text}

--- FULL DEBATE TRANSCRIPT & OPINION SHIFTS ---
{debate_turns_text}

--- RESUME TEXT ---
{resume_text}

--- INTERVIEW TRANSCRIPT TEXT ---
{transcript_text}

Output your step-by-step judicial verdict strictly following the required JSON schema.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JudgeDecision,
            temperature=0.2,
        ),
    )

    if hasattr(response, "parsed") and response.parsed is not None:
        result = response.parsed
        result.candidate_name = candidate_name
        return result

    result = JudgeDecision.model_validate_json(response.text)
    result.candidate_name = candidate_name
    return result
