"""
Debate Module
Orchestrates a turn-based cross-examination debate between panel agents.
Logs every turn with {agent, message, changed_opinion, previous_score, new_score} so opinion changes are fully traceable.
"""
import os
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from agents.base_agent import AgentEvaluation


class DebateTurnLog(BaseModel):
    turn_number: int = Field(description="Sequential turn index in debate")
    agent: str = Field(description="Role of responding agent (e.g., 'Technical Lead Agent')")
    target_agent: str = Field(description="Role of target agent being addressed (e.g., 'Skeptic Agent')")
    topic: str = Field(description="Specific topic or claim under debate")
    message: str = Field(description="Response message containing rebuttal, agreement, or defense with evidence")
    changed_opinion: bool = Field(description="Whether the responding agent changed their score or opinion this turn")
    previous_score: int = Field(description="Agent score before this turn")
    new_score: int = Field(description="Agent score after this turn")
    revised_confidence: Literal["low", "medium", "high"] = Field(description="Updated confidence level")


class DebateResult(BaseModel):
    candidate_name: str = Field(description="Name of candidate evaluated")
    transcript_logs: List[DebateTurnLog] = Field(description="List of turn logs detailing the full debate")
    final_agent_scores: Dict[str, int] = Field(description="Mapping of agent role title to final score after debate")
    opinion_shifts: List[str] = Field(description="Human-readable summary of score shifts and reasons")


def run_debate(
    candidate_name: str,
    candidate_profile: dict | BaseModel,
    initial_evaluations: Dict[str, AgentEvaluation],
    resume_text: str,
    transcript_text: str,
    job_description_text: str,
    client: Optional[genai.Client] = None,
    model: str = "gemini-3.6-flash",
) -> DebateResult:
    """Runs a multi-turn debate between the 4 panel agents.

    Args:
        candidate_name: Name/ID of candidate
        candidate_profile: Structured candidate profile
        initial_evaluations: Dict mapping role titles to AgentEvaluation objects
        resume_text: Resume text
        transcript_text: Interview transcript text
        job_description_text: Job description text
        client: Optional genai.Client
        model: Gemini model ID

    Returns:
        DebateResult containing debate transcript logs and updated score state.
    """
    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        client = genai.Client(api_key=api_key)

    # Current state of agent scores
    current_scores: Dict[str, int] = {
        role: eval_obj.score for role, eval_obj in initial_evaluations.items()
    }
    current_confidences: Dict[str, str] = {
        role: eval_obj.confidence for role, eval_obj in initial_evaluations.items()
    }

    # Define key debate pairings / sequence
    debate_sequence = [
        {
            "speaker": "Skeptic Agent",
            "target": "Technical Lead Agent",
            "topic": "Technical depth vs potential resume exaggeration & unverified claims",
        },
        {
            "speaker": "Technical Lead Agent",
            "target": "Skeptic Agent",
            "topic": "Rebuttal on technical evidence, complexity of projects, and transcript answers",
        },
        {
            "speaker": "Hiring Manager Agent",
            "target": "Skeptic Agent",
            "topic": "Weighing red flags against role deliverables and overall business ROI",
        },
        {
            "speaker": "HR Culture Agent",
            "target": "Skeptic Agent",
            "topic": "Communication clarity, candidate transparency under probing, and teamwork potential",
        },
    ]

    turn_logs: List[DebateTurnLog] = []
    opinion_shifts: List[str] = []

    profile_str = (
        candidate_profile.model_dump_json(indent=2)
        if isinstance(candidate_profile, BaseModel)
        else str(candidate_profile)
    )

    for turn_idx, plan in enumerate(debate_sequence, start=1):
        speaker_role = plan["speaker"]
        target_role = plan["target"]
        topic = plan["topic"]

        speaker_initial = initial_evaluations[speaker_role]
        target_initial = initial_evaluations[target_role]

        prev_score = current_scores[speaker_role]

        # Construct context for turn
        previous_turns_summary = ""
        if turn_logs:
            previous_turns_summary = "\n".join(
                f"Turn {t.turn_number} [{t.agent} -> {t.target_agent}]: {t.message} (Score: {t.previous_score} -> {t.new_score})"
                for t in turn_logs
            )
        else:
            previous_turns_summary = "No previous debate turns yet."

        prompt = f"""
You are simulating a turn-based panel debate for hiring '{candidate_name}'.
You are now speaking strictly as the **{speaker_role}**.

TARGET TO ADDRESS: **{target_role}**
TOPIC: {topic}

--- SPEAKER'S INITIAL OPINION ({speaker_role}) ---
Score: {speaker_initial.score}/10 | Confidence: {speaker_initial.confidence}
Opinion: {speaker_initial.opinion}
Evidence Quotes: {[q.model_dump() for q in speaker_initial.evidence_quotes]}

--- TARGET AGENT'S INITIAL OPINION ({target_role}) ---
Score: {target_initial.score}/10 | Confidence: {target_initial.confidence}
Opinion: {target_initial.opinion}
Evidence Quotes: {[q.model_dump() for q in target_initial.evidence_quotes]}

--- PREVIOUS DEBATE TURNS SO FAR ---
{previous_turns_summary}

--- CANDIDATE PROFILE & EVIDENCE ---
Profile: {profile_str}
Resume Text: {resume_text}
Transcript Text: {transcript_text}

TASK FOR {speaker_role}:
1. Address the target agent ({target_role}) directly on the topic.
2. Evaluate their points and compare with transcript/resume evidence.
3. Decide if you maintain your score ({prev_score}) or change it based on the evidence discussed.
4. Output your response structured according to schema. Include whether changed_opinion is True/False, your previous score ({prev_score}), and your new score.
"""

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DebateTurnLog,
                temperature=0.3,
            ),
        )

        turn_data: DebateTurnLog
        if hasattr(response, "parsed") and response.parsed is not None:
            turn_data = response.parsed
        else:
            turn_data = DebateTurnLog.model_validate_json(response.text)

        # Force metadata sanity
        turn_data.turn_number = turn_idx
        turn_data.agent = speaker_role
        turn_data.target_agent = target_role
        turn_data.previous_score = prev_score

        # Update current score state
        current_scores[speaker_role] = turn_data.new_score
        current_confidences[speaker_role] = turn_data.revised_confidence

        turn_logs.append(turn_data)

        if turn_data.changed_opinion and turn_data.previous_score != turn_data.new_score:
            opinion_shifts.append(
                f"Turn {turn_idx}: {speaker_role} changed score from {turn_data.previous_score} to {turn_data.new_score} "
                f"when addressing {target_role} on '{topic}'. Reason: {turn_data.message}"
            )

    return DebateResult(
        candidate_name=candidate_name,
        transcript_logs=turn_logs,
        final_agent_scores=current_scores,
        opinion_shifts=opinion_shifts,
    )
