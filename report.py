"""
Report Generator Module
Formats and generates comprehensive candidate markdown evaluation reports.
"""
from pathlib import Path
from typing import Dict, Optional
from agents.base_agent import AgentEvaluation
from debate import DebateResult
from judge import JudgeDecision
from profile_builder import CandidateProfile


def generate_candidate_report(
    candidate_name: str,
    profile: CandidateProfile,
    initial_evaluations: Dict[str, AgentEvaluation],
    debate_result: DebateResult,
    judge_decision: JudgeDecision,
    output_path: Optional[str | Path] = None,
) -> str:
    """Generates a structured markdown report for a candidate.

    Args:
        candidate_name: Identifier for candidate (e.g. Candidate A)
        profile: CandidateProfile pydantic model
        initial_evaluations: Dict mapping agent roles to initial evaluations
        debate_result: DebateResult object containing turn logs and final agent scores
        judge_decision: JudgeDecision object with verdict and step-by-step reasoning
        output_path: Optional file path to save the generated markdown

    Returns:
        Markdown string containing complete report.
    """
    md_lines = []

    # Title & Header
    md_lines.append(f"# Multi-Agent Interview Evaluation Report: {candidate_name}")
    md_lines.append("")

    # Executive Summary Card / Verdict Alert Box
    rec_upper = judge_decision.final_recommendation.upper()
    alert_type = "NOTE"
    if "STRONG HIRE" in rec_upper or "HIRE" in rec_upper:
        alert_type = "TIP"
    elif "NO HIRE" in rec_upper:
        alert_type = "WARNING"
    elif "WEAK HIRE" in rec_upper:
        alert_type = "IMPORTANT"

    md_lines.append(f"> [{alert_type}]")
    md_lines.append(f"> **FINAL RECOMMENDATION:** `{judge_decision.final_recommendation}`")
    md_lines.append(f"> **CONFIDENCE LEVEL:** `{judge_decision.confidence.upper()}`")
    md_lines.append(f"> **CANDIDATE:** {candidate_name} ({profile.candidate_name})")
    md_lines.append("")

    # Candidate Facts & Profile Summary
    md_lines.append("## 1. Candidate Overview & Extracted Profile")
    md_lines.append(f"**Experience:** {profile.years_experience}")
    md_lines.append(f"**Summary:** {profile.summary}")
    md_lines.append("")
    md_lines.append("**Key Skills:**")
    md_lines.append(", ".join([f"`{s}`" for s in profile.skills]))
    md_lines.append("")

    if profile.projects:
        md_lines.append("**Highlighted Projects:**")
        for proj in profile.projects:
            techs = f" (Tech: {', '.join(proj.technologies)})" if proj.technologies else ""
            md_lines.append(f"- **{proj.name}**: {proj.description}{techs}")
        md_lines.append("")

    if profile.claims:
        md_lines.append("**Key Stated Claims:**")
        for claim in profile.claims:
            md_lines.append(f"- {claim}")
        md_lines.append("")

    # Panel Evaluation Matrix
    md_lines.append("## 2. Panel Agent Score Matrix (Pre vs Post Debate)")
    md_lines.append("| Agent Persona | Initial Score | Post-Debate Score | Initial Confidence | Insufficient Info Flags |")
    md_lines.append("| :--- | :---: | :---: | :---: | :--- |")

    for role, init_eval in initial_evaluations.items():
        post_score = debate_result.final_agent_scores.get(role, init_eval.score)
        flags = ", ".join(init_eval.insufficient_info_flags) if init_eval.insufficient_info_flags else "None"
        md_lines.append(
            f"| **{role}** | {init_eval.score}/10 | **{post_score}/10** | {init_eval.confidence} | {flags} |"
        )
    md_lines.append("")

    # Independent Opinions & Quotes
    md_lines.append("## 3. Independent Agent Opinions & Evidence Quotes")
    for role, init_eval in initial_evaluations.items():
        md_lines.append(f"### {role}")
        md_lines.append(f"**Initial Assessment (Score: {init_eval.score}/10):**")
        md_lines.append(f"{init_eval.opinion}")
        md_lines.append("")
        if init_eval.evidence_quotes:
            md_lines.append("**Supporting Evidence Quotes:**")
            for q in init_eval.evidence_quotes:
                md_lines.append(f"- *[{q.source.upper()}]* \"{q.quote}\"")
            md_lines.append("")

    # Debate Highlights & Traceable Opinion Shifts
    md_lines.append("## 4. Multi-Agent Turn-Based Debate Transcript & Score Shifts")
    if debate_result.opinion_shifts:
        md_lines.append("> [!IMPORTANT]")
        md_lines.append("> **Traceable Opinion Shifts During Debate:**")
        for shift in debate_result.opinion_shifts:
            md_lines.append(f"> - {shift}")
        md_lines.append("")

    md_lines.append("### Debate Turn Log:")
    for turn in debate_result.transcript_logs:
        changed_str = " (OPINION CHANGED!)" if turn.changed_opinion else ""
        md_lines.append(f"#### Turn {turn.turn_number}: {turn.agent} → {turn.target_agent}{changed_str}")
        md_lines.append(f"**Topic:** {turn.topic}")
        md_lines.append(f"**Score Transition:** `{turn.previous_score}/10` → `{turn.new_score}/10` (Confidence: {turn.revised_confidence})")
        md_lines.append(f"**Argument/Response:**\n{turn.message}")
        md_lines.append("")

    # Judicial Reasoning & Key Evidence
    md_lines.append("## 5. Judicial Adjudication & Evidence Weighing")
    md_lines.append("*(Decision reached through step-by-step judicial evidence synthesis — NOT simple score averaging)*")
    md_lines.append("")
    md_lines.append(f"### Reasoning:\n{judge_decision.reasoning}")
    md_lines.append("")

    md_lines.append("### Key Evidence & Quotes That Decided Verdict:")
    for ev in judge_decision.key_evidence:
        md_lines.append(f"- {ev}")
    md_lines.append("")

    if judge_decision.unresolved_disagreements:
        md_lines.append("### Unresolved Panel Disagreements / Open Risks:")
        for dis in judge_decision.unresolved_disagreements:
            md_lines.append(f"- ⚠️ {dis}")
        md_lines.append("")

    report_content = "\n".join(md_lines)

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_content, encoding="utf-8")

    return report_content
