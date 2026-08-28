"""
Main Pipeline Runner for Multi-Agent AI Interview Panel Simulator
Processes Candidates A and B through Profile Builder, 4 Independent Agents, Turn-Based Debate, Judge Adjudication, and Report Generation.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from google import genai

from utils.pdf_reader import extract_text_from_pdf
from profile_builder import build_candidate_profile
from agents import (
    evaluate_technical,
    evaluate_hr,
    evaluate_hiring_manager,
    evaluate_skeptic,
)
from debate import run_debate
from judge import adjudicate
from report import generate_candidate_report


def check_api_key() -> str:
    """Ensures GEMINI_API_KEY environment variable is configured."""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n" + "=" * 70)
        print(" [!] ERROR: GEMINI_API_KEY environment variable is missing.")
        print("=" * 70)
        print("To fix this, create a file named '.env' in this directory with:")
        print("    GEMINI_API_KEY=your_actual_gemini_api_key_here")
        print("\nOr export it in your shell environment:")
        print("  Windows PowerShell: $env:GEMINI_API_KEY='your_key'")
        print("  Linux/Mac bash:     export GEMINI_API_KEY='your_key'")
        print("=" * 70 + "\n")
        sys.exit(1)
    return api_key


def process_candidate(
    candidate_id: str,
    resume_path: Path | str,
    transcript_path: Path | str,
    job_description_text: str,
    output_dir: Path,
    client: genai.Client,
    model: str = "gemini-3.6-flash",
) -> dict:
    """Processes a single candidate through the full 5-stage multi-agent pipeline.

    Args:
        candidate_id: Human-readable ID (e.g., 'Candidate A')
        resume_path: File path to resume PDF
        transcript_path: File path to interview transcript PDF
        job_description_text: Text of job description
        output_dir: Path to directory for report outputs
        client: genai.Client instance
        model: Gemini model ID

    Returns:
        Summary dict containing verdict, confidence, and scores.
    """
    print(f"\n========================================================")
    print(f"   STARTING PIPELINE FOR: {candidate_id}")
    print(f"========================================================")

    # 1. Read PDF texts
    print(f"[1/5] Extracting PDF texts for {candidate_id}...")
    resume_text = extract_text_from_pdf(resume_path)
    transcript_text = extract_text_from_pdf(transcript_path)

    # 2. Build Structured Candidate Profile (1 Gemini call)
    print(f"[2/5] Building Candidate Profile via Gemini API...")
    profile = build_candidate_profile(
        candidate_name=candidate_id,
        resume_text=resume_text,
        transcript_text=transcript_text,
        job_description_text=job_description_text,
        client=client,
        model=model,
    )
    print(f"      -> Extracted Skills: {len(profile.skills)} skills found.")
    print(f"      -> Stated Experience: {profile.years_experience}")

    # 3. Four Independent Agent Evaluations (4 fully separate Gemini calls)
    print(f"[3/5] Running 4 Independent Agent Evaluations (Zero Cross-Visibility)...")
    
    import time
    print("      -> Running Technical Lead Agent...")
    tech_eval = evaluate_technical(
        candidate_id, profile, resume_text, transcript_text, job_description_text, client, model
    )
    time.sleep(2)

    print("      -> Running HR Culture Agent...")
    hr_eval = evaluate_hr(
        candidate_id, profile, resume_text, transcript_text, job_description_text, client, model
    )
    time.sleep(2)

    print("      -> Running Hiring Manager Agent...")
    hm_eval = evaluate_hiring_manager(
        candidate_id, profile, resume_text, transcript_text, job_description_text, client, model
    )
    time.sleep(2)

    print("      -> Running Skeptic Agent (Red Flag Investigator)...")
    skeptic_eval = evaluate_skeptic(
        candidate_id, profile, resume_text, transcript_text, job_description_text, client, model
    )
    time.sleep(2)

    initial_evaluations = {
        "Technical Lead Agent": tech_eval,
        "HR Culture Agent": hr_eval,
        "Hiring Manager Agent": hm_eval,
        "Skeptic Agent": skeptic_eval,
    }

    print("      -> Initial Scores:")
    for role, ev in initial_evaluations.items():
        print(f"         * {role:22s}: {ev.score}/10 (Conf: {ev.confidence})")

    # 4. Turn-Based Panel Debate Step (4 turn-based cross-examination Gemini calls)
    print(f"[4/5] Orchestrating Panel Debate & Cross-Examination...")
    debate_res = run_debate(
        candidate_name=candidate_id,
        candidate_profile=profile,
        initial_evaluations=initial_evaluations,
        resume_text=resume_text,
        transcript_text=transcript_text,
        job_description_text=job_description_text,
        client=client,
        model=model,
    )

    print("      -> Debate Completed. Post-Debate Agent Scores:")
    for role, post_score in debate_res.final_agent_scores.items():
        init_score = initial_evaluations[role].score
        change_indicator = f" (Shifted from {init_score})" if init_score != post_score else ""
        print(f"         * {role:22s}: {post_score}/10{change_indicator}")

    if debate_res.opinion_shifts:
        print(f"      -> Traceable Opinion Shifts ({len(debate_res.opinion_shifts)} recorded):")
        for shift in debate_res.opinion_shifts:
            print(f"         - {shift}")

    # 5. Final Decision / Judicial Adjudication (1 Gemini call, step-by-step reasoning)
    print(f"[5/5] Running Judicial Evidence Adjudication...")
    judge_dec = adjudicate(
        candidate_name=candidate_id,
        candidate_profile=profile,
        initial_evaluations=initial_evaluations,
        debate_result=debate_res,
        resume_text=resume_text,
        transcript_text=transcript_text,
        job_description_text=job_description_text,
        client=client,
        model=model,
    )

    print(f"\n   >>> FINAL VERDICT FOR {candidate_id}: {judge_dec.final_recommendation} (Confidence: {judge_dec.confidence}) <<<")

    # 6. Generate Markdown Report
    filename = f"{candidate_id.lower().replace(' ', '_')}_report.md"
    report_file_path = output_dir / filename
    generate_candidate_report(
        candidate_name=candidate_id,
        profile=profile,
        initial_evaluations=initial_evaluations,
        debate_result=debate_res,
        judge_decision=judge_dec,
        output_path=report_file_path,
    )
    print(f"   -> Markdown report saved to: {report_file_path}")

    return {
        "candidate_id": candidate_id,
        "recommendation": judge_dec.final_recommendation,
        "confidence": judge_dec.confidence,
        "scores": debate_res.final_agent_scores,
        "report_path": report_file_path,
    }


def main():
    """Main execution pipeline."""
    api_key = check_api_key()
    client = genai.Client(api_key=api_key)
    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

    base_dir = Path(__file__).parent.resolve()
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Job description
    jd_path = data_dir / "02_Job_Description.pdf"
    if not jd_path.exists():
        print(f"[!] Error: Job description PDF not found at {jd_path}")
        sys.exit(1)
    
    print("Loading Job Description...")
    job_description_text = extract_text_from_pdf(jd_path)

    # Candidates to process
    candidates = [
        {
            "id": "Candidate A",
            "resume": data_dir / "03_Resume_A.pdf",
            "transcript": data_dir / "05_Transcript_A.pdf",
        },
        {
            "id": "Candidate B",
            "resume": data_dir / "04_Resume_B.pdf",
            "transcript": data_dir / "06_Transcript_B.pdf",
        },
    ]

    results = []
    for cand in candidates:
        if not cand["resume"].exists() or not cand["transcript"].exists():
            print(f"[!] Warning: Missing files for {cand['id']}. Skipping.")
            continue
        
        res = process_candidate(
            candidate_id=cand["id"],
            resume_path=cand["resume"],
            transcript_path=cand["transcript"],
            job_description_text=job_description_text,
            output_dir=output_dir,
            client=client,
            model=model,
        )
        results.append(res)

    # Final Panel Comparative Summary
    print("\n" + "=" * 70)
    print("              PANEL EVALUATION COMPARATIVE SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"Candidate: {r['candidate_id']:15s} | Recommendation: {r['recommendation']:15s} | Confidence: {r['confidence']}")
        print(f"Final Scores: {r['scores']}")
        print(f"Report: {r['report_path']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
