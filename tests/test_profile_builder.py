"""
Unit tests for Candidate Profile Builder and Agent Pydantic Schemas (Offline/Fast)
"""
import pytest
from pydantic import ValidationError
from profile_builder import CandidateProfile, ProjectHighlight
from agents.base_agent import AgentEvaluation, EvidenceQuote


def test_candidate_profile_valid_schema():
    """Test creating a CandidateProfile with valid data."""
    valid_data = {
        "candidate_name": "Candidate A",
        "skills": ["Python", "FastAPI", "Docker", "Chroma"],
        "years_experience": "3.5 years",
        "projects": [
            {
                "name": "Exception Handling Engine",
                "description": "Built an automated error handling pipeline for freight quotes",
                "technologies": ["Python", "CrewAI"]
            }
        ],
        "claims": ["Sole architect of exception engine", "Reduced errors by 40%"],
        "summary": "Solid backend engineer with intermediate agent experience."
    }
    
    profile = CandidateProfile.model_validate(valid_data)
    assert profile.candidate_name == "Candidate A"
    assert len(profile.skills) == 4
    assert profile.years_experience == "3.5 years"
    assert profile.projects[0].name == "Exception Handling Engine"


def test_candidate_profile_invalid_schema():
    """Test that missing required fields in CandidateProfile raises ValidationError."""
    invalid_data = {
        "candidate_name": "Candidate A"
        # Missing required fields: skills, years_experience, projects, claims, summary
    }
    with pytest.raises(ValidationError):
        CandidateProfile.model_validate(invalid_data)


def test_agent_evaluation_schema_valid():
    """Smoke test for AgentEvaluation Pydantic schema validation with valid dict."""
    valid_agent_dict = {
        "agent_name": "Technical Lead Agent",
        "opinion": "Candidate displays strong backend fundamentals in Python and FastAPI.",
        "score": 8,
        "evidence_quotes": [
            {"quote": "Built FastAPI microservices", "source": "resume"},
            {"quote": "Explained Chroma indexing clearly", "source": "transcript"}
        ],
        "confidence": "high",
        "insufficient_info_flags": ["No production experience with CrewAI"]
    }
    
    eval_obj = AgentEvaluation.model_validate(valid_agent_dict)
    assert eval_obj.agent_name == "Technical Lead Agent"
    assert eval_obj.score == 8
    assert eval_obj.confidence == "high"
    assert len(eval_obj.evidence_quotes) == 2
    assert eval_obj.evidence_quotes[0].source == "resume"


def test_agent_evaluation_schema_invalid():
    """Test that invalid values (e.g. score out of range or bad source string) raise ValidationError."""
    invalid_agent_dict = {
        "agent_name": "Technical Lead Agent",
        "opinion": "Opinion string",
        "score": "not_an_integer",  # Invalid type
        "evidence_quotes": [
            {"quote": "Quote text", "source": "invalid_source_type"}  # Source must be 'resume' or 'transcript'
        ],
        "confidence": "super_high",  # Must be 'low', 'medium', or 'high'
        "insufficient_info_flags": []
    }
    
    with pytest.raises(ValidationError):
        AgentEvaluation.model_validate(invalid_agent_dict)
