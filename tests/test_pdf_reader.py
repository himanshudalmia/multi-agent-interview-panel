"""
Unit tests for PDF reader utility
"""
from pathlib import Path
import pytest
from utils.pdf_reader import extract_text_from_pdf


def test_extract_text_from_valid_pdf():
    """Test extracting text from an existing valid PDF document."""
    sample_pdf = Path(__file__).parent.parent / "data" / "02_Job_Description.pdf"
    assert sample_pdf.exists(), f"Sample PDF missing at {sample_pdf}"
    
    text = extract_text_from_pdf(sample_pdf)
    assert isinstance(text, str)
    assert len(text.strip()) > 0
    assert "Job Description" in text or "Cargonet" in text


def test_extract_text_from_missing_pdf():
    """Test that extract_text_from_pdf raises FileNotFoundError for non-existent file."""
    fake_path = "data/non_existent_file_12345.pdf"
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(fake_path)
