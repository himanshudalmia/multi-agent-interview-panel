"""
PDF Reader Utility
Extracts clean text from PDF documents using pypdf.
"""
from pathlib import Path
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extracts raw text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text string from all pages.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        ValueError: If the file is not a valid PDF or extraction fails.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    try:
        reader = PdfReader(str(path))
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text.strip())
        
        full_text = "\n\n".join(extracted_pages)
        if not full_text.strip():
            raise ValueError(f"No readable text extracted from PDF: {pdf_path}")
            
        return full_text
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise e
        raise ValueError(f"Failed to read PDF file '{pdf_path}': {str(e)}") from e
