"""
Retry Helper Module
Provides rate-limit handling and automatic retries for Gemini API calls.
"""
import time
from typing import Any
from google.genai import errors


def generate_content_with_retry(
    client: Any,
    model: str,
    contents: Any,
    config: Any,
    max_retries: int = 6,
    initial_delay: float = 15.0,
) -> Any:
    """Executes client.models.generate_content with automatic retry on 429 RESOURCE_EXHAUSTED rate limits.

    Args:
        client: genai.Client instance
        model: Gemini model ID
        contents: Prompt contents
        config: GenerateContentConfig object
        max_retries: Maximum number of retry attempts
        initial_delay: Initial sleep delay in seconds

    Returns:
        GenerateContentResponse object.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or getattr(e, "code", None) == 429:
                if attempt < max_retries:
                    print(f"      [!] Rate limit encountered (429). Retrying in {delay:.1f}s (Attempt {attempt}/{max_retries})...")
                    time.sleep(delay)
                    delay *= 1.5
                else:
                    raise e
            else:
                raise e
