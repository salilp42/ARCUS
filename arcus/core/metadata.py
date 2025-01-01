"""Metadata verification using LLM."""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

# Optional import for LLM usage
try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)


def run_llm_metadata_check(text: str, api_key: str = "") -> str:
    """Run LLM-based metadata verification.
    
    Args:
        text: Text to analyze
        api_key: OpenAI API key
        
    Returns:
        Analysis result as string
    """
    if openai is None:
        logger.warning("OpenAI not installed. Skipping LLM check.")
        return "Skipped LLM check."
        
    if not api_key:
        logger.warning("No OpenAI API key provided. Skipping LLM check.")
        return "Skipped LLM check."

    openai.api_key = api_key
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Analyze the following metadata for inconsistencies:\n\n{text}",
            max_tokens=100,
            temperature=0.0
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        return "LLM check failed"


def metadata_verification(ehr_file: str, llm_api_key: str = "") -> str:
    """Verify EHR metadata using LLM.
    
    Args:
        ehr_file: Path to EHR CSV file
        llm_api_key: OpenAI API key
        
    Returns:
        Verification result
    """
    if not Path(ehr_file).exists():
        logger.warning(f"EHR file {ehr_file} does not exist.")
        return "EHR file not found."

    df = pd.read_csv(ehr_file)
    sample_text = df.head().to_string()

    result = run_llm_metadata_check(sample_text, api_key=llm_api_key)
    return result
