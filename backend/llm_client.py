
import json
import logging
import os
import re
from typing import Optional
logger = logging.getLogger(__name__)
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-5")
class LLMUnavailableError(RuntimeError):
    """Raised when no LLM provider is configured or the call fails."""
def is_llm_available() -> bool:
    """True if an Anthropic API key is configured in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
_client = None
def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client
def _extract_json(raw_text: str) -> dict:
    """
    Best-effort extraction of a JSON object from an LLM's raw text reply.
    Handles the common case of the model wrapping JSON in ```json fences.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    return json.loads(text)
def call_llm_json(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> dict:
    """
    Call the configured LLM and parse its reply as JSON.
    Raises
    ------
    LLMUnavailableError
        If no provider is configured, the API call fails, or the reply
        cannot be parsed as JSON. Callers (the judge agents) catch this
        and fall back to their heuristic evaluator.
    """
    if not is_llm_available():
        raise LLMUnavailableError("No ANTHROPIC_API_KEY configured in the environment.")
    try:
        client = _get_client()
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return _extract_json(raw_text)
    except Exception as exc:
        logger.warning("LLM call failed, caller will use heuristic fallback: %s", exc)
        raise LLMUnavailableError(str(exc)) from exc
