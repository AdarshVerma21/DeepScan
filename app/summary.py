from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from groq import Groq

from .config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_DAILY_CALL_LIMIT,
    ANTHROPIC_MODEL,
    ANTHROPIC_RATE_LIMIT_REQUESTS,
    ANTHROPIC_RATE_LIMIT_WINDOW_SECONDS,
    ANTHROPIC_TIMEOUT_SECONDS,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TIMEOUT_SECONDS,
)

LOGGER = logging.getLogger(__name__)
SUMMARY_RATE_LOCK = threading.Lock()
SUMMARY_REQUESTS_BY_CLIENT: dict[str, list[float]] = {}
SUMMARY_CALLS_BY_DAY: dict[str, int] = {}
SYSTEM_PROMPT = """You are a digital-forensics analyst. Using ONLY the evidence provided, write plain-language markdown with exactly these ## headings, in this order: "Verdict", "Key reasons", "What's uncertain", "Next steps". Under each heading use short bullets. State the verdict and confidence first. Explain technical terms in parentheses. Reference specific evidence and video timestamps when provided. Never claim certainty, never invent evidence, and do not add other headings."""


@dataclass(frozen=True)
class SummaryResult:
    markdown: str
    is_fallback: bool


def _fallback(evidence: dict[str, Any]) -> str:
    lines = [
        "## Verdict",
        f"- **{evidence.get('verdict', 'Inconclusive')}** with {evidence.get('confidence', 'Low')} confidence.",
        f"- Fake-likelihood score: {evidence.get('fake_score', 50)}/100.",
        "",
        "## Key reasons",
    ]
    for item in evidence.get("evidence", [])[:5]:
        lines.append(f"- **{item.get('signal', 'Signal')}:** {item.get('finding', 'No finding provided')}")
    lines += [
        "",
        "## What's uncertain",
        "- This is a probabilistic screening result, not proof. Compression, missing faces, metadata stripping, and detector disagreement can affect it.",
        *[f"- {item}" for item in evidence.get("limitations", [])],
        "",
        "## Next steps",
        "- Preserve the original file and compare it with a trusted source or capture history.",
    ]
    return "\n".join(lines)


def deterministic_fallback(evidence: dict[str, Any]) -> str:
    return _fallback(evidence)


def _log_failure(exc: Exception) -> None:
    name = type(exc).__name__
    if name == "AuthenticationError":
        reason = "authentication failed; check GROQ_API_KEY"
    elif name == "RateLimitError":
        reason = "rate limited by Groq"
    elif name in {"APITimeoutError", "TimeoutError"}:
        reason = "request timed out"
    elif name in {"APIConnectionError", "ConnectError"}:
        reason = "network connection failed"
    elif name in {"BadRequestError", "NotFoundError"}:
        reason = "request rejected; check GROQ_MODEL and payload"
    elif isinstance(exc, ValueError):
        reason = "provider returned an empty or invalid explanation"
    else:
        reason = "provider request failed"
    LOGGER.warning("Groq summary unavailable (%s): %s; using deterministic fallback", reason, name)


def _request_summary(evidence: dict[str, Any]) -> str:
    client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS, max_retries=1)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=700,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(evidence, default=str)},
        ],
    )
    choices = getattr(response, "choices", None)
    message = getattr(choices[0], "message", None) if choices else None
    text = getattr(message, "content", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Groq returned an empty explanation")
    return text.strip()

def _request_anthropic_summary(evidence: dict[str, Any]) -> str:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("Anthropic SDK is not installed") from exc
    client = Anthropic(api_key=ANTHROPIC_API_KEY, timeout=ANTHROPIC_TIMEOUT_SECONDS, max_retries=1)
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(evidence, default=str)}],
    )
    content = getattr(response, "content", None)
    text = "".join(
        str(getattr(block, "text", ""))
        for block in (content or [])
        if getattr(block, "type", "") == "text"
    ).strip()
    if not text:
        raise ValueError("Anthropic returned an empty explanation")
    return text

def _reserve_anthropic_call(client_key: str) -> bool:
    if not ANTHROPIC_API_KEY:
        return False
    now = time.monotonic()
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with SUMMARY_RATE_LOCK:
        timestamps = SUMMARY_REQUESTS_BY_CLIENT.setdefault(client_key, [])
        cutoff = now - ANTHROPIC_RATE_LIMIT_WINDOW_SECONDS
        timestamps[:] = [timestamp for timestamp in timestamps if timestamp > cutoff]
        if len(timestamps) >= max(1, ANTHROPIC_RATE_LIMIT_REQUESTS):
            LOGGER.warning("Anthropic per-client summary limit reached for %s", client_key)
            return False
        if SUMMARY_CALLS_BY_DAY.get(today, 0) >= max(1, ANTHROPIC_DAILY_CALL_LIMIT):
            LOGGER.warning("Anthropic daily summary limit reached")
            return False
        timestamps.append(now)
        SUMMARY_CALLS_BY_DAY[today] = SUMMARY_CALLS_BY_DAY.get(today, 0) + 1
        for day in list(SUMMARY_CALLS_BY_DAY):
            if day != today:
                del SUMMARY_CALLS_BY_DAY[day]
        return True


async def generate_summary(evidence: dict[str, Any], client_key: str = "unknown") -> SummaryResult:
    if ANTHROPIC_API_KEY:
        if not _reserve_anthropic_call(client_key):
            return SummaryResult(_fallback(evidence), True)
        try:
            text = await asyncio.to_thread(_request_anthropic_summary, evidence)
            return SummaryResult(text, False)
        except Exception as exc:
            _log_failure(exc)
            return SummaryResult(_fallback(evidence), True)
    if not GROQ_API_KEY:
        LOGGER.info("GROQ_API_KEY is not configured; using deterministic fallback")
        return SummaryResult(_fallback(evidence), True)
    try:
        text = await asyncio.to_thread(_request_summary, evidence)
        return SummaryResult(text, False)
    except Exception as exc:
        _log_failure(exc)
        return SummaryResult(_fallback(evidence), True)
