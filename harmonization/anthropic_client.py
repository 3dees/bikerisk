"""
Centralized Anthropic API client creation.

This module provides a single source of truth for creating Anthropic clients
with consistent configuration across extraction, harmonization, and consolidation.
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional, Iterable

import httpx
from anthropic import Anthropic
import anthropic
from anthropic import APIConnectionError, APIStatusError, RateLimitError

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------

logger = logging.getLogger("anthropic_client")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Default to INFO, caller can override in app entry point
logger.setLevel(logging.INFO)


# -------------------------------------------------------------------
# Environment handling
# -------------------------------------------------------------------

def _build_no_proxy() -> str:
    """
    Ensure NO_PROXY covers Anthropic domains so local proxies do not
    MITM TLS. Merge with any existing NO_PROXY.
    """
    existing = os.environ.get("NO_PROXY", "") or os.environ.get("no_proxy", "")
    extra = ["anthropic.com", "api.anthropic.com", ".anthropic.com"]

    parts = [p.strip() for p in existing.split(",") if p.strip()] if existing else []
    for host in extra:
        if host not in parts:
            parts.append(host)

    new_val = ",".join(parts)
    # Set both cases to be safe
    os.environ["NO_PROXY"] = new_val
    os.environ["no_proxy"] = new_val
    return new_val


def _should_trust_env() -> bool:
    """
    ANTHROPIC_TRUST_ENV:
    - unset or "1" or "true" (case insensitive) => trust_env=True
    - "0" or "false" => trust_env=False
    """
    raw = os.environ.get("ANTHROPIC_TRUST_ENV", "").strip().lower()
    if raw in ("0", "false", "no"):
        return False
    if raw in ("1", "true", "yes"):
        return True
    # Default: trust env, but we still override NO_PROXY for Anthropic
    return True


def _merge_no_proxy(values: Iterable[str]) -> str:
    """Merge and normalize NO_PROXY values into a comma-separated string."""
    items = []
    for v in values:
        if not v:
            continue
        for part in str(v).split(","):
            part = part.strip()
            if part and part not in items:
                items.append(part)
    return ",".join(items)


def _ensure_no_proxy_for_anthropic(verbose: bool = True) -> None:
    """Ensure NO_PROXY and no_proxy include Anthropic domains.

    Notes:
    - Many HTTP stacks don't honor wildcard patterns like *.domain; prefer suffix ".domain".
    - We include both "anthropic.com" and "api.anthropic.com" and suffix ".anthropic.com".
    - Set both NO_PROXY and no_proxy to cover case-sensitive checks on Windows.
    """
    existing_upper = os.getenv("NO_PROXY", "")
    existing_lower = os.getenv("no_proxy", "")
    required = ["anthropic.com", "api.anthropic.com", ".anthropic.com"]
    merged = _merge_no_proxy([existing_upper, existing_lower] + required)
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged
    if verbose:
        logger.info(f"NO_PROXY configured: {merged}")


# -------------------------------------------------------------------
# Centralized Anthropic client
# -------------------------------------------------------------------

_no_proxy = _build_no_proxy()
_trust_env = _should_trust_env()

logger.info(f"Anthropic trust_env: {_trust_env}")

# Read API key once; can also rely on default ANTHROPIC_API_KEY env
_api_key = os.environ.get("ANTHROPIC_API_KEY")
if not _api_key:
    logger.warning("ANTHROPIC_API_KEY not set. Client will fail without it.")

_default_timeout = httpx.Timeout(
    connect=10.0,
    read=120.0,
    write=10.0,
    pool=5.0,
)

_http_transport = httpx.HTTPTransport(
    verify=True,
    http2=False,          # Important: force HTTP/1.1 for stability
    retries=0,            # We implement our own retry loop
)

_http_client = httpx.Client(
    timeout=_default_timeout,
    transport=_http_transport,
    trust_env=_trust_env,
)

anthropic_client = Anthropic(
    api_key=_api_key,
    http_client=_http_client,
)


# -------------------------------------------------------------------
# Retry wrapper
# -------------------------------------------------------------------

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.PoolTimeout)):
        return True
    return False


def messages_create_with_retries(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_attempts: int = 4,
    initial_backoff: float = 1.0,
    max_backoff: float = 10.0,
    **kwargs: Any,
) -> anthropic.types.Message:
    """
    Central wrapper that all Anthropic message calls should use.

    - Retries on connection failures and transient status codes
    - Simple exponential backoff with cap
    - Logs attempts and failures
    """
    attempt = 0
    last_exc: Optional[Exception] = None

    while attempt < max_attempts:
        attempt += 1
        try:
            start = time.time()
            logger.info(
                f"Anthropic call start - model={model}, attempt={attempt}/{max_attempts}"
            )

            resp = anthropic_client.messages.create(
                model=model,
                messages=messages,
                **kwargs,
            )

            elapsed = time.time() - start
            logger.info(
                f"Anthropic call success - model={model}, "
                f"attempt={attempt}, elapsed={elapsed:.2f}s"
            )
            return resp

        except Exception as exc:
            last_exc = exc
            if not _is_retryable_exception(exc) or attempt >= max_attempts:
                logger.error(
                    f"Anthropic call failed - model={model}, attempt={attempt}, "
                    f"giving up. Exception: {exc!r}"
                )
                raise

            backoff = min(initial_backoff * (2 ** (attempt - 1)), max_backoff)
            logger.warning(
                f"Anthropic call error (attempt {attempt}/{max_attempts}) - "
                f"retrying in {backoff:.1f}s. Exception: {exc!r}"
            )
            time.sleep(backoff)

    # Should not reach here, defensive
    if last_exc:
        raise last_exc
    raise RuntimeError("Anthropic call failed after retries without exception")


# -------------------------------------------------------------------
# Legacy compatibility (Phase 1)
# -------------------------------------------------------------------
# For callers still expecting get_anthropic_client, we keep it briefly.
# Eventually all callsites should just use messages_create_with_retries
# directly, which uses the global anthropic_client.


def get_anthropic_client(
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    verbose: bool = False,
    trust_env: Optional[bool] = None,
) -> Anthropic:
    """
    Legacy compatibility shim. Returns the global anthropic_client.
    
    Parameters are ignored; the global client is pre-configured with
    environment-based settings. Keep for backwards compatibility during
    the transition period.
    """
    if verbose:
        logger.info("get_anthropic_client() called; returning global anthropic_client")
    return anthropic_client
