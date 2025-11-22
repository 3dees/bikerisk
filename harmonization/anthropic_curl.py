"""
Curl-based Anthropic API client as workaround for Python SSL/TLS issues.

Use this when Python's SSL layer (httpx, requests, urllib) fails but curl works.
"""
import json
import os
import subprocess
from typing import Optional


def call_anthropic_via_curl(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-5-20250929",
    max_tokens: int = 4096,
    temperature: float = 0.0
) -> str:
    """
    Call Anthropic API using curl as a workaround for Python SSL issues.

    Args:
        prompt: The prompt to send to Claude
        api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
        model: Model to use
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        The text response from Claude

    Raises:
        RuntimeError: If curl fails or returns error
    """
    # Get API key
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key provided and ANTHROPIC_API_KEY not set")

    # Build request payload
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    # Build curl command
    cmd = [
        "curl",
        "-X", "POST",
        "https://api.anthropic.com/v1/messages",
        "-H", f"x-api-key: {api_key}",
        "-H", "anthropic-version: 2023-06-01",
        "-H", "content-type: application/json",
        "-d", json.dumps(payload),
        "--silent",
        "--show-error",
        "--max-time", "180"  # 3 minute timeout
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=200,  # 200 seconds total
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"curl failed with code {result.returncode}: {result.stderr}"
            )

        # Parse response
        response_data = json.loads(result.stdout)

        # Check for API error
        if "error" in response_data:
            error_msg = response_data["error"].get("message", "Unknown error")
            raise RuntimeError(f"Anthropic API error: {error_msg}")

        # Extract text from response
        if "content" in response_data and len(response_data["content"]) > 0:
            return response_data["content"][0]["text"]
        else:
            raise RuntimeError(f"Unexpected response format: {result.stdout[:200]}")

    except subprocess.TimeoutExpired:
        raise RuntimeError("Curl request timed out after 200 seconds")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse API response as JSON: {e}\nResponse: {result.stdout[:500]}")
    except Exception as e:
        raise RuntimeError(f"Curl-based API call failed: {e}")


# Test function
if __name__ == "__main__":
    print("Testing curl-based Anthropic API client...")
    try:
        response = call_anthropic_via_curl("Say: API test successful")
        print(f"SUCCESS: {response}")
    except Exception as e:
        print(f"FAILED: {e}")
