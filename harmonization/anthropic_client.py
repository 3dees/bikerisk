"""
Centralized Anthropic API client creation.

This module provides a single source of truth for creating Anthropic clients
with consistent configuration across extraction, harmonization, and consolidation.
"""

import os
import httpx
from anthropic import Anthropic
from typing import Optional


def get_anthropic_client(
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    verbose: bool = True
) -> Anthropic:
    """
    Create an Anthropic client with standard configuration.
    
    This function handles:
    - API key resolution from environment
    - Proxy bypass for anthropic.com
    - HTTP client with timeout
    - Fallback to simple client if httpx fails
    
    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        timeout: HTTP timeout in seconds (default: 120.0)
        verbose: If True, prints initialization messages
    
    Returns:
        Configured Anthropic client
        
    Raises:
        ValueError: If no API key is provided or found in environment
    """
    # Get API key
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "No API key provided and ANTHROPIC_API_KEY not set. "
                "Please set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )
    
    # Setup proxy bypass for Anthropic API
    # This is needed in some corporate/proxy environments
    no_proxy = os.getenv('NO_PROXY', '')
    if 'anthropic.com' not in no_proxy:
        new_no_proxy = (
            f"{no_proxy},anthropic.com,*.anthropic.com" 
            if no_proxy 
            else "anthropic.com,*.anthropic.com"
        )
        os.environ['NO_PROXY'] = new_no_proxy
        if verbose:
            print(f"[ANTHROPIC CLIENT] Set NO_PROXY to include anthropic.com")
    
    # Initialize client with timeout
    try:
        http_client = httpx.Client(timeout=timeout)
        client = Anthropic(api_key=api_key, http_client=http_client)
        if verbose:
            print(f"[ANTHROPIC CLIENT] Initialized with {timeout}s timeout")
        return client
    except Exception as e:
        if verbose:
            print(f"[ANTHROPIC CLIENT] httpx.Client init failed: {e}")
            print(f"[ANTHROPIC CLIENT] Falling back to simple client (no timeout)")
        # Fallback to simple client without httpx
        return Anthropic(api_key=api_key)
