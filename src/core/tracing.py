"""LangSmith tracing configuration and utilities.

This module sets up LangSmith tracing for the full email processing pipeline,
including LLM calls, node execution, and service operations.

Tracing levels:
1. LangGraph workflow: Automatic when LANGCHAIN_TRACING_V2=true (node transitions, state)
2. Node functions: @traceable(run_type="chain") captures inputs/outputs of each node
3. LLM calls: wrap_openai() patches AsyncOpenAI to capture every API request/response
   with full prompt messages, completions, token usage, model name, and latency
"""

import os
from typing import Any, Callable

from loguru import logger

from src.core.config import settings


def setup_langsmith():
    """Initialize LangSmith tracing by setting required environment variables.

    LangSmith uses environment variables for configuration. This function
    ensures they are set from our application settings before any
    LangChain/LangGraph code runs.

    Returns:
        True if tracing was enabled, False otherwise.
    """
    if not settings.LANGSMITH_ENABLED:
        logger.info("LangSmith tracing is disabled (LANGSMITH_ENABLED=false)")
        return False

    if not settings.LANGCHAIN_API_KEY:
        logger.warning("LangSmith enabled but LANGCHAIN_API_KEY is not set - tracing disabled")
        return False

    # Set environment variables that LangSmith/LangChain SDK reads
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT

    logger.info(
        f"LangSmith tracing enabled: "
        f"project={settings.LANGCHAIN_PROJECT}, "
        f"endpoint={settings.LANGCHAIN_ENDPOINT}"
    )
    return True


def get_traceable():
    """Get the langsmith traceable decorator if available.

    Returns a no-op decorator if langsmith is not installed or tracing is disabled.
    """
    if not settings.LANGSMITH_ENABLED:
        def noop_decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def wrapper(func):
                return func
            return wrapper
        return noop_decorator

    try:
        from langsmith import traceable
        return traceable
    except ImportError:
        logger.warning("langsmith package not installed - tracing decorators disabled")
        def noop_decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            def wrapper(func):
                return func
            return wrapper
        return noop_decorator


def wrap_openai_client(client):
    """Wrap an OpenAI client with LangSmith tracing.

    This patches the AsyncOpenAI client so every chat.completions.create() call
    is automatically traced in LangSmith with:
    - Input messages (system prompt, user prompt)
    - Output message (assistant response)
    - Token usage (prompt_tokens, completion_tokens, total_tokens)
    - Model name
    - Request latency
    - Temperature, max_tokens, and other parameters

    Args:
        client: An AsyncOpenAI client instance

    Returns:
        The wrapped client (or original if tracing is disabled)
    """
    if not settings.LANGSMITH_ENABLED:
        return client

    try:
        from langsmith.wrappers import wrap_openai
        wrapped = wrap_openai(client)
        logger.info("OpenAI client wrapped with LangSmith tracing")
        return wrapped
    except ImportError:
        logger.warning("langsmith.wrappers not available - OpenAI client not wrapped")
        return client
    except Exception as e:
        logger.warning(f"Failed to wrap OpenAI client with LangSmith: {e}")
        return client


# Module-level traceable decorator
traceable = get_traceable()
