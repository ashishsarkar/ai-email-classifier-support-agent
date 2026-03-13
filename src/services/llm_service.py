"""LLM service with circuit breaker failover between primary (vLLM/SLM) and fallback (Claude).

Traffic flow:
  Normal:    All calls → Primary (vLLM + Qwen)
  Failover:  Primary down → Circuit opens → All calls → Fallback (Claude via LiteLLM)
  Recovery:  Cooldown expires → Test call to primary → If OK, traffic returns to primary
"""

import json
import math
from typing import Optional, Dict, Any, List

from loguru import logger
from openai import AsyncOpenAI

from src.core.config import settings
from src.core.tracing import traceable, wrap_openai_client
from src.prompts.templates import (
    SYSTEM_PROMPT_CUSTOMER_SUPPORT,
    EMAIL_CLASSIFICATION_PROMPT,
    EMAIL_PRIORITY_PROMPT,
    RESPONSE_GENERATION_PROMPT,
    SENTIMENT_ANALYSIS_PROMPT,
)
from src.services.base import BaseService
from src.services.circuit_breaker import CircuitBreaker


class LLMService(BaseService):
    """Service for LLM operations with circuit breaker failover."""

    def __init__(self):
        """Initialize primary and fallback LLM clients."""
        # Primary client: vLLM / SLM (Qwen)
        self.primary_client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.VLLM_BASE_URL or None,
        )
        self.primary_client = wrap_openai_client(self.primary_client)
        self.primary_model = settings.OPENAI_MODEL

        # Fallback client: Claude via LiteLLM (OpenAI-compatible proxy)
        self.fallback_client: Optional[AsyncOpenAI] = None
        self.fallback_model = settings.FALLBACK_MODEL

        if settings.ANTHROPIC_API_KEY:
            self.fallback_client = AsyncOpenAI(
                api_key=settings.ANTHROPIC_API_KEY,
                base_url=settings.FALLBACK_BASE_URL or None,
            )
            self.fallback_client = wrap_openai_client(self.fallback_client)
            logger.info(
                f"Fallback LLM configured: {self.fallback_model} "
                f"(base_url={settings.FALLBACK_BASE_URL})"
            )
        else:
            logger.warning(
                "No ANTHROPIC_API_KEY set — fallback LLM disabled. "
                "If primary goes down, all LLM calls will fail."
            )

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name="llm",
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            cooldown_seconds=settings.CB_COOLDOWN_SECONDS,
            half_open_max_calls=settings.CB_HALF_OPEN_MAX_CALLS,
        )

        # Legacy compatibility
        self.client = self.primary_client
        self.model = self.primary_model
        self.total_tokens_used = 0

    def _get_active_client(self) -> tuple[AsyncOpenAI, str]:
        """Return (client, model) based on circuit breaker state.

        Returns:
            Tuple of (client, model_name) for the currently active provider.

        Raises:
            RuntimeError: If fallback is needed but not configured.
        """
        if self.circuit_breaker.should_use_fallback:
            if not self.fallback_client:
                logger.error(
                    "Circuit breaker wants fallback but no fallback client configured!"
                )
                # Still try primary as last resort
                return self.primary_client, self.primary_model
            logger.debug(
                f"[CircuitBreaker] Routing to fallback: {self.fallback_model}"
            )
            return self.fallback_client, self.fallback_model
        return self.primary_client, self.primary_model

    async def _call_with_fallback(self, call_name: str, **kwargs) -> Any:
        """Execute an LLM call with automatic circuit breaker failover.

        1. Get active client from circuit breaker state.
        2. If primary: try call → on success record_success, on failure record_failure → retry with fallback.
        3. If fallback: call fallback → record result.

        Args:
            call_name: Name for logging (e.g. "classify_email")
            **kwargs: Arguments passed to chat.completions.create

        Returns:
            The API response object.
        """
        client, model = self._get_active_client()
        is_primary = client is self.primary_client
        kwargs["model"] = model

        # If in half-open state and using primary, track the test call
        if is_primary and self.circuit_breaker.state.value == "half_open":
            self.circuit_breaker.record_half_open_attempt()

        try:
            response = await client.chat.completions.create(**kwargs)

            if is_primary:
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_fallback_success()

            return response

        except Exception as primary_error:
            if is_primary:
                self.circuit_breaker.record_failure()
                logger.warning(
                    f"[{call_name}] Primary failed: {primary_error}. "
                    f"Circuit state: {self.circuit_breaker.state.value}"
                )

                # Immediately try fallback if available
                if self.fallback_client:
                    try:
                        logger.info(
                            f"[{call_name}] Retrying with fallback: {self.fallback_model}"
                        )
                        kwargs["model"] = self.fallback_model
                        # Remove logprobs if fallback doesn't support it
                        fallback_kwargs = {
                            k: v for k, v in kwargs.items()
                            if k not in ("logprobs", "top_logprobs")
                        }
                        response = await self.fallback_client.chat.completions.create(
                            **fallback_kwargs
                        )
                        self.circuit_breaker.record_fallback_success()
                        return response
                    except Exception as fallback_error:
                        self.circuit_breaker.record_fallback_failure()
                        logger.error(
                            f"[{call_name}] Both primary and fallback failed! "
                            f"Primary: {primary_error}, Fallback: {fallback_error}"
                        )
                        raise fallback_error
                else:
                    raise primary_error
            else:
                # Already on fallback and it failed
                self.circuit_breaker.record_fallback_failure()
                logger.error(f"[{call_name}] Fallback failed: {primary_error}")
                raise

    @traceable(run_type="llm", name="classify_email")
    async def classify_email(self, subject: str, body: str) -> Dict[str, Any]:
        """Classify email into categories using logprobs for real confidence scoring."""
        try:
            prompt = EMAIL_CLASSIFICATION_PROMPT.format(
                subject=subject,
                email_body=body
            )

            response = await self._call_with_fallback(
                "classify_email",
                messages=[
                    {"role": "system", "content": "You are an email classification expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=20,
                logprobs=True,
                top_logprobs=5,
            )

            category = response.choices[0].message.content.strip().lower()
            self.total_tokens_used += response.usage.total_tokens

            # Extract real confidence from logprobs (fallback may not have logprobs)
            logprobs = getattr(response.choices[0], "logprobs", None)
            confidence_score = self._extract_confidence_from_logprobs(logprobs, category)

            model_used = response.model or "unknown"
            logger.debug(
                f"Email classified as: {category} "
                f"(confidence: {confidence_score:.3f}, model: {model_used})"
            )
            return {
                "category": category,
                "confidence_score": confidence_score,
                "tokens_used": response.usage.total_tokens,
                "model_used": model_used,
            }

        except Exception as e:
            logger.error(f"Email classification failed: {e}")
            return {
                "category": "other",
                "confidence_score": 0.0,
                "tokens_used": 0,
                "error": str(e),
            }

    def _extract_confidence_from_logprobs(self, logprobs_data, category: str) -> float:
        """Extract confidence score from logprobs.

        For multi-token categories, uses geometric mean of token probabilities.
        Falls back to 0.5 if logprobs unavailable (e.g. when using Claude fallback).
        """
        if not logprobs_data or not getattr(logprobs_data, "content", None):
            logger.warning("No logprobs data available, falling back to default confidence")
            return 0.5

        token_probs: List[float] = []

        for token_info in logprobs_data.content:
            prob = math.exp(token_info.logprob)
            token_probs.append(prob)

            if token_info.top_logprobs:
                alternatives = [
                    f"{alt.token}({math.exp(alt.logprob):.3f})"
                    for alt in token_info.top_logprobs[:3]
                ]
                logger.debug(
                    f"Token '{token_info.token}' prob={prob:.3f}, "
                    f"alternatives: {', '.join(alternatives)}"
                )

        if not token_probs:
            return 0.5

        if len(token_probs) == 1:
            confidence = token_probs[0]
        else:
            log_sum = sum(math.log(p) for p in token_probs if p > 0)
            confidence = math.exp(log_sum / len(token_probs))

        return max(0.0, min(1.0, confidence))

    @traceable(run_type="llm", name="assess_priority")
    async def assess_priority(self, body: str) -> Dict[str, Any]:
        """Assess email priority level using logprobs for confidence."""
        try:
            prompt = EMAIL_PRIORITY_PROMPT.format(email_body=body)

            response = await self._call_with_fallback(
                "assess_priority",
                messages=[
                    {"role": "system", "content": "You are an email priority assessment expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=10,
                logprobs=True,
                top_logprobs=5,
            )

            priority = response.choices[0].message.content.strip().lower()
            self.total_tokens_used += response.usage.total_tokens

            logprobs = getattr(response.choices[0], "logprobs", None)
            priority_confidence = self._extract_confidence_from_logprobs(logprobs, priority)

            logger.debug(
                f"Email priority assessed as: {priority} "
                f"(confidence: {priority_confidence:.3f})"
            )
            return {
                "priority": priority,
                "priority_confidence": priority_confidence,
                "tokens_used": response.usage.total_tokens,
            }

        except Exception as e:
            logger.error(f"Priority assessment failed: {e}")
            return {
                "priority": "medium",
                "priority_confidence": 0.0,
                "tokens_used": 0,
                "error": str(e),
            }

    @traceable(run_type="llm", name="analyze_sentiment")
    async def analyze_sentiment(self, subject: str, body: str) -> Dict[str, Any]:
        """Analyze email sentiment, emotion, tone, and churn risk."""
        try:
            prompt = SENTIMENT_ANALYSIS_PROMPT.format(
                subject=subject,
                email_body=body,
            )

            response = await self._call_with_fallback(
                "analyze_sentiment",
                messages=[
                    {"role": "system", "content": "You are a sentiment and emotion analysis expert. Respond with only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=300,
            )

            raw = response.choices[0].message.content.strip()
            self.total_tokens_used += response.usage.total_tokens

            result = json.loads(raw)

            logger.debug(
                f"Sentiment: {result.get('sentiment')}, "
                f"score: {result.get('sentiment_score')}, "
                f"tone: {result.get('tone')}, "
                f"escalation_risk: {result.get('escalation_risk')}, "
                f"churn_risk: {result.get('churn_risk')}"
            )

            return {
                "sentiment": result.get("sentiment", "neutral"),
                "sentiment_score": float(result.get("sentiment_score", 0.0)),
                "tone": result.get("tone", "neutral"),
                "escalation_risk": result.get("escalation_risk", "low"),
                "churn_risk": bool(result.get("churn_risk", False)),
                "churn_indicators": result.get("churn_indicators", []),
                "tokens_used": response.usage.total_tokens,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse sentiment JSON: {e}")
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "tone": "neutral",
                "escalation_risk": "low",
                "churn_risk": False,
                "churn_indicators": [],
                "tokens_used": 0,
                "error": f"JSON parse error: {str(e)}",
            }

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "tone": "neutral",
                "escalation_risk": "low",
                "churn_risk": False,
                "churn_indicators": [],
                "tokens_used": 0,
                "error": str(e),
            }

    @traceable(run_type="llm", name="generate_response")
    async def generate_response(
        self, subject: str, body: str, category: str, priority: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate response to customer email."""
        try:
            context_str = f"\nContext from knowledge base:\n{context}" if context else ""

            prompt = RESPONSE_GENERATION_PROMPT.format(
                subject=subject,
                email_body=body,
                classification=category,
                priority=priority,
                context=context_str,
            )

            response = await self._call_with_fallback(
                "generate_response",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CUSTOMER_SUPPORT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            response_text = response.choices[0].message.content.strip()
            self.total_tokens_used += response.usage.total_tokens
            model_used = response.model or self.primary_model

            logger.debug(f"Response generated ({response.usage.total_tokens} tokens, model: {model_used})")
            return {
                "response_text": response_text,
                "model_used": model_used,
                "tokens_used": response.usage.total_tokens,
                "confidence_score": 0.85,
            }

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response_text": None,
                "model_used": self.primary_model,
                "tokens_used": 0,
                "error": str(e),
            }

    async def health_check(self) -> dict:
        """Check LLM service health including circuit breaker status."""
        result = {
            "service": "llm",
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "fallback_configured": self.fallback_client is not None,
            "total_tokens_used": self.total_tokens_used,
            "circuit_breaker": self.circuit_breaker.get_status(),
        }

        try:
            await self.primary_client.chat.completions.create(
                model=self.primary_model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5,
            )
            result["primary_status"] = "healthy"
        except Exception as e:
            result["primary_status"] = f"unhealthy: {e}"

        if self.fallback_client:
            try:
                await self.fallback_client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=5,
                )
                result["fallback_status"] = "healthy"
            except Exception as e:
                result["fallback_status"] = f"unhealthy: {e}"
        else:
            result["fallback_status"] = "not configured"

        # Overall status
        if result["primary_status"] == "healthy":
            result["status"] = "healthy"
        elif self.fallback_client and result.get("fallback_status") == "healthy":
            result["status"] = "degraded"  # Running on fallback
        else:
            result["status"] = "unhealthy"

        return result
