"""Circuit Breaker pattern for LLM service failover.

States:
  CLOSED   → Primary is healthy, all traffic goes to primary.
  OPEN     → Primary is down, all traffic goes to fallback.
  HALF_OPEN → Cooldown expired, testing primary with limited calls.

Transitions:
  CLOSED → OPEN:      After `failure_threshold` consecutive failures.
  OPEN → HALF_OPEN:   After `cooldown_seconds` have elapsed.
  HALF_OPEN → CLOSED: If the test call succeeds.
  HALF_OPEN → OPEN:   If the test call fails (resets cooldown timer).
"""

import time
from enum import Enum
from typing import Optional

from loguru import logger


class CircuitState(str, Enum):
    CLOSED = "closed"       # Primary healthy
    OPEN = "open"           # Primary down, using fallback
    HALF_OPEN = "half_open" # Testing primary recovery


class CircuitBreaker:
    """Circuit breaker for managing primary/fallback LLM routing."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        cooldown_seconds: int = 60,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        # Metrics
        self.total_primary_calls = 0
        self.total_fallback_calls = 0
        self.total_primary_failures = 0
        self.total_fallback_failures = 0
        self.total_recoveries = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, checking if OPEN should transition to HALF_OPEN."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.cooldown_seconds:
                logger.info(
                    f"[CircuitBreaker:{self.name}] Cooldown expired "
                    f"({elapsed:.0f}s >= {self.cooldown_seconds}s). "
                    f"Transitioning OPEN → HALF_OPEN"
                )
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    @property
    def should_use_fallback(self) -> bool:
        """Return True if traffic should go to fallback."""
        current = self.state  # triggers OPEN → HALF_OPEN check
        if current == CircuitState.CLOSED:
            return False
        if current == CircuitState.OPEN:
            return True
        # HALF_OPEN: route limited calls to primary, rest to fallback
        return self._half_open_calls >= self.half_open_max_calls

    def record_success(self):
        """Record a successful call to primary."""
        self.total_primary_calls += 1

        if self._state == CircuitState.HALF_OPEN:
            # Primary recovered
            logger.info(
                f"[CircuitBreaker:{self.name}] Primary recovered! "
                f"HALF_OPEN → CLOSED"
            )
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0
            self.total_recoveries += 1
        elif self._state == CircuitState.CLOSED:
            # Reset consecutive failure count on success
            self._failure_count = 0

    def record_failure(self):
        """Record a failed call to primary."""
        self.total_primary_failures += 1
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            # Test call failed, go back to OPEN
            logger.warning(
                f"[CircuitBreaker:{self.name}] Test call failed. "
                f"HALF_OPEN → OPEN (will retry after {self.cooldown_seconds}s)"
            )
            self._state = CircuitState.OPEN
            self._half_open_calls = 0

        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.failure_threshold:
                logger.error(
                    f"[CircuitBreaker:{self.name}] {self._failure_count} consecutive "
                    f"failures. CLOSED → OPEN (switching to fallback)"
                )
                self._state = CircuitState.OPEN

    def record_fallback_success(self):
        """Record a successful call to fallback."""
        self.total_fallback_calls += 1

    def record_fallback_failure(self):
        """Record a failed call to fallback (both providers down)."""
        self.total_fallback_calls += 1
        self.total_fallback_failures += 1

    def record_half_open_attempt(self):
        """Track a test call in half-open state."""
        self._half_open_calls += 1

    def get_status(self) -> dict:
        """Get circuit breaker status for health/metrics endpoints."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "seconds_until_retry": max(
                0,
                self.cooldown_seconds - (time.time() - self._last_failure_time)
            ) if self._state == CircuitState.OPEN and self._last_failure_time else 0,
            "metrics": {
                "total_primary_calls": self.total_primary_calls,
                "total_fallback_calls": self.total_fallback_calls,
                "total_primary_failures": self.total_primary_failures,
                "total_fallback_failures": self.total_fallback_failures,
                "total_recoveries": self.total_recoveries,
            },
        }
