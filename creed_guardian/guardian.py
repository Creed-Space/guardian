"""
Creed Guardian - Local AI Safety for Any Device.

Free forever. No API costs. No data leaves your device.
"""

import asyncio
import functools
import logging
import time
from typing import Callable, TypeVar

import psutil

from creed_guardian.client import OllamaClient
from creed_guardian.exceptions import (
    EvaluationTimeoutError,
    GuardianError,
    ModelUnavailableError,
    OllamaConnectionError,
)
from creed_guardian.types import (
    TIER_MODELS,
    TIER_RAM_REQUIREMENTS,
    GuardianResult,
    Tier,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

# Default safety principle
DEFAULT_PRINCIPLE = (
    "Never perform actions that could harm users, violate privacy, "
    "enable illegal activity, or cause damage to systems or data."
)


class Guardian:
    """
    Creed Guardian - Local AI Safety for Any Device.

    Free forever. No API costs. No data leaves your device.

    Example:
        guardian = Guardian()  # Auto-selects best model for your hardware

        result = guardian.check(
            action="Delete all files in /home",
            context="User said: clean up my downloads"
        )

        if result.allowed:
            execute_action()
        else:
            print(f"Blocked: {result.reason}")
    """

    def __init__(
        self,
        tier: str = "auto",
        api_key: str | None = None,
        ollama_url: str = "http://localhost:11434",
        constitution: str | None = None,
        escalate_uncertain: bool = False,
        fail_closed: bool = True,
        auto_download: bool = True,
        evaluation_timeout: float = 30.0,
    ):
        """
        Initialize Creed Guardian.

        Args:
            tier: Model tier ("auto", "1.5b", "3b", "7b", "14b", "32b")
            api_key: Optional Creed Space API key for cloud features
            ollama_url: Ollama server URL
            constitution: Constitution ID or YAML path
            escalate_uncertain: Send uncertain cases to cloud (requires api_key)
            fail_closed: Block uncertain cases when offline (default: True)
            auto_download: Automatically download model if not available
            evaluation_timeout: Timeout in seconds for evaluation
        """
        self.tier = self._resolve_tier(tier)
        self.model = TIER_MODELS[self.tier]
        self.api_key = api_key
        self.ollama_url = ollama_url
        self.constitution = constitution
        self.escalate_uncertain = escalate_uncertain and api_key is not None
        self.fail_closed = fail_closed
        self.auto_download = auto_download
        self.evaluation_timeout = evaluation_timeout

        self._client: OllamaClient | None = None
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

        logger.info(f"Guardian initialized: tier={self.tier.value}, model={self.model}")

    def _resolve_tier(self, tier: str) -> Tier:
        """Resolve tier, auto-selecting if needed based on available RAM."""
        if tier == "auto":
            available_ram = psutil.virtual_memory().total / (1024**3)  # Convert to GB
            for t in [Tier.T32B, Tier.T14B, Tier.T7B, Tier.T3B, Tier.T1_5B]:
                if available_ram >= TIER_RAM_REQUIREMENTS[t]:
                    logger.info(
                        f"Auto-selected tier {t.value} for {available_ram:.1f}GB RAM"
                    )
                    return t
            return Tier.T1_5B
        return Tier(tier)

    async def _ensure_initialized(self) -> None:
        """Ensure model is downloaded and ready."""
        if self._initialized:
            return

        async with self._initialization_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            self._client = OllamaClient(self.ollama_url)

            # Check if Ollama is running
            if not await self._client.check_connection():
                raise OllamaConnectionError(self.ollama_url)

            # Check if model is available
            if not await self._client.is_model_available(self.model):
                if self.auto_download:
                    logger.info(f"Model {self.model} not found, downloading...")
                    await self._client.pull_model(self.model)
                else:
                    raise ModelUnavailableError(self.model)

            self._initialized = True

    def _build_prompt(self, action: str, context: str | None, principle: str) -> str:
        """Build evaluation prompt optimized for small models."""
        prompt = f"""Is this action safe to execute?

PRINCIPLE: {principle}

ACTION: {action}
"""
        if context:
            prompt += f"\nCONTEXT: {context}"

        prompt += "\n\nAnswer ONLY: SAFE or UNSAFE"
        return prompt

    async def check(
        self,
        action: str,
        context: str | None = None,
        principle: str | None = None,
    ) -> GuardianResult:
        """
        Check if an action is safe to execute.

        Args:
            action: The action to evaluate (what the AI wants to do)
            context: Optional context (user request, conversation history)
            principle: Optional specific principle to check against

        Returns:
            GuardianResult with verdict and explanation

        Raises:
            GuardianError: If evaluation fails and cannot recover
        """
        await self._ensure_initialized()

        start = time.perf_counter()

        # Use default principle if not specified
        if principle is None:
            principle = DEFAULT_PRINCIPLE

        prompt = self._build_prompt(action, context, principle)

        try:
            if self._client is None:
                raise GuardianError("Client not initialized")

            response = await asyncio.wait_for(
                self._client.generate(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=50,
                    timeout=self.evaluation_timeout,
                ),
                timeout=self.evaluation_timeout + 5,  # Buffer for network
            )

            latency = (time.perf_counter() - start) * 1000
            text = response.upper().strip()

            # Parse verdict
            if any(kw in text for kw in ["FAIL", "BLOCK", "UNSAFE", "DANGEROUS"]):
                return GuardianResult.create_blocked(
                    reason="Action violates safety principle",
                    tier=self.tier.value,
                    latency_ms=latency,
                )
            elif any(kw in text for kw in ["PASS", "SAFE", "ALLOW", "OK"]):
                return GuardianResult.create_pass(
                    reason="Action appears safe",
                    tier=self.tier.value,
                    latency_ms=latency,
                )
            else:
                # Uncertain - either escalate or fail closed
                return await self._handle_uncertain(action, context, principle, latency)

        except asyncio.TimeoutError as e:
            latency = (time.perf_counter() - start) * 1000
            if self.fail_closed:
                return GuardianResult.create_blocked(
                    reason="Evaluation timeout (fail-closed)",
                    tier=self.tier.value,
                    latency_ms=latency,
                )
            raise EvaluationTimeoutError(self.evaluation_timeout) from e

        except (ModelUnavailableError, OllamaConnectionError):
            latency = (time.perf_counter() - start) * 1000
            if self.fail_closed:
                return GuardianResult.create_blocked(
                    reason="Evaluation service unavailable (fail-closed)",
                    tier=self.tier.value,
                    latency_ms=latency,
                )
            raise

    async def _handle_uncertain(
        self,
        action: str,
        context: str | None,
        principle: str,
        local_latency: float,
    ) -> GuardianResult:
        """Handle uncertain verdict."""
        if self.escalate_uncertain and self.api_key:
            # Try cloud escalation
            try:
                return await self._escalate_to_cloud(
                    action, context, principle, local_latency
                )
            except Exception as e:
                logger.warning(f"Cloud escalation failed: {e}")

        if self.fail_closed:
            return GuardianResult.create_blocked(
                reason="Uncertain verdict (fail-closed mode)",
                tier=self.tier.value,
                latency_ms=local_latency,
            )

        return GuardianResult.create_uncertain(
            reason="Unable to determine safety",
            tier=self.tier.value,
            latency_ms=local_latency,
        )

    async def _escalate_to_cloud(
        self,
        action: str,
        context: str | None,
        principle: str,
        local_latency: float,
    ) -> GuardianResult:
        """Escalate uncertain case to Creed Space cloud.

        TODO: Implement cloud escalation via Creed Space API.
        """
        # Placeholder for cloud escalation
        logger.info("Cloud escalation not yet implemented")
        return GuardianResult.create_uncertain(
            reason="Cloud escalation not yet implemented",
            tier=self.tier.value,
            latency_ms=local_latency,
            source="local",
        )

    def check_sync(
        self,
        action: str,
        context: str | None = None,
        principle: str | None = None,
    ) -> GuardianResult:
        """Synchronous version of check().

        Note: This blocks the current thread. For async applications,
        use the async check() method instead.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.check(action, context, principle))

    def protect(self, func: F) -> F:
        """Decorator to protect a function with Guardian evaluation.

        Example:
            @guardian.protect
            async def delete_files(path: str):
                ...

        If the action is blocked, raises PermissionError.
        """

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            action = f"Call {func.__name__}({args}, {kwargs})"
            result = await self.check(action)

            if not result.allowed:
                raise PermissionError(f"Guardian blocked: {result.reason}")

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            action = f"Call {func.__name__}({args}, {kwargs})"
            result = self.check_sync(action)

            if not result.allowed:
                raise PermissionError(f"Guardian blocked: {result.reason}")

            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    async def close(self) -> None:
        """Close the Guardian and release resources."""
        if self._client:
            await self._client.close()
            self._client = None
        self._initialized = False

    async def __aenter__(self) -> "Guardian":
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.close()

    def get_status(self) -> dict:
        """Get current Guardian status."""
        return {
            "initialized": self._initialized,
            "tier": self.tier.value,
            "model": self.model,
            "ollama_url": self.ollama_url,
            "fail_closed": self.fail_closed,
            "escalate_uncertain": self.escalate_uncertain,
            "has_api_key": self.api_key is not None,
        }
