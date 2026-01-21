"""Ollama client wrapper for Creed Guardian."""

import asyncio
import logging
from typing import Any

import httpx

from creed_guardian.exceptions import (
    ModelUnavailableError,
    OllamaConnectionError,
)

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for Ollama API.

    Handles model checking, downloading, and inference.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize the Ollama client.

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.base_url = base_url.rstrip("/")
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def check_connection(self) -> bool:
        """Check if Ollama server is reachable.

        Returns:
            True if Ollama is reachable, False otherwise.
        """
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    async def list_models(self) -> list[str]:
        """List available models in Ollama.

        Returns:
            List of model names.

        Raises:
            OllamaConnectionError: If Ollama is not reachable.
        """
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/api/tags", timeout=10.0)
            if resp.status_code != 200:
                raise OllamaConnectionError(
                    self.base_url, f"Unexpected status: {resp.status_code}"
                )
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.ConnectError as e:
            raise OllamaConnectionError(self.base_url) from e

    async def is_model_available(self, model: str) -> bool:
        """Check if a specific model is available.

        Args:
            model: Model name (e.g., "qwen2.5:1.5b")

        Returns:
            True if model is available.
        """
        try:
            models = await self.list_models()
            # Check exact match or base name match
            base_name = model.split(":")[0]
            return model in models or any(m.startswith(base_name) for m in models)
        except OllamaConnectionError:
            return False

    async def pull_model(
        self,
        model: str,
        timeout: float = 600.0,
        progress_callback: Any | None = None,
    ) -> bool:
        """Download a model from Ollama registry.

        Args:
            model: Model name to pull
            timeout: Timeout in seconds (default: 10 minutes)
            progress_callback: Optional callback for progress updates

        Returns:
            True if successful.

        Raises:
            ModelUnavailableError: If model cannot be downloaded.
        """
        logger.info(f"Downloading model {model}...")
        try:
            client = await self._get_client()
            async with client.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": model},
                timeout=timeout,
            ) as resp:
                if resp.status_code != 200:
                    raise ModelUnavailableError(
                        model, f"Failed to pull: HTTP {resp.status_code}"
                    )

                async for line in resp.aiter_lines():
                    if progress_callback and line:
                        import json

                        try:
                            data = json.loads(line)
                            progress_callback(data)
                        except json.JSONDecodeError:
                            pass

            logger.info(f"Model {model} downloaded successfully")
            return True
        except httpx.TimeoutException as e:
            raise ModelUnavailableError(
                model, f"Download timed out after {timeout}s"
            ) from e
        except httpx.ConnectError as e:
            raise OllamaConnectionError(self.base_url) from e

    async def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 50,
        timeout: float = 30.0,
    ) -> str:
        """Generate text using a model.

        Args:
            model: Model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            timeout: Timeout in seconds

        Returns:
            Generated text.

        Raises:
            ModelUnavailableError: If model is not available.
            EvaluationTimeoutError: If generation times out.
        """
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                    },
                },
                timeout=timeout,
            )

            if resp.status_code == 404:
                raise ModelUnavailableError(model)
            if resp.status_code != 200:
                raise OllamaConnectionError(
                    self.base_url, f"Generation failed: HTTP {resp.status_code}"
                )

            data = resp.json()
            return data.get("response", "")

        except httpx.TimeoutException as e:
            from creed_guardian.exceptions import EvaluationTimeoutError

            raise EvaluationTimeoutError(timeout) from e
        except httpx.ConnectError as e:
            raise OllamaConnectionError(self.base_url) from e

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()


# Synchronous wrapper for convenience
class OllamaClientSync:
    """Synchronous wrapper for OllamaClient."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._async_client = OllamaClient(base_url)

    def _run(self, coro: Any) -> Any:
        """Run coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    def check_connection(self) -> bool:
        return self._run(self._async_client.check_connection())

    def list_models(self) -> list[str]:
        return self._run(self._async_client.list_models())

    def is_model_available(self, model: str) -> bool:
        return self._run(self._async_client.is_model_available(model))

    def pull_model(self, model: str, timeout: float = 600.0) -> bool:
        return self._run(self._async_client.pull_model(model, timeout))

    def generate(
        self, model: str, prompt: str, max_tokens: int = 50, timeout: float = 30.0
    ) -> str:
        return self._run(
            self._async_client.generate(model, prompt, max_tokens, timeout)
        )

    def close(self) -> None:
        self._run(self._async_client.close())
