"""Custom exceptions for Creed Guardian."""


class GuardianError(Exception):
    """Base exception for all Guardian errors."""

    pass


class ModelUnavailableError(GuardianError):
    """Raised when the required model is not available in Ollama."""

    def __init__(self, model: str, message: str | None = None):
        self.model = model
        msg = message or f"Model '{model}' is not available. Run: ollama pull {model}"
        super().__init__(msg)


class EvaluationTimeoutError(GuardianError):
    """Raised when evaluation times out."""

    def __init__(self, timeout_seconds: float, message: str | None = None):
        self.timeout_seconds = timeout_seconds
        msg = message or f"Evaluation timed out after {timeout_seconds}s"
        super().__init__(msg)


class OllamaConnectionError(GuardianError):
    """Raised when Ollama is not reachable."""

    def __init__(self, url: str, message: str | None = None):
        self.url = url
        msg = message or f"Cannot connect to Ollama at {url}. Is Ollama running?"
        super().__init__(msg)


class ConstitutionError(GuardianError):
    """Raised when there's an issue with the constitution."""

    pass


class CloudEscalationError(GuardianError):
    """Raised when cloud escalation fails."""

    pass
