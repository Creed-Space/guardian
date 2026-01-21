"""Custom exceptions for Creed Guardian."""


class GuardianError(Exception):
    """Base exception for all Guardian errors."""

    pass


class ModelUnavailableError(GuardianError):
    """Raised when the required model is not available in Ollama."""

    def __init__(self, model: str, message: str | None = None):
        self.model = model
        # Generic message - details available via model attribute
        msg = message or "Required model is not available"
        super().__init__(msg)


class EvaluationTimeoutError(GuardianError):
    """Raised when evaluation times out."""

    def __init__(self, timeout_seconds: float, message: str | None = None):
        self.timeout_seconds = timeout_seconds
        # Generic message - details available via timeout_seconds attribute
        msg = message or "Safety evaluation timed out"
        super().__init__(msg)


class OllamaConnectionError(GuardianError):
    """Raised when Ollama is not reachable."""

    def __init__(self, url: str, message: str | None = None):
        self.url = url
        # Generic message - details available via url attribute
        msg = message or "Cannot connect to Ollama server"
        super().__init__(msg)


class ConstitutionError(GuardianError):
    """Raised when there's an issue with the constitution."""

    pass


class CloudEscalationError(GuardianError):
    """Raised when cloud escalation fails."""

    pass
