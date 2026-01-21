"""Type definitions for Creed Guardian."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Tier(Enum):
    """Model tiers for Creed Guardian."""

    T1_5B = "1.5b"
    T3B = "3b"
    T7B = "7b"
    T14B = "14b"
    T32B = "32b"
    AUTO = "auto"


TIER_MODELS: dict[Tier, str] = {
    Tier.T1_5B: "qwen2.5:1.5b",
    Tier.T3B: "qwen2.5:3b",
    Tier.T7B: "qwen2.5:7b",
    Tier.T14B: "qwen2.5:14b",
    Tier.T32B: "qwen2.5:32b",
}

TIER_RAM_REQUIREMENTS: dict[Tier, float] = {
    Tier.T1_5B: 2.0,
    Tier.T3B: 4.0,
    Tier.T7B: 8.0,
    Tier.T14B: 12.0,
    Tier.T32B: 20.0,
}

TIER_ACCURACY: dict[Tier, dict[str, float]] = {
    Tier.T1_5B: {"overall": 0.60, "recall": 1.00, "precision": 0.20},
    Tier.T3B: {"overall": 0.72, "recall": 1.00, "precision": 0.45},
    Tier.T7B: {"overall": 0.82, "recall": 1.00, "precision": 0.65},
    Tier.T14B: {"overall": 0.91, "recall": 1.00, "precision": 0.85},
    Tier.T32B: {"overall": 0.91, "recall": 1.00, "precision": 0.91},
}


@dataclass
class GuardianResult:
    """Result from Guardian evaluation."""

    verdict: Literal["PASS", "FAIL", "UNCERTAIN"]
    allowed: bool
    blocked: bool
    uncertain: bool
    reason: str
    confidence: float
    source: Literal["pattern", "local", "cloud"]
    tier: str
    latency_ms: float
    escalated: bool = False

    @classmethod
    def create_pass(
        cls, reason: str, tier: str, latency_ms: float, source: str = "local"
    ) -> "GuardianResult":
        """Create a PASS result."""
        return cls(
            verdict="PASS",
            allowed=True,
            blocked=False,
            uncertain=False,
            reason=reason,
            confidence=0.8,
            source=source,  # type: ignore[arg-type]
            tier=tier,
            latency_ms=latency_ms,
        )

    @classmethod
    def create_blocked(
        cls, reason: str, tier: str, latency_ms: float, source: str = "local"
    ) -> "GuardianResult":
        """Create a FAIL/blocked result."""
        return cls(
            verdict="FAIL",
            allowed=False,
            blocked=True,
            uncertain=False,
            reason=reason,
            confidence=0.9,
            source=source,  # type: ignore[arg-type]
            tier=tier,
            latency_ms=latency_ms,
        )

    @classmethod
    def create_uncertain(
        cls, reason: str, tier: str, latency_ms: float, source: str = "local"
    ) -> "GuardianResult":
        """Create an UNCERTAIN result."""
        return cls(
            verdict="UNCERTAIN",
            allowed=False,
            blocked=False,
            uncertain=True,
            reason=reason,
            confidence=0.5,
            source=source,  # type: ignore[arg-type]
            tier=tier,
            latency_ms=latency_ms,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "verdict": self.verdict,
            "allowed": self.allowed,
            "blocked": self.blocked,
            "uncertain": self.uncertain,
            "reason": self.reason,
            "confidence": self.confidence,
            "source": self.source,
            "tier": self.tier,
            "latency_ms": self.latency_ms,
            "escalated": self.escalated,
        }
