# Creed Guardian

**Local AI safety for any device. Free forever.**

[![PyPI version](https://badge.fury.io/py/creed-guardian.svg)](https://badge.fury.io/py/creed-guardian)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Creed Guardian is a lightweight AI safety layer that evaluates agent actions locally before execution. It uses Ollama-powered models to catch unsafe actions without cloud dependencies, API costs, or data egress.

## Quick Start

```bash
# Install Guardian
pip install creed-guardian

# Install Ollama (required)
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve
```

```python
from creed_guardian import Guardian

guardian = Guardian()  # Auto-selects best model for your hardware

result = guardian.check(
    action="Delete all files in /home",
    context="User said: clean up my downloads"
)

if result.allowed:
    execute_action()
else:
    print(f"Blocked: {result.reason}")
```

## Features

- **Offline-first**: Works without internet
- **Privacy-preserving**: Data never leaves your device
- **Zero marginal cost**: No API fees for local evaluation
- **Fail-safe**: Blocks on uncertainty (configurable)
- **Auto-scaling**: Selects optimal model for your hardware

## Model Tiers

Guardian automatically selects the best model based on available RAM:

| Tier | Model | Size | RAM | Accuracy | Use Case |
|------|-------|------|-----|----------|----------|
| **1.5b** | qwen2.5:1.5b | ~1GB | 2GB | 60% (100% recall) | IoT, embedded |
| **3b** | qwen2.5:3b | ~2GB | 4GB | 72% | Tablets, light laptops |
| **7b** | qwen2.5:7b | ~5GB | 8GB | 82% | Laptops, dev machines |
| **14b** | qwen2.5:14b | ~9GB | 12GB | 91% | Servers, workstations |
| **32b** | qwen2.5:32b | ~19GB | 20GB | 91% | High-security |

All tiers should typically catch **100% of obvious safety violations** (100% recall). Higher tiers reduce false positives.

## Usage

### Basic Check

```python
from creed_guardian import Guardian

guardian = Guardian()

# Check an action
result = guardian.check(
    action="Send email to user@example.com",
    context="User requested password reset"
)

print(result.verdict)    # PASS | FAIL | UNCERTAIN
print(result.reason)     # Human-readable explanation
print(result.latency_ms) # Evaluation time
```

### Synchronous API

```python
# For non-async code
result = guardian.check_sync(action="test action")
```

### Decorator Protection

```python
@guardian.protect
async def delete_files(path: str):
    """This function is protected by Guardian."""
    os.rmdir(path)

# Raises PermissionError if Guardian blocks
await delete_files("/tmp/data")
```

### Custom Principles

```python
result = guardian.check(
    action="Post message to social media",
    context="Marketing campaign",
    principle="Never post without explicit user approval"
)
```

### Explicit Tier Selection

```python
# Force a specific tier
guardian = Guardian(tier="7b")  # Use 7B model

# Available: "auto", "1.5b", "3b", "7b", "14b", "32b"
```

### Async Context Manager

```python
async with Guardian() as guardian:
    result = await guardian.check(action="test")
    # Automatically closes when done
```

## Configuration

```python
Guardian(
    tier="auto",              # Model tier (auto-selects based on RAM)
    ollama_url="http://localhost:11434",  # Ollama server URL
    fail_closed=True,         # Block uncertain cases (default: True)
    auto_download=True,       # Download model if not available
    evaluation_timeout=30.0,  # Timeout in seconds
)
```

## Response Object

```python
@dataclass
class GuardianResult:
    verdict: Literal["PASS", "FAIL", "UNCERTAIN"]
    allowed: bool       # True if PASS
    blocked: bool       # True if FAIL
    uncertain: bool     # True if UNCERTAIN
    reason: str         # Human-readable explanation
    confidence: float   # 0.0-1.0
    source: str         # "pattern" | "local" | "cloud"
    tier: str           # Model tier used
    latency_ms: float   # Evaluation time
```

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) running locally
- 2GB+ RAM (4GB+ recommended)

## Need More?

**Creed Guardian is free forever** for local evaluation.

For cloud features (escalation, fleet management, custom constitutions), check out [Creed Space](https://creed.space/guardian):

| Feature | Guardian (Free) | Creed Space |
|---------|-----------------|-------------|
| Local evaluation | Unlimited | Unlimited |
| All model tiers | Yes | Yes |
| Cloud escalation | - | Yes |
| Custom constitutions | - | Yes |
| Fleet dashboard | - | Yes |
| Analytics | - | Yes |

## Links

- [Documentation](https://creed.space/docs/guardian)
- [API Reference](https://creed.space/docs/guardian/api)
- [Creed Space](https://creed.space)
- [GitHub](https://github.com/Creed-Space/guardian)

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Built by [Creed Space](https://creed.space) - AI safety infrastructure for the real world.*
