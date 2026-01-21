# Changelog

All notable changes to Creed Guardian will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-01-21

### Security

- **H1: Prompt injection mitigation** - Added input sanitization that detects suspicious patterns ("ignore previous", "disregard instructions", etc.) and logs warnings. Inputs are now length-limited to 10KB. Multiple newlines are collapsed to prevent section injection.

- **H2: SSRF protection** - Added URL validation for `ollama_url` parameter. Cloud metadata endpoints (AWS 169.254.169.254, GCP metadata.google.internal, Azure metadata.azure.com) and private IP ranges (10.x, 172.16-31.x, 192.168.x) are now blocked. Only localhost and public URLs are allowed.

- **M1: Removed API key exposure** - `get_status()` no longer returns `has_api_key` or `escalate_uncertain` fields to prevent information leakage.

- **M2: Reduced logging verbosity** - Changed operational logs from INFO to DEBUG level to reduce log exposure.

- **M3: TLS verification option** - Added `verify_ssl` parameter (default: True) to support custom CA bundles or disable verification for self-signed certificates.

- **M4: Sanitized exception messages** - Error messages are now generic and don't expose internal details. Technical details available via exception attributes.

- **L1: Specific exception handling** - Cloud escalation now catches specific exceptions instead of bare `except`.

- **L2: Fixed async patterns** - `check_sync()` now properly detects running event loops and uses `asyncio.run()`.

- **L4: Pinned dependencies** - httpx and psutil now have upper version bounds.

### Added

- Security section in README documenting prompt injection limitations, network security, and vulnerability reporting.
- Comprehensive security tests for SSRF protection, input sanitization, and TLS configuration.

## [0.1.0] - 2026-01-20

### Added

- Initial release of Creed Guardian
- Local AI safety evaluation using Ollama-powered models
- Auto-tier selection based on available RAM (1.5B to 32B models)
- Fail-closed mode (default) for uncertain verdicts
- Synchronous and async APIs
- `@guardian.protect` decorator for protecting functions
- Async context manager support
