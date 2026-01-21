# Contributing to Creed Guardian

Thank you for your interest in contributing to Creed Guardian!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/guardian.git`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
# Clone the repo
git clone https://github.com/Creed-Space/guardian.git
cd guardian

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install Ollama for testing
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=creed_guardian

# Run specific test file
pytest tests/test_guardian.py -v
```

## Code Style

We use:
- [Black](https://black.readthedocs.io/) for code formatting
- [Ruff](https://docs.astral.sh/ruff/) for linting
- [MyPy](https://mypy.readthedocs.io/) for type checking

```bash
# Format code
black creed_guardian tests

# Lint
ruff check creed_guardian tests

# Type check
mypy creed_guardian
```

## Pull Request Process

1. Ensure tests pass: `pytest tests/ -v`
2. Update documentation if needed
3. Add a clear PR description
4. Link any related issues

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps for bugs
- Check existing issues before creating new ones

## Code of Conduct

Be respectful and constructive. We're building AI safety infrastructure together.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Questions? Open an issue or reach out at hello@creed.space.
