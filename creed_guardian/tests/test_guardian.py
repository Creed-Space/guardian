"""Unit tests for Creed Guardian."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from creed_guardian import Guardian, GuardianResult, Tier


class TestGuardianResult:
    """Tests for GuardianResult dataclass."""

    def test_passed_result(self):
        result = GuardianResult.create_pass("Safe action", "nano", 100.0)
        assert result.verdict == "PASS"
        assert result.allowed is True
        assert result.blocked is False
        assert result.uncertain is False
        assert result.tier == "nano"

    def test_blocked_result(self):
        result = GuardianResult.create_blocked("Unsafe action", "lite", 150.0)
        assert result.verdict == "FAIL"
        assert result.allowed is False
        assert result.blocked is True
        assert result.uncertain is False

    def test_uncertain_result(self):
        result = GuardianResult.create_uncertain("Cannot determine", "standard", 200.0)
        assert result.verdict == "UNCERTAIN"
        assert result.allowed is False
        assert result.blocked is False
        assert result.uncertain is True

    def test_to_dict(self):
        result = GuardianResult.create_pass("Test", "nano", 50.0)
        d = result.to_dict()
        assert d["verdict"] == "PASS"
        assert d["allowed"] is True
        assert d["tier"] == "nano"
        assert d["latency_ms"] == 50.0


class TestTierSelection:
    """Tests for tier selection logic."""

    def test_explicit_tier_nano(self):
        with patch.object(Guardian, "_ensure_initialized", new_callable=AsyncMock):
            guardian = Guardian(tier="1.5b")
            assert guardian.tier == Tier.T1_5B
            assert guardian.model == "qwen2.5:1.5b"

    def test_explicit_tier_lite(self):
        with patch.object(Guardian, "_ensure_initialized", new_callable=AsyncMock):
            guardian = Guardian(tier="7b")
            assert guardian.tier == Tier.T7B
            assert guardian.model == "qwen2.5:7b"

    def test_explicit_tier_standard(self):
        with patch.object(Guardian, "_ensure_initialized", new_callable=AsyncMock):
            guardian = Guardian(tier="14b")
            assert guardian.tier == Tier.T14B
            assert guardian.model == "qwen2.5:14b"

    def test_explicit_tier_pro(self):
        with patch.object(Guardian, "_ensure_initialized", new_callable=AsyncMock):
            guardian = Guardian(tier="32b")
            assert guardian.tier == Tier.T32B
            assert guardian.model == "qwen2.5:32b"

    def test_auto_tier_selection(self):
        """Auto selection should pick a valid tier based on RAM."""
        with patch("psutil.virtual_memory") as mock_mem:
            # Simulate 16GB RAM
            mock_mem.return_value = MagicMock(total=16 * 1024**3)
            guardian = Guardian(tier="auto")
            # Should select Standard (12GB) or lower
            assert guardian.tier in [Tier.T14B, Tier.T7B, Tier.T1_5B]

    def test_auto_tier_low_ram(self):
        """Low RAM should select Nano tier."""
        with patch("psutil.virtual_memory") as mock_mem:
            # Simulate 3GB RAM
            mock_mem.return_value = MagicMock(total=3 * 1024**3)
            guardian = Guardian(tier="auto")
            assert guardian.tier == Tier.T1_5B


class TestGuardianCheck:
    """Tests for Guardian.check() method."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Ollama client."""
        with patch("creed_guardian.guardian.OllamaClient") as MockClient:
            client = AsyncMock()
            client.check_connection = AsyncMock(return_value=True)
            client.is_model_available = AsyncMock(return_value=True)
            MockClient.return_value = client
            yield client

    @pytest.mark.asyncio
    async def test_check_blocks_unsafe_action(self, mock_client):
        """Should block obviously unsafe actions."""
        mock_client.generate = AsyncMock(
            return_value="UNSAFE - This would delete files"
        )

        guardian = Guardian(tier="1.5b")
        result = await guardian.check(
            action="rm -rf /", context="User wants to delete everything"
        )

        assert result.blocked is True
        assert result.verdict == "FAIL"

    @pytest.mark.asyncio
    async def test_check_allows_safe_action(self, mock_client):
        """Should allow safe actions."""
        mock_client.generate = AsyncMock(return_value="SAFE - Reading a file is fine")

        guardian = Guardian(tier="1.5b")
        result = await guardian.check(
            action="Read file /tmp/data.txt", context="User wants to view data"
        )

        assert result.allowed is True
        assert result.verdict == "PASS"

    @pytest.mark.asyncio
    async def test_check_timeout_fails_closed(self, mock_client):
        """Timeout should block when fail_closed=True."""
        mock_client.generate = AsyncMock(side_effect=asyncio.TimeoutError())

        guardian = Guardian(tier="1.5b", fail_closed=True, evaluation_timeout=1.0)
        result = await guardian.check(action="test action")

        assert result.blocked is True
        assert "timeout" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_check_uncertain_fails_closed(self, mock_client):
        """Uncertain verdict should block when fail_closed=True."""
        mock_client.generate = AsyncMock(return_value="I'm not sure about this one")

        guardian = Guardian(tier="1.5b", fail_closed=True)
        result = await guardian.check(action="ambiguous action")

        assert result.blocked is True

    @pytest.mark.asyncio
    async def test_check_uncertain_returns_uncertain(self, mock_client):
        """Uncertain verdict should return uncertain when fail_closed=False."""
        mock_client.generate = AsyncMock(return_value="I'm not sure about this one")

        guardian = Guardian(tier="1.5b", fail_closed=False)
        result = await guardian.check(action="ambiguous action")

        assert result.uncertain is True
        assert result.verdict == "UNCERTAIN"

    @pytest.mark.asyncio
    async def test_check_with_custom_principle(self, mock_client):
        """Should use custom principle when provided."""
        mock_client.generate = AsyncMock(return_value="UNSAFE")

        guardian = Guardian(tier="1.5b")
        await guardian.check(
            action="Send marketing email", principle="Never send unsolicited emails"
        )

        # Verify the generate call included the custom principle
        call_args = mock_client.generate.call_args
        assert "Never send unsolicited emails" in call_args.kwargs["prompt"]


class TestGuardianSyncMethods:
    """Tests for synchronous Guardian methods."""

    def test_check_sync(self):
        """check_sync should work synchronously."""
        with patch("creed_guardian.guardian.OllamaClient") as MockClient:
            client = AsyncMock()
            client.check_connection = AsyncMock(return_value=True)
            client.is_model_available = AsyncMock(return_value=True)
            client.generate = AsyncMock(return_value="SAFE")
            MockClient.return_value = client

            guardian = Guardian(tier="1.5b")
            result = guardian.check_sync(action="test action")

            assert result.allowed is True


class TestGuardianDecorator:
    """Tests for @guardian.protect decorator."""

    @pytest.mark.asyncio
    async def test_protect_allows_safe_function(self):
        """Protected function should execute when action is safe."""
        with patch("creed_guardian.guardian.OllamaClient") as MockClient:
            client = AsyncMock()
            client.check_connection = AsyncMock(return_value=True)
            client.is_model_available = AsyncMock(return_value=True)
            client.generate = AsyncMock(return_value="SAFE")
            MockClient.return_value = client

            guardian = Guardian(tier="1.5b")

            @guardian.protect
            async def safe_operation():
                return "success"

            result = await safe_operation()
            assert result == "success"

    @pytest.mark.asyncio
    async def test_protect_blocks_unsafe_function(self):
        """Protected function should raise PermissionError when blocked."""
        with patch("creed_guardian.guardian.OllamaClient") as MockClient:
            client = AsyncMock()
            client.check_connection = AsyncMock(return_value=True)
            client.is_model_available = AsyncMock(return_value=True)
            client.generate = AsyncMock(return_value="UNSAFE")
            MockClient.return_value = client

            guardian = Guardian(tier="1.5b")

            @guardian.protect
            async def dangerous_operation():
                return "should not reach here"

            with pytest.raises(PermissionError) as exc_info:
                await dangerous_operation()

            assert "Guardian blocked" in str(exc_info.value)


class TestGuardianStatus:
    """Tests for Guardian status methods."""

    def test_get_status(self):
        """get_status should return current configuration."""
        guardian = Guardian(
            tier="7b",
            fail_closed=True,
            escalate_uncertain=False,
        )
        status = guardian.get_status()

        assert status["tier"] == "7b"
        assert status["model"] == "qwen2.5:7b"
        assert status["fail_closed"] is True
        assert status["escalate_uncertain"] is False
        assert status["initialized"] is False


class TestGuardianContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Guardian should work as async context manager."""
        with patch("creed_guardian.guardian.OllamaClient") as MockClient:
            client = AsyncMock()
            client.check_connection = AsyncMock(return_value=True)
            client.is_model_available = AsyncMock(return_value=True)
            client.generate = AsyncMock(return_value="SAFE")
            client.close = AsyncMock()
            MockClient.return_value = client

            async with Guardian(tier="1.5b") as guardian:
                result = await guardian.check(action="test")
                assert result.allowed is True

            # Verify close was called
            client.close.assert_called_once()
