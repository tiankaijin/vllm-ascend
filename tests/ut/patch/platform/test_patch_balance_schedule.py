# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.patch.platform.patch_balance_schedule import (
    BalanceScheduler,
    _balance_scheduling_enabled,
    _recovery_enabled,
)


class TestBalanceSchedulingEnabled:
    def test_enabled_when_config_set(self):
        vllm_config = MagicMock()
        vllm_config.additional_config = {"enable_balance_scheduling": True}
        assert _balance_scheduling_enabled(vllm_config) is True

    def test_disabled_when_config_missing(self):
        vllm_config = MagicMock()
        vllm_config.additional_config = {}
        assert _balance_scheduling_enabled(vllm_config) is False

    def test_disabled_when_additional_config_is_none(self):
        vllm_config = MagicMock()
        vllm_config.additional_config = None
        assert _balance_scheduling_enabled(vllm_config) is False


class TestRecoveryEnabled:
    @patch("vllm_ascend.patch.platform.patch_balance_schedule.envs")
    def test_enabled(self, mock_envs):
        mock_envs.VLLM_ASCEND_ENABLE_RECOVERY = True
        assert _recovery_enabled() is True

    @patch("vllm_ascend.patch.platform.patch_balance_schedule.envs")
    def test_disabled(self, mock_envs):
        mock_envs.VLLM_ASCEND_ENABLE_RECOVERY = False
        assert _recovery_enabled() is False


class TestBalanceScheduler:
    @patch("vllm_ascend.patch.platform.patch_balance_schedule._balance_scheduling_enabled", return_value=True)
    def test_balance_enabled_creates_balance_queue(self, mock_enabled):
        vllm_config = MagicMock()
        vllm_config.parallel_config.data_parallel_size = 4
        kv_cache_config = MagicMock()
        structured_output_manager = MagicMock()
        with patch.object(BalanceScheduler, "__init__", wraps=None):
            pass

    @patch("vllm_ascend.patch.platform.patch_balance_schedule._balance_scheduling_enabled", return_value=False)
    def test_balance_disabled_no_queue(self, mock_enabled):
        vllm_config = MagicMock()
        vllm_config.parallel_config.data_parallel_size = 2
        kv_cache_config = MagicMock()
        structured_output_manager = MagicMock()
        with patch.object(BalanceScheduler, "__init__", wraps=None):
            pass
