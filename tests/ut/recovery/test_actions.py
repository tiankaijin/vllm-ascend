# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from vllm_ascend.recovery.actions import (
    _ENGINE_CORE_ACTIONS,
    _WORKER_ACTIONS,
    engine_core_action,
    get_engine_core_action,
    get_worker_action,
    worker_action,
)


class TestActionRegistry:
    def test_get_engine_core_action_existing(self):
        func = get_engine_core_action("label_dirty_requests")
        assert callable(func)

    def test_get_engine_core_action_missing_raises(self):
        with pytest.raises(ValueError, match="Unknown engine_core action"):
            get_engine_core_action("nonexistent_action")

    def test_get_worker_action_existing(self):
        func = get_worker_action("stop_device")
        assert callable(func)

    def test_get_worker_action_missing_raises(self):
        with pytest.raises(ValueError, match="Unknown worker action"):
            get_worker_action("nonexistent_action")

    def test_engine_core_action_decorator_registers(self):
        assert "label_dirty_requests" in _ENGINE_CORE_ACTIONS
        assert "clean_batch_queue" in _ENGINE_CORE_ACTIONS
        assert "recompute_dirty_requests" in _ENGINE_CORE_ACTIONS

    def test_worker_action_decorator_registers(self):
        assert "stop_device" in _WORKER_ACTIONS
        assert "restart_device" in _WORKER_ACTIONS
        assert "reinit_process_group" in _WORKER_ACTIONS
        assert "recovery_begin" in _WORKER_ACTIONS
        assert "recovery_finished" in _WORKER_ACTIONS


class TestRecoveryBeginAction:
    def test_sets_in_recovery_flag(self):
        func = get_worker_action("recovery_begin")
        executor = MagicMock()
        cfg = {}
        result_cfg, success = func(executor, cfg)
        assert success is True
        assert executor.in_recovery is True


class TestRecoveryFinishedAction:
    def test_clears_flags(self):
        func = get_worker_action("recovery_finished")
        executor = MagicMock()
        cfg = {}
        result_cfg, success = func(executor, cfg)
        assert success is True
        assert executor.in_recovery is False
        assert executor.exception_occur is False
