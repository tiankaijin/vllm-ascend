# SPDX-License-Identifier: Apache-2.0

import threading
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.recovery.engine_core_recovery_handler import RecoveryHandler
from vllm_ascend.recovery.types import (
    RecoveryComplete,
    RecoveryPlan,
    RecoveryStep,
    StepTarget,
)


def _make_handler():
    engine_core = MagicMock()
    vllm_config = MagicMock()
    vllm_config.parallel_config.local_world_size = 2
    vllm_config.parallel_config.data_parallel_rank = 0
    vllm_config.needs_dp_coordinator = False
    handler = RecoveryHandler(engine_core, vllm_config)
    return handler


class TestRecoveryHandlerInit:
    def test_initial_state(self):
        handler = _make_handler()
        assert handler.is_recovering is False
        assert handler._recovery_success is False
        assert handler._worker_count == 2
        assert handler._engine_index == 0


class TestWaitForRecovery:
    def test_wait_for_recovery_returns_on_set(self):
        handler = _make_handler()
        handler._recovery_done_event.set()
        result = handler.wait_for_recovery(timeout=1.0)
        assert result is True

    def test_wait_for_recovery_timeout(self):
        handler = _make_handler()
        handler._recovery_done_event.clear()
        result = handler.wait_for_recovery(timeout=0.1)
        assert result is False


class TestGetRecoverySuccess:
    def test_returns_success_flag(self):
        handler = _make_handler()
        handler._recovery_success = True
        assert handler.get_recovery_success() is True

    def test_returns_false_by_default(self):
        handler = _make_handler()
        assert handler.get_recovery_success() is False


class TestBeginRecovery:
    def test_sets_recovering_state(self):
        handler = _make_handler()
        handler._recovery_done_event.set()
        handler._begin_recovery()
        assert handler.is_recovering is True
        assert handler._recovery_success is False
        assert not handler._recovery_done_event.is_set()


class TestFinishRecovery:
    @patch.object(RecoveryHandler, "_dispatch_worker_step", return_value=({}, True))
    def test_finish_recovery_success(self, mock_dispatch):
        handler = _make_handler()
        handler._finish_recovery(success=True)
        assert handler.is_recovering is False
        assert handler._recovery_success is True
        assert handler._recovery_done_event.is_set()

    @patch.object(RecoveryHandler, "_dispatch_worker_step", return_value=({}, True))
    def test_finish_recovery_failure(self, mock_dispatch):
        handler = _make_handler()
        handler._finish_recovery(success=False)
        assert handler.is_recovering is False
        assert handler._recovery_success is False


class TestHandleRecoveryComplete:
    def test_success_updates_engine_core(self):
        handler = _make_handler()
        handler._engine_core = MagicMock()
        msg = RecoveryComplete(plan_name="test", success=True, current_wave=5)
        handler._handle_recovery_complete(msg)
        assert handler._engine_core.current_wave == 5
        assert handler._engine_core.step_counter == 0
        assert handler._engine_core.engines_running is True

    @patch.object(RecoveryHandler, "_finish_recovery")
    def test_failure_calls_finish_with_false(self, mock_finish):
        handler = _make_handler()
        handler._engine_core = MagicMock()
        msg = RecoveryComplete(plan_name="test", success=False, current_wave=3)
        handler._handle_recovery_complete(msg)
        mock_finish.assert_called_once_with(success=False)


class TestExecuteRecovery:
    @patch.object(RecoveryHandler, "_execute_engine_core_step", return_value=({}, True))
    @patch.object(RecoveryHandler, "_dispatch_worker_step", return_value=({}, True))
    @patch.object(RecoveryHandler, "_finalize_recovery")
    def test_all_steps_succeed(self, mock_finalize, mock_dispatch, mock_engine_step):
        handler = _make_handler()
        plan = RecoveryPlan(
            name="test_plan",
            timeout_s=60,
            steps=[
                RecoveryStep(name="step1", target=StepTarget.ENGINE_CORE.value),
                RecoveryStep(name="step2", target=StepTarget.WORKER.value),
            ],
        )
        handler._execute_recovery(plan)
        assert mock_finalize.called

    @patch.object(RecoveryHandler, "_execute_engine_core_step", return_value=({}, False))
    @patch.object(RecoveryHandler, "_finalize_recovery")
    def test_step_failure_aborts_plan(self, mock_finalize, mock_engine_step):
        handler = _make_handler()
        plan = RecoveryPlan(
            name="test_plan",
            timeout_s=60,
            steps=[RecoveryStep(name="step1", target=StepTarget.ENGINE_CORE.value)],
        )
        handler._execute_recovery(plan)
        result = mock_finalize.call_args[0][1]
        assert result.success is False
