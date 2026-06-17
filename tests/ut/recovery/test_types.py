# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import msgspec.msgpack

from tests.ut.base import TestBase
from vllm_ascend.recovery.types import (
    ExceptionInfo,
    FaultReport,
    RecoveryAction,
    RecoveryComplete,
    RecoveryPlan,
    RecoveryPlanResult,
    RecoveryStep,
    StepResult,
    StepTarget,
    WorkerStepDispatch,
)


class TestTypes(TestBase):

    def test_step_target_values(self):
        self.assertEqual(StepTarget.ENGINE_CORE.value, "engine_core")
        self.assertEqual(StepTarget.WORKER.value, "worker")

    def _roundtrip(self, obj):
        encoded = msgspec.msgpack.encode(obj)
        decoded = msgspec.msgpack.decode(encoded, type=type(obj))
        return decoded

    def test_exception_info_create_and_roundtrip(self):
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="test error")
        decoded = self._roundtrip(exc)
        self.assertEqual(decoded.exception_type, "RuntimeError")
        self.assertEqual(decoded.exception_msg, "test error")

    def test_recovery_action_roundtrip(self):
        action = RecoveryAction(name="test_action")
        decoded = self._roundtrip(action)
        self.assertEqual(decoded.name, "test_action")

    def test_recovery_step_roundtrip(self):
        step = RecoveryStep(
            name="test_step",
            target="worker",
            timeout_s=60,
            actions=[RecoveryAction(name="act1"), RecoveryAction(name="act2")],
        )
        decoded = self._roundtrip(step)
        self.assertEqual(decoded.name, "test_step")
        self.assertEqual(decoded.target, "worker")
        self.assertEqual(decoded.timeout_s, 60)
        self.assertEqual(len(decoded.actions), 2)
        self.assertEqual(decoded.actions[0].name, "act1")
        self.assertEqual(decoded.actions[1].name, "act2")

    def test_recovery_plan_roundtrip(self):
        plan = RecoveryPlan(
            name="test_plan",
            timeout_s=300,
            steps=[RecoveryStep(name="s1", target="worker")],
            cfg={"key": "value"},
        )
        decoded = self._roundtrip(plan)
        self.assertEqual(decoded.name, "test_plan")
        self.assertEqual(decoded.timeout_s, 300)
        self.assertEqual(len(decoded.steps), 1)
        self.assertEqual(decoded.steps[0].name, "s1")
        self.assertEqual(decoded.cfg["key"], "value")

    def test_step_result_roundtrip(self):
        result = StepResult(
            step_name="step1",
            success=True,
            worker_rank=0,
            cfg={"a": 1},
        )
        decoded = self._roundtrip(result)
        self.assertEqual(decoded.step_name, "step1")
        self.assertTrue(decoded.success)
        self.assertEqual(decoded.worker_rank, 0)
        self.assertEqual(decoded.cfg["a"], 1)

    def test_fault_report_roundtrip(self):
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="fail")
        plan = RecoveryPlan(name="recovery_plan", timeout_s=300)
        report = FaultReport(worker_rank=1, engine_index=0, exp=exc, plan=plan)

        decoded = self._roundtrip(report)
        self.assertEqual(decoded.worker_rank, 1)
        self.assertEqual(decoded.engine_index, 0)
        self.assertEqual(decoded.exp.exception_type, "RuntimeError")
        self.assertEqual(decoded.exp.exception_msg, "fail")
        self.assertEqual(decoded.plan.name, "recovery_plan")

    def test_worker_step_dispatch_roundtrip(self):
        step = RecoveryStep(name="worker_step", target="worker")
        dispatch = WorkerStepDispatch(step=step, cfg={"k": "v"})

        decoded = self._roundtrip(dispatch)
        self.assertEqual(decoded.step.name, "worker_step")
        self.assertEqual(decoded.cfg["k"], "v")

    def test_recovery_plan_result_roundtrip(self):
        result = RecoveryPlanResult(
            plan_name="plan1",
            engine_index=0,
            success=True,
            step_results=[
                StepResult(step_name="s1", success=True, worker_rank=0),
                StepResult(step_name="s2", success=True, worker_rank=1),
            ],
        )
        decoded = self._roundtrip(result)
        self.assertEqual(decoded.plan_name, "plan1")
        self.assertEqual(decoded.engine_index, 0)
        self.assertTrue(decoded.success)
        self.assertEqual(len(decoded.step_results), 2)
        self.assertEqual(decoded.step_results[0].step_name, "s1")
        self.assertEqual(decoded.step_results[1].step_name, "s2")

    def test_recovery_complete_roundtrip(self):
        complete = RecoveryComplete(plan_name="plan1", success=True, current_wave=3)
        decoded = self._roundtrip(complete)
        self.assertEqual(decoded.plan_name, "plan1")
        self.assertTrue(decoded.success)
        self.assertEqual(decoded.current_wave, 3)

    def test_step_execute_empty_actions(self):
        step = RecoveryStep(name="empty_step", target="worker", actions=[])
        cfg, success = step.execute(Mock(), {})
        self.assertEqual(cfg, {})
        self.assertTrue(success)

    @patch("vllm_ascend.recovery.types.get_engine_core_action")
    def test_step_execute_success_chain(self, mock_get_action):
        mock_executer = Mock()
        base_cfg = {"count": 0}

        def act1(executer, cfg):
            cfg["count"] += 1
            return cfg, True

        def act2(executer, cfg):
            cfg["count"] += 1
            return cfg, True

        def act3(executer, cfg):
            cfg["count"] += 1
            return cfg, True

        mock_get_action.side_effect = [act1, act2, act3]

        step = RecoveryStep(
            name="chain_step",
            target="engine_core",
            actions=[
                RecoveryAction(name="a1"),
                RecoveryAction(name="a2"),
                RecoveryAction(name="a3"),
            ],
        )
        cfg, success = step.execute(mock_executer, base_cfg)
        self.assertTrue(success)
        self.assertEqual(cfg["count"], 3)

    @patch("vllm_ascend.recovery.types.get_worker_action")
    def test_step_execute_failure_stops(self, mock_get_action):
        mock_executer = Mock()
        base_cfg = {"called": []}

        def act1(executer, cfg):
            cfg["called"].append("act1")
            return cfg, True

        def act2(executer, cfg):
            cfg["called"].append("act2")
            return cfg, False

        def act3(executer, cfg):
            cfg["called"].append("act3")
            return cfg, True

        mock_get_action.side_effect = [act1, act2, act3]

        step = RecoveryStep(
            name="fail_chain_step",
            target="worker",
            actions=[
                RecoveryAction(name="a1"),
                RecoveryAction(name="a2"),
                RecoveryAction(name="a3"),
            ],
        )
        cfg, success = step.execute(mock_executer, base_cfg)
        self.assertFalse(success)
        self.assertEqual(cfg["called"], ["act1", "act2"])

    @patch("vllm_ascend.recovery.types.get_engine_core_action")
    @patch("vllm_ascend.recovery.types.get_worker_action")
    def test_action_execute_dispatches_to_correct_registry(
        self, mock_get_worker_action, mock_get_engine_core_action
    ):
        mock_executer = Mock()

        ec_action = Mock(return_value=({}, True))
        w_action = Mock(return_value=({}, True))
        mock_get_engine_core_action.return_value = ec_action
        mock_get_worker_action.return_value = w_action

        action_ec = RecoveryAction(name="test_action")
        action_ec.execute(mock_executer, {}, target="engine_core")
        mock_get_engine_core_action.assert_called_once_with("test_action")
        ec_action.assert_called_once()

        action_w = RecoveryAction(name="test_action")
        action_w.execute(mock_executer, {}, target="worker")
        mock_get_worker_action.assert_called_once_with("test_action")
        w_action.assert_called_once()
