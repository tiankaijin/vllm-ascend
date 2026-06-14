# SPDX-License-Identifier: Apache-2.0

import msgspec
import msgspec.msgpack
import pytest

from vllm_ascend.recovery.types import (
    ExceptionInfo,
    FaultReport,
    NetworkCheck,
    RecoveryAction,
    RecoveryComplete,
    RecoveryPlan,
    RecoveryPlanResult,
    RecoveryStep,
    StepResult,
    StepTarget,
    WorkerStepDispatch,
)


class TestExceptionInfo:
    def test_encode_decode_roundtrip(self):
        info = ExceptionInfo(exception_type="RuntimeError", exception_msg="NPU error 507057")
        encoded = msgspec.msgpack.encode(info)
        decoded = msgspec.msgpack.decode(encoded, type=ExceptionInfo)
        assert decoded.exception_type == "RuntimeError"
        assert decoded.exception_msg == "NPU error 507057"

    def test_fields(self):
        info = ExceptionInfo(exception_type="ValueError", exception_msg="bad value")
        assert info.exception_type == "ValueError"
        assert info.exception_msg == "bad value"


class TestRecoveryStep:
    def test_default_values(self):
        step = RecoveryStep(name="test_step", target="worker")
        assert step.name == "test_step"
        assert step.target == "worker"
        assert step.timeout_s == 5
        assert step.actions == []

    def test_with_actions(self):
        action = RecoveryAction(name="stop_device")
        step = RecoveryStep(name="recovery", target="worker", actions=[action])
        assert len(step.actions) == 1
        assert step.actions[0].name == "stop_device"


class TestRecoveryPlan:
    def test_default_values(self):
        plan = RecoveryPlan(name="test_plan", timeout_s=60)
        assert plan.name == "test_plan"
        assert plan.timeout_s == 60
        assert plan.steps == []
        assert plan.cfg == {}

    def test_encode_decode_roundtrip(self):
        plan = RecoveryPlan(
            name="network_recover",
            timeout_s=300,
            steps=[RecoveryStep(name="step1", target="worker")],
            cfg={"key": "value"},
        )
        encoded = msgspec.msgpack.encode(plan)
        decoded = msgspec.msgpack.decode(encoded, type=RecoveryPlan)
        assert decoded.name == "network_recover"
        assert decoded.timeout_s == 300
        assert len(decoded.steps) == 1


class TestFaultReport:
    def test_encode_decode_roundtrip(self):
        report = FaultReport(
            worker_rank=0,
            engine_index=1,
            exp=ExceptionInfo(exception_type="RuntimeError", exception_msg="link down"),
            plan=RecoveryPlan(name="plan_a", timeout_s=60),
        )
        encoded = msgspec.msgpack.encode(report)
        decoded = msgspec.msgpack.decode(encoded, type=FaultReport)
        assert decoded.worker_rank == 0
        assert decoded.engine_index == 1
        assert decoded.exp.exception_type == "RuntimeError"
        assert decoded.plan.name == "plan_a"


class TestRecoveryPlanResult:
    def test_encode_decode_roundtrip(self):
        result = RecoveryPlanResult(
            plan_name="network_recover",
            engine_index=0,
            success=True,
            step_results=[StepResult(step_name="step1", success=True, worker_rank=0)],
        )
        encoded = msgspec.msgpack.encode(result)
        decoded = msgspec.msgpack.decode(encoded, type=RecoveryPlanResult)
        assert decoded.plan_name == "network_recover"
        assert decoded.success is True
        assert len(decoded.step_results) == 1


class TestRecoveryComplete:
    def test_encode_decode_roundtrip(self):
        rc = RecoveryComplete(plan_name="plan_a", success=True, current_wave=3)
        encoded = msgspec.msgpack.encode(rc)
        decoded = msgspec.msgpack.decode(encoded, type=RecoveryComplete)
        assert decoded.plan_name == "plan_a"
        assert decoded.success is True
        assert decoded.current_wave == 3


class TestNetworkCheck:
    def test_encode_decode_roundtrip(self):
        nc = NetworkCheck(engine_index=2)
        encoded = msgspec.msgpack.encode(nc)
        decoded = msgspec.msgpack.decode(encoded, type=NetworkCheck)
        assert decoded.engine_index == 2


class TestWorkerStepDispatch:
    def test_encode_decode_roundtrip(self):
        dispatch = WorkerStepDispatch(
            step=RecoveryStep(name="step1", target="worker"),
            cfg={"key": "val"},
        )
        encoded = msgspec.msgpack.encode(dispatch)
        decoded = msgspec.msgpack.decode(encoded, type=WorkerStepDispatch)
        assert decoded.step.name == "step1"
        assert decoded.cfg == {"key": "val"}


class TestStepTarget:
    def test_values(self):
        assert StepTarget.ENGINE_CORE.value == "engine_core"
        assert StepTarget.WORKER.value == "worker"
