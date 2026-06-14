# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import msgspec.msgpack
import pytest

from vllm.v1.engine.coordinator import DPCoordinator, DPCoordinatorProc
from vllm_ascend.recovery.types import (
    ExceptionInfo,
    FaultReport,
    NetworkCheck,
    RecoveryComplete,
    RecoveryPlan,
    RecoveryPlanResult,
    RecoveryStep,
)


class TestPatchedDPCoordinatorInit:
    @patch("vllm_ascend.patch.platform.patch_recovery_coordinator.get_engine_client_zmq_addr")
    @patch("vllm_ascend.patch.platform.patch_recovery_coordinator.get_mp_context")
    def test_init_sets_recovery_addresses(self, mock_mp, mock_zmq_addr):
        mock_zmq_addr.return_value = "tcp://127.0.0.1:12345"
        mock_ctx = MagicMock()
        mock_mp.return_value = mock_ctx

        parallel_config = MagicMock()
        parallel_config.data_parallel_size = 2
        parallel_config.data_parallel_master_ip = "127.0.0.1"
        parallel_config.local_engines_only = False
        parallel_config.data_parallel_size_local = 1
        parallel_config.enable_elastic_ep = False
        parallel_config.enable_wave_coordination = True

        coord = DPCoordinator.__new__(DPCoordinator)
        coord.__init__(parallel_config)

        assert hasattr(coord, "recovery_pub_address")
        assert hasattr(coord, "recovery_pull_address")


class TestRecoveryMsgPrefix:
    def test_prefix_value(self):
        from vllm_ascend.patch.platform.patch_recovery_coordinator import _RECOVERY_MSG_PREFIX
        assert _RECOVERY_MSG_PREFIX == b"\x00REC"


class TestRecoveryMessageSerialization:
    def test_fault_report_roundtrip(self):
        plan = RecoveryPlan(
            name="network_recover_plan",
            timeout_s=300,
            steps=[RecoveryStep(name="step1", target="worker")],
        )
        report = FaultReport(
            worker_rank=0,
            engine_index=1,
            exp=ExceptionInfo(exception_type="RuntimeError", exception_msg="507057"),
            plan=plan,
        )
        encoded = msgspec.msgpack.encode(("faultreport", report))
        decoded = msgspec.msgpack.decode(encoded)
        assert decoded[0] == "faultreport"
        assert decoded[1].worker_rank == 0

    def test_recovery_complete_roundtrip(self):
        rc = RecoveryComplete(plan_name="plan_a", success=True, current_wave=5)
        encoded = msgspec.msgpack.encode(("recoverycomplete", rc))
        decoded = msgspec.msgpack.decode(encoded)
        assert decoded[0] == "recoverycomplete"
        assert decoded[1].success is True
        assert decoded[1].current_wave == 5

    def test_network_check_roundtrip(self):
        nc = NetworkCheck(engine_index=2)
        encoded = msgspec.msgpack.encode(("networkcheck", nc))
        decoded = msgspec.msgpack.decode(encoded)
        assert decoded[0] == "networkcheck"
        assert decoded[1].engine_index == 2
