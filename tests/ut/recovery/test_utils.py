# SPDX-License-Identifier: Apache-2.0

from vllm_ascend.recovery.utils import get_engine_recovery_bind_address


class TestGetEngineRecoveryBindAddress:
    def test_returns_three_addresses(self):
        result = get_engine_recovery_bind_address(0)
        assert len(result) == 3

    def test_addresses_contain_engine_index(self):
        result = get_engine_recovery_bind_address(3)
        assert "engine_recovery_step_xpub_3" in result[0]
        assert "engine_recovery_fault_report_pull_3" in result[1]
        assert "engine_recovery_step_result_pull_3" in result[2]

    def test_addresses_use_ipc_protocol(self):
        result = get_engine_recovery_bind_address(0)
        for addr in result:
            assert addr.startswith("ipc://")
