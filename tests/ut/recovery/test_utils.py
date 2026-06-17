# SPDX-License-Identifier: Apache-2.0

from tests.ut.base import TestBase
from vllm_ascend.recovery.utils import get_engine_recovery_bind_address


class TestUtils(TestBase):

    def test_bind_address_format(self):
        step_addr, report_addr, result_addr = get_engine_recovery_bind_address(0)
        self.assertEqual(step_addr, "ipc:///tmp/engine_recovery_step_xpub_0")
        self.assertEqual(report_addr, "ipc:///tmp/engine_recovery_fault_report_pull_0")
        self.assertEqual(result_addr, "ipc:///tmp/engine_recovery_step_result_pull_0")

    def test_bind_address_different_indices(self):
        addrs_0 = get_engine_recovery_bind_address(0)
        addrs_1 = get_engine_recovery_bind_address(1)
        self.assertNotEqual(addrs_0, addrs_1)
        self.assertEqual(addrs_0[0], "ipc:///tmp/engine_recovery_step_xpub_0")
        self.assertEqual(addrs_1[0], "ipc:///tmp/engine_recovery_step_xpub_1")
