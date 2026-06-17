# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.ut.base import TestBase


class TestBalanceHelpers(TestBase):

    def test_balance_scheduling_disabled_no_additional_config(self):
        from vllm_ascend.patch.platform.patch_balance_schedule import (
            _balance_scheduling_enabled,
        )
        vc = MagicMock()
        del vc.additional_config
        self.assertFalse(_balance_scheduling_enabled(vc))

    def test_balance_scheduling_disabled_empty_dict(self):
        from vllm_ascend.patch.platform.patch_balance_schedule import (
            _balance_scheduling_enabled,
        )
        vc = SimpleNamespace(additional_config={})
        self.assertFalse(_balance_scheduling_enabled(vc))

    def test_balance_scheduling_enabled_true(self):
        from vllm_ascend.patch.platform.patch_balance_schedule import (
            _balance_scheduling_enabled,
        )
        vc = SimpleNamespace(
            additional_config={"enable_balance_scheduling": True}
        )
        self.assertTrue(_balance_scheduling_enabled(vc))

    def test_balance_scheduling_enabled_false(self):
        from vllm_ascend.patch.platform.patch_balance_schedule import (
            _balance_scheduling_enabled,
        )
        vc = SimpleNamespace(
            additional_config={"enable_balance_scheduling": False}
        )
        self.assertFalse(_balance_scheduling_enabled(vc))

    def test_recovery_enabled_reads_env(self):
        with patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_RECOVERY": "1"}):
            import vllm_ascend.envs as envs
            envs.VLLM_ASCEND_ENABLE_RECOVERY = True
            from vllm_ascend.patch.platform.patch_balance_schedule import (
                _recovery_enabled,
            )
            self.assertTrue(_recovery_enabled())


class TestBalanceScheduler(TestBase):

    def setUp(self):
        super().setUp()
        from vllm_ascend.patch.platform.patch_balance_schedule import (
            BalanceScheduler,
        )
        self.BalanceScheduler = BalanceScheduler

    def _make_mock_scheduler(self, vllm_config=None):
        if vllm_config is None:
            vllm_config = MagicMock()
            vllm_config.parallel_config.data_parallel_size = 2
            vllm_config.additional_config = {}
        kv_cache_config = MagicMock()
        kv_cache_config.kv_cache_groups = [MagicMock()]
        smo = MagicMock()
        bs = self.BalanceScheduler.__new__(self.BalanceScheduler)
        bs.vllm_config = vllm_config
        bs.max_num_running_reqs = 4
        return bs

    def test_init_balance_disabled(self):
        from vllm_ascend.patch.platform.patch_balance_schedule import (
            _balance_scheduling_enabled,
        )
        vc = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=2),
            additional_config={},
        )
        self.assertFalse(_balance_scheduling_enabled(vc))

    def test_init_balance_enabled(self):
        from vllm_ascend.patch.platform.patch_balance_schedule import (
            _balance_scheduling_enabled,
        )
        vc = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=2),
            additional_config={"enable_balance_scheduling": True},
        )
        self.assertTrue(_balance_scheduling_enabled(vc))

    def test_init_balance_queue_length_matches_dp_size(self):
        vc = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=3),
            additional_config={"enable_balance_scheduling": True},
        )
        kvc = MagicMock()
        kvc.kv_cache_groups = [MagicMock()]
        smo = MagicMock()

        def fake_super_init(self_, *a, **kw):
            self_.vllm_config = vc

        with patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".Scheduler.__init__",
            new=fake_super_init,
        ), patch(
            "torch.tensor",
            return_value=MagicMock(),
        ):
            bs = self.BalanceScheduler(vc, kvc, smo, 16)
            self.assertTrue(bs._balance_enabled)
            self.assertEqual(len(bs.balance_queue), 3)

    def test_balance_gather_disabled_noop(self):
        bs = self._make_mock_scheduler()
        bs._balance_enabled = False
        with patch("torch.distributed.all_gather") as mock_ag:
            bs.balance_gather(MagicMock())
            mock_ag.assert_not_called()

    def test_balance_gather_enabled_calls_all_gather(self):
        bs = self._make_mock_scheduler()
        bs._balance_enabled = True
        bs.balance_queue = [
            MagicMock(), MagicMock(),
        ]
        bs.running = [MagicMock(), MagicMock(), MagicMock()]
        dp_group = MagicMock()
        with patch("torch.distributed.all_gather") as mock_ag:
            bs.balance_gather(dp_group)
            mock_ag.assert_called_once()

    def test_schedule_delegates_when_disabled(self):
        bs = self._make_mock_scheduler()
        bs._balance_enabled = False
        expected = object()
        with patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".Scheduler.schedule",
            return_value=expected,
        ):
            result = bs.schedule()
            self.assertIs(result, expected)


class TestDPEngineCoreProcWithRecovery(TestBase):

    def setUp(self):
        super().setUp()
        from vllm_ascend.patch.platform.patch_balance_schedule import (
            DPEngineCoreProcWithRecovery,
        )
        self.RecoveryEC = DPEngineCoreProcWithRecovery

    def _mock_ec(self, **overrides):
        ec = self.RecoveryEC.__new__(self.RecoveryEC)
        ec.dp_rank = 0
        ec.current_wave = 5
        ec._recovery_handler = MagicMock()
        ec.eep_scaling_state = None
        ec.exception_occurred = False
        ec.engines_running = False
        ec.has_coordinator = True
        ec.step_counter = 0
        ec.dp_group = MagicMock()
        return ec

    def test_init_creates_recovery_handler(self):
        vc = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
        with patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".get_engine_recovery_bind_address",
            return_value=("xpub", "rpull", "rpush"),
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".RecoveryHandler",
        ) as mock_rh, patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".DPEngineCoreProc.__init__",
            return_value=None,
        ):
            fake_args = MagicMock()
            ec = self.RecoveryEC(vllm_config=vc, **{"_mock": fake_args})
            mock_rh.assert_called_once()

    def test_init_calls_setup_recover_sockets(self):
        vc = SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
        with patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".get_engine_recovery_bind_address",
            return_value=("xpub", "rpull", "rpush"),
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".RecoveryHandler",
        ) as mock_rh, patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".DPEngineCoreProc.__init__",
            return_value=None,
        ):
            fake_args = MagicMock()
            ec = self.RecoveryEC(vllm_config=vc, **{"_mock": fake_args})
            handler = mock_rh.return_value
            handler.setup_recover_sockets.assert_called_once_with(
                "xpub", "rpull", "rpush",
            )
            handler.start.assert_called_once()
            handler.wait_for_worker_subscriptions.assert_called_once()

    def test_wait_for_recovery_success(self):
        ec = self._mock_ec()
        ec._recovery_handler.get_recovery_success.return_value = True
        ec._wait_for_recovery()

    def test_wait_for_recovery_failure(self):
        ec = self._mock_ec()
        ec._recovery_handler.get_recovery_success.return_value = False
        with self.assertRaises(RuntimeError):
            ec._wait_for_recovery()

    def test_wait_for_recovery_on_exception_signal_received(self):
        ec = self._mock_ec()
        ec._recovery_handler.is_recovering = True
        result = ec._wait_for_recovery_on_exception()
        self.assertTrue(result)

    def test_wait_for_recovery_on_exception_timeout(self):
        ec = self._mock_ec()
        ec._recovery_handler.is_recovering = False
        start = 0
        with patch("time.time", side_effect=[0, 121]):
            result = ec._wait_for_recovery_on_exception()
            self.assertFalse(result)

    def test_run_busy_loop_recovery_active(self):
        ec = self._mock_ec()
        ec._recovery_handler.is_recovering = True
        calls = []

        def handle_shutdown():
            calls.append("handle_shutdown")
            if len(calls) >= 4:
                raise SystemExit()
            return True

        ec._handle_shutdown = handle_shutdown
        ec._wait_for_recovery = lambda: calls.append("wait_recovery")
        ec._process_input_queue = _noop
        ec._process_engine_step = lambda: (None, True)
        ec._maybe_publish_request_counts = _noop
        ec._has_global_unfinished_reqs = lambda x: False
        ec.execute_dummy_batch = _noop
        ec.scheduler = MagicMock(_balance_enabled=False)
        ec.output_queue = MagicMock()

        with self.assertRaises(SystemExit):
            ec.run_busy_loop()
        self.assertIn("wait_recovery", calls)

    def test_run_busy_loop_exception_recovery_success(self):
        ec = self._mock_ec()
        ec._recovery_handler.is_recovering = False
        calls = []

        def handle_shutdown():
            calls.append("shutdown")
            if len(calls) >= 4:
                raise SystemExit()
            return True

        def process_engine_step():
            calls.append("step")
            if "step" in calls:
                raise ValueError("test error")

        ec._handle_shutdown = handle_shutdown
        ec._process_input_queue = _noop
        ec._process_engine_step = process_engine_step
        ec._wait_for_recovery_on_exception = lambda: True
        ec._maybe_publish_request_counts = _noop
        ec._has_global_unfinished_reqs = lambda x: False
        ec.execute_dummy_batch = _noop
        ec.scheduler = MagicMock(_balance_enabled=False)
        ec.output_queue = MagicMock()

        with self.assertRaises(SystemExit):
            ec.run_busy_loop()
        self.assertIn("step", calls)


def _noop(*a, **kw):
    pass


class TestRunEngineCore(TestBase):

    def test_both_disabled_delegates_to_original(self):
        with patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            "._recovery_enabled", return_value=False,
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            "._balance_scheduling_enabled", return_value=False,
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            "._ORIGINAL_RUN_ENGINE_CORE",
        ) as mock_orig:
            from vllm_ascend.patch.platform.patch_balance_schedule import (
                run_engine_core,
            )
            mock_orig.return_value = "ok"
            result = run_engine_core()
            mock_orig.assert_called_once()
            self.assertEqual(result, "ok")

    def test_recovery_true_creates_recovery_ec(self):
        vc = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                data_parallel_rank=1,
                data_parallel_rank_local=0,
            ),
        )
        with patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            "._recovery_enabled", return_value=True,
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            "._balance_scheduling_enabled", return_value=False,
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".set_process_title",
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".decorate_logs",
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".DPEngineCoreProcWithRecovery",
        ) as mock_rec_ec, patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".maybe_register_config_serialize_by_value",
        ):
            from vllm_ascend.patch.platform.patch_balance_schedule import (
                run_engine_core,
            )
            try:
                run_engine_core(vllm_config=vc, dp_rank=1, local_dp_rank=0)
            except SystemExit:
                pass
            mock_rec_ec.assert_called_once()

    def test_recovery_false_balance_true_creates_balance_ec(self):
        vc = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=2,
                data_parallel_rank=1,
                data_parallel_rank_local=0,
            ),
            additional_config={"enable_balance_scheduling": True},
        )
        with patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            "._recovery_enabled", return_value=False,
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            "._balance_scheduling_enabled", return_value=True,
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".set_process_title",
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".decorate_logs",
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".BalanceDPEngineCoreProc",
        ) as mock_bal_ec, patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".maybe_register_config_serialize_by_value",
        ):
            from vllm_ascend.patch.platform.patch_balance_schedule import (
                run_engine_core,
            )
            try:
                run_engine_core(vllm_config=vc, dp_rank=1, local_dp_rank=0)
            except SystemExit:
                pass
            mock_bal_ec.assert_called_once()

    def test_single_dp_rank_uses_engine_core_proc(self):
        vc = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                data_parallel_rank=0,
                data_parallel_rank_local=0,
            ),
        )
        with patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            "._recovery_enabled", return_value=True,
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            "._balance_scheduling_enabled", return_value=True,
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".set_process_title",
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".decorate_logs",
        ), patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".EngineCoreProc",
        ) as mock_ecp, patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".maybe_register_config_serialize_by_value",
        ):
            from vllm_ascend.patch.platform.patch_balance_schedule import (
                run_engine_core,
            )
            try:
                run_engine_core(vllm_config=vc)
            except SystemExit:
                pass
            mock_ecp.assert_called_once()

class TestBalanceDPEngineCoreProc(TestBase):

    def test_run_busy_loop_calls_balance_gather(self):
        from vllm_ascend.patch.platform.patch_balance_schedule import (
            BalanceDPEngineCoreProc,
        )

        ec = BalanceDPEngineCoreProc.__new__(BalanceDPEngineCoreProc)
        ec.scheduler = MagicMock()
        ec.scheduler._balance_enabled = True
        ec.dp_group = MagicMock()
        ec.engines_running = False
        ec.current_wave = 0
        ec.step_counter = 0
        ec.dp_rank = 0
        ec.has_coordinator = True
        ec.eep_scaling_state = None
        ec.output_queue = MagicMock()

        calls = []

        def process_input_queue():
            calls.append("piq")
            if len(calls) >= 3:
                raise SystemExit()

        ec._process_input_queue = process_input_queue
        ec._process_engine_step = lambda: (None, False)
        ec._maybe_publish_request_counts = _noop
        ec._has_global_unfinished_reqs = lambda x: False
        ec.execute_dummy_batch = _noop

        with self.assertRaises(SystemExit):
            ec.run_busy_loop()
        ec.scheduler.balance_gather.assert_called_with(ec.dp_group)


class TestBalanceSchedulerSchedule(TestBase):

    def test_schedule_balance_enabled_passthrough(self):
        from vllm_ascend.patch.platform.patch_balance_schedule import (
            BalanceScheduler,
        )
        bs = BalanceScheduler.__new__(BalanceScheduler)
        bs._balance_enabled = False
        expected = object()
        with patch(
            "vllm_ascend.patch.platform.patch_balance_schedule"
            ".Scheduler.schedule", return_value=expected,
        ):
            result = bs.schedule()
            self.assertIs(result, expected)
