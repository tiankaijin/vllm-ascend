# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, Mock, patch

import torch

from tests.ut.base import TestBase


class TestActions(TestBase):

    @staticmethod
    def _make_req(req_id, finished=False):
        req = Mock()
        req.request_id = req_id
        req.is_finished.return_value = finished
        return req

    def _ec_action(self, name):
        from vllm_ascend.recovery.actions import get_engine_core_action
        return get_engine_core_action(name)

    def _w_action(self, name):
        from vllm_ascend.recovery.actions import get_worker_action
        return get_worker_action(name)

    def test_engine_core_action_decorator_registration(self):
        func = self._ec_action("label_dirty_requests")
        self.assertTrue(callable(func))

    def test_worker_action_decorator_registration(self):
        func = self._w_action("recovery_begin")
        self.assertTrue(callable(func))

    def test_get_engine_core_action_unknown_raises(self):
        from vllm_ascend.recovery.actions import get_engine_core_action
        with self.assertRaisesRegex(ValueError, "Unknown engine_core action"):
            get_engine_core_action("nonexistent")

    def test_get_worker_action_unknown_raises(self):
        from vllm_ascend.recovery.actions import get_worker_action
        with self.assertRaisesRegex(ValueError, "Unknown worker action"):
            get_worker_action("nonexistent")

    def test_recovery_begin_sets_flag(self):
        executor = Mock()
        executor.in_recovery = False
        cfg, success = self._w_action("recovery_begin")(executor, {})
        self.assertTrue(executor.in_recovery)
        self.assertTrue(success)

    def test_recovery_finished_clears_flags(self):
        executor = Mock()
        executor.in_recovery = True
        executor.exception_occur = True
        cfg, success = self._w_action("recovery_finished")(executor, {})
        self.assertFalse(executor.in_recovery)
        self.assertFalse(executor.exception_occur)
        self.assertTrue(success)

    def test_label_dirty_requests_collects_running(self):
        executor = Mock()
        executor.dp_rank = 0
        executor.batch_queue = None
        executor.last_scheduler_output = None
        scheduler = Mock()
        scheduler.running = [self._make_req("req1"), self._make_req("req2", finished=True)]
        scheduler.waiting = []
        executor.scheduler = scheduler

        cfg, success = self._ec_action("label_dirty_requests")(executor, {})
        self.assertTrue(success)
        self.assertIn("req1", cfg["dirty_requests_list"])
        self.assertNotIn("req2", cfg["dirty_requests_list"])

    def test_label_dirty_requests_skips_finished(self):
        executor = Mock()
        executor.dp_rank = 0
        executor.batch_queue = None
        executor.last_scheduler_output = None
        scheduler = Mock()
        scheduler.running = [self._make_req("finished_req", finished=True)]
        scheduler.waiting = []
        executor.scheduler = scheduler

        cfg, success = self._ec_action("label_dirty_requests")(executor, {})
        self.assertTrue(success)
        self.assertEqual(cfg["dirty_requests_list"], [])

    def test_label_dirty_requests_collects_scheduled_waiting(self):
        executor = Mock()
        executor.dp_rank = 0
        sched_output = Mock()
        sched_output.num_scheduled_tokens = {"req_w": 5}
        executor.batch_queue = [(Mock(), sched_output, Mock())]
        executor.last_scheduler_output = None
        scheduler = Mock()
        scheduler.running = []
        scheduler.waiting = [self._make_req("req_w")]
        executor.scheduler = scheduler

        cfg, success = self._ec_action("label_dirty_requests")(executor, {})
        self.assertTrue(success)
        self.assertIn("req_w", cfg["dirty_requests_list"])
        self.assertIn("req_w", cfg["waiting_dirty_requests_list"])

    def test_label_dirty_requests_returns_correct_cfg(self):
        executor = Mock()
        executor.dp_rank = 0
        executor.batch_queue = None
        executor.last_scheduler_output = None
        scheduler = Mock()
        scheduler.running = [self._make_req("r1")]
        scheduler.waiting = []
        executor.scheduler = scheduler

        cfg, success = self._ec_action("label_dirty_requests")(executor, {})
        self.assertTrue(success)
        self.assertEqual(cfg["dirty_requests_list"], ["r1"])
        self.assertEqual(cfg["waiting_dirty_requests_list"], [])

    def test_clean_batch_queue_wait_timeout(self):
        executor = Mock()
        executor.dp_rank = 0
        executor.exception_occurred = False
        with patch("vllm_ascend.recovery.actions.time.sleep", return_value=None):
            cfg, success = self._ec_action("clean_batch_queue")(executor, {})
        self.assertFalse(success)

    def test_clean_batch_queue_drains(self):
        executor = Mock()
        executor.dp_rank = 0
        executor.exception_occurred = True
        f1, f2, f3 = Mock(), Mock(), Mock()
        executor.batch_queue = [(f1, Mock(), Mock()), (f2, Mock(), Mock()), (f3, Mock(), Mock())]
        with patch("vllm_ascend.recovery.actions.time.sleep", return_value=None):
            cfg, success = self._ec_action("clean_batch_queue")(executor, {})
        self.assertTrue(success)
        f1.result.assert_called_once()
        f2.result.assert_called_once()
        f3.result.assert_called_once()
        self.assertEqual(len(executor.batch_queue), 0)

    def test_clean_batch_queue_none_queue(self):
        executor = Mock()
        executor.dp_rank = 0
        executor.exception_occurred = True
        executor.batch_queue = None
        cfg, success = self._ec_action("clean_batch_queue")(executor, {})
        self.assertTrue(success)

    def test_clean_batch_queue_future_swallows_exception(self):
        executor = Mock()
        executor.dp_rank = 0
        executor.exception_occurred = True
        future = Mock()
        future.result.side_effect = RuntimeError("future failed")
        executor.batch_queue = [(future, Mock(), Mock())]
        with patch("vllm_ascend.recovery.actions.time.sleep", return_value=None):
            cfg, success = self._ec_action("clean_batch_queue")(executor, {})
        self.assertTrue(success)
        self.assertEqual(len(executor.batch_queue), 0)

    def test_label_dirty_requests_with_last_scheduler_output(self):
        executor = Mock()
        executor.dp_rank = 0
        executor.batch_queue = None
        last_output = Mock()
        last_output.num_scheduled_tokens = {"req_last": 3}
        executor.last_scheduler_output = last_output
        scheduler = Mock()
        scheduler.running = []
        scheduler.waiting = [self._make_req("req_last")]
        executor.scheduler = scheduler

        cfg, success = self._ec_action("label_dirty_requests")(executor, {})
        self.assertTrue(success)
        self.assertIn("req_last", cfg["dirty_requests_list"])
        self.assertIn("req_last", cfg["waiting_dirty_requests_list"])

    def test_collect_request_block_ids(self):
        from vllm_ascend.recovery.actions import _collect_request_block_ids

        block1 = Mock(block_id=10)
        block2 = Mock(block_id=20)
        scheduler = Mock()
        scheduler.kv_cache_manager.coordinator.get_blocks.return_value = [[block1], [block2]]

        result = _collect_request_block_ids(scheduler, "test_req")
        self.assertEqual(result, {10, 20})

    def test_collect_request_block_ids_empty(self):
        from vllm_ascend.recovery.actions import _collect_request_block_ids

        scheduler = Mock()
        scheduler.kv_cache_manager.coordinator.get_blocks.return_value = []
        self.assertEqual(_collect_request_block_ids(scheduler, "test_req"), set())

    def test_stop_device_success(self):
        executor = Mock()
        executor.device.index = 0
        with patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch("vllm_ascend.recovery.actions.torch_npu.npu.stop_device", return_value=0):
            cfg, success = self._w_action("stop_device")(executor, {})
        self.assertTrue(success)

    def test_stop_device_nonzero_result(self):
        executor = Mock()
        executor.device.index = 0
        with patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch("vllm_ascend.recovery.actions.torch_npu.npu.stop_device", return_value=1):
            cfg, success = self._w_action("stop_device")(executor, {})
        self.assertFalse(success)

    def test_stop_device_exception(self):
        executor = Mock()
        executor.device.index = 0
        with patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch("vllm_ascend.recovery.actions.torch_npu.npu.stop_device",
                   side_effect=RuntimeError("device error")):
            cfg, success = self._w_action("stop_device")(executor, {})
        self.assertFalse(success)

    def test_restart_device_success(self):
        executor = Mock()
        executor.device.index = 0
        executor.exception_occur = True
        mock_cfg = Mock()
        mock_cfg.recovery_config.cpu_process_group_timeout = 30
        with patch("vllm_ascend.recovery.actions.time.sleep", return_value=None), \
             patch("vllm_ascend.ascend_config.get_ascend_config", return_value=mock_cfg), \
             patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch("vllm_ascend.recovery.actions.torch_npu.npu.restart_device"):
            cfg, success = self._w_action("restart_device")(executor, {})
        self.assertTrue(success)

    def test_restart_device_exception(self):
        executor = Mock()
        executor.device.index = 0
        executor.exception_occur = True
        mock_patches = [
            patch("vllm_ascend.recovery.actions.torch_npu.npu.restart_device",
                  side_effect=RuntimeError("restart failed")),
        ]
        # Manually apply patches since _restart_device_mocks doesn't work well with context manager chaining
        mock_cfg = Mock()
        mock_cfg.recovery_config.cpu_process_group_timeout = 30
        with patch("vllm_ascend.recovery.actions.time.sleep", return_value=None), \
             patch("vllm_ascend.ascend_config.get_ascend_config", return_value=mock_cfg), \
             patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch("vllm_ascend.recovery.actions.torch_npu.npu.restart_device",
                   side_effect=RuntimeError("restart failed")):
            cfg, success = self._w_action("restart_device")(executor, {})
        self.assertFalse(success)

    def test_restart_device_with_cfg_rebuild_flag(self):
        executor = Mock()
        executor.device.index = 0
        executor.exception_occur = True
        mock_cfg = Mock()
        mock_cfg.recovery_config.cpu_process_group_timeout = 30
        with patch("vllm_ascend.recovery.actions.time.sleep", return_value=None), \
             patch("vllm_ascend.ascend_config.get_ascend_config", return_value=mock_cfg), \
             patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch("vllm_ascend.recovery.actions.torch_npu.npu.restart_device") as mock_restart:
            cfg, success = self._w_action("restart_device")(
                executor, {"restart_device_rebuild_all_resources": True}
            )
        self.assertTrue(success)
        mock_restart.assert_called_once_with(0, rebuild_all_resources=True)

    def test_reinit_process_group_success(self):
        executor = Mock()
        executor.device.index = 0
        with patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch.object(torch.distributed, "reinit_process_group", create=True):
            cfg, success = self._w_action("reinit_process_group")(executor, {})
        self.assertTrue(success)

    def test_reinit_process_group_exception(self):
        executor = Mock()
        executor.device.index = 0
        with patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch.object(torch.distributed, "reinit_process_group", create=True,
                          side_effect=RuntimeError("reinit failed")):
            cfg, success = self._w_action("reinit_process_group")(executor, {})
        self.assertFalse(success)

    def test_reinit_process_group_with_cfg(self):
        executor = Mock()
        executor.device.index = 0
        mock_group = Mock()
        with patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch.object(torch.distributed, "reinit_process_group", create=True) as mock_reinit:
            cfg, success = self._w_action("reinit_process_group")(
                executor, {"group": mock_group, "reinit_process_group_rebuild_link": True},
            )
        self.assertTrue(success)
        mock_reinit.assert_called_once_with(group=mock_group, rebuild_link=True)

    def test_worker_clean_dirty_requests_cache_success(self):
        executor = Mock()
        model_runner = Mock()
        model_runner.requests = MagicMock()
        model_runner.num_prompt_logprobs = MagicMock()
        executor.model_runner = model_runner

        cfg, success = self._w_action("worker_clean_dirty_requests_cache")(
            executor, {"dirty_requests_list": ["req_a", "req_b", "req_c"]}
        )
        self.assertTrue(success)
        model_runner.requests.pop.assert_any_call("req_a", None)
        model_runner.requests.pop.assert_any_call("req_b", None)
        model_runner.num_prompt_logprobs.pop.assert_any_call("req_a", None)
        model_runner.input_batch.remove_request.assert_any_call("req_a")
        model_runner.input_batch.remove_request.assert_any_call("req_b")
        model_runner.input_batch.condense.assert_called_once()
        model_runner.input_batch.refresh_metadata.assert_called_once()

    def test_worker_clean_dirty_requests_cache_exception(self):
        executor = Mock()
        model_runner = Mock()
        model_runner.requests = MagicMock()
        model_runner.requests.pop.side_effect = RuntimeError("cache error")
        executor.model_runner = model_runner

        cfg, success = self._w_action("worker_clean_dirty_requests_cache")(
            executor, {"dirty_requests_list": ["req_a"]}
        )
        self.assertFalse(success)

    def test_worker_rebuild_cpu_group_success(self):
        executor = Mock()
        with patch("vllm.distributed.parallel_state.get_world_group") as mock_world, \
             patch("vllm.distributed.parallel_state.get_dp_group") as mock_dp, \
             patch("vllm.distributed.parallel_state.get_pp_group") as mock_pp:
            cfg, success = self._w_action("worker_rebuild_cpu_group")(executor, {})
        self.assertTrue(success)
        mock_world.return_value.barrier.assert_called_once()
        mock_dp.return_value.reinit_cpu_group.assert_called_once()
        mock_pp.return_value.reinit_cpu_group.assert_called_once()

    def test_worker_rebuild_cpu_group_assertion_swallowed(self):
        executor = Mock()
        with patch("vllm.distributed.parallel_state.get_world_group"), \
             patch("vllm.distributed.parallel_state.get_dp_group") as mock_dp, \
             patch("vllm.distributed.parallel_state.get_pp_group") as mock_pp:
            mock_dp.return_value.reinit_cpu_group.side_effect = AssertionError
            mock_pp.return_value.reinit_cpu_group.side_effect = AssertionError
            cfg, success = self._w_action("worker_rebuild_cpu_group")(executor, {})
        self.assertTrue(success)

    def test_worker_rebuild_cpu_group_exception(self):
        executor = Mock()
        with patch("vllm.distributed.parallel_state.get_world_group",
                   side_effect=RuntimeError("group error")):
            cfg, success = self._w_action("worker_rebuild_cpu_group")(executor, {})
        self.assertFalse(success)

    def test_worker_recapture_graph_success(self):
        executor = Mock()
        executor.device.index = 0
        model_runner = Mock()
        model_runner.compilation_config.cudagraph_capture_sizes = [2, 4, 1]
        executor.model_runner = model_runner
        with patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch("vllm_ascend.compilation.acl_graph.reset_graph_params"), \
             patch("vllm_ascend.compilation.acl_graph.set_graph_params"), \
             patch("vllm_ascend.recovery.actions.ACLGraphWrapper.label_reset_all_graphs") as mock_label_reset:
            cfg, success = self._w_action("worker_recapture_graph")(executor, {})
        self.assertTrue(success)
        mock_label_reset.assert_called_once_with(reset_graph_pool=False)
        model_runner.capture_model.assert_called_once()

    def test_worker_recapture_graph_exception(self):
        executor = Mock()
        executor.device.index = 0
        model_runner = Mock()
        model_runner.compilation_config.cudagraph_capture_sizes = [1]
        model_runner.capture_model.side_effect = RuntimeError("capture failed")
        executor.model_runner = model_runner
        with patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch("vllm_ascend.compilation.acl_graph.reset_graph_params"), \
             patch("vllm_ascend.compilation.acl_graph.set_graph_params"), \
             patch("vllm_ascend.recovery.actions.ACLGraphWrapper.label_reset_all_graphs"):
            cfg, success = self._w_action("worker_recapture_graph")(executor, {})
        self.assertFalse(success)

    def test_recompute_dirty_requests_success(self):
        executor = Mock()
        executor.dp_rank = 0
        scheduler = Mock()
        req_running = self._make_req("req_r")
        req_waiting = self._make_req("req_w")
        scheduler.requests = {"req_r": req_running, "req_w": req_waiting}
        scheduler.running = {req_running}
        executor.scheduler = scheduler
        cfg = {"dirty_requests_list": ["req_r", "req_w"], "waiting_dirty_requests_list": ["req_w"]}

        with patch("vllm_ascend.recovery.actions._collect_request_block_ids",
                   return_value={5, 10}) as mock_collect, \
             patch("vllm_ascend.recovery.actions._rebuild_request"):
            cfg, success = self._ec_action("recompute_dirty_requests")(executor, cfg)
        self.assertTrue(success)
        self.assertEqual(mock_collect.call_count, 2)

    def test_recompute_dirty_requests_request_not_found(self):
        executor = Mock()
        executor.dp_rank = 0
        scheduler = Mock()
        scheduler.requests = {"req_r": self._make_req("req_r")}
        scheduler.running = set()
        executor.scheduler = scheduler
        cfg = {"dirty_requests_list": ["req_r"], "waiting_dirty_requests_list": []}

        with patch("vllm_ascend.recovery.actions._collect_request_block_ids",
                   return_value=set()):
            cfg, success = self._ec_action("recompute_dirty_requests")(executor, cfg)
        self.assertFalse(success)

    def test_restart_device_timeout(self):
        executor = Mock()
        executor.device.index = 0
        executor.exception_occur = False
        mock_cfg = Mock()
        mock_cfg.recovery_config.cpu_process_group_timeout = 1
        with patch("vllm_ascend.recovery.actions.time.sleep", return_value=None), \
             patch("vllm_ascend.ascend_config.get_ascend_config", return_value=mock_cfg), \
             patch("vllm_ascend.recovery.actions.NPUPlatform.set_device"), \
             patch("vllm_ascend.recovery.actions.torch_npu.npu.restart_device"):
            cfg, success = self._w_action("restart_device")(executor, {})
        self.assertTrue(success)
