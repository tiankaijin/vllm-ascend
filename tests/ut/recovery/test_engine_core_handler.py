# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import msgspec.msgpack
import zmq

from tests.ut.base import TestBase
from vllm_ascend.recovery.types import (
    ExceptionInfo,
    FaultReport,
    RecoveryComplete,
    RecoveryPlan,
    RecoveryPlanResult,
    RecoveryStep,
)


class TestEngineCoreHandler(TestBase):

    def _create_handler(self, **mock_kwargs):
        from vllm_ascend.recovery.engine_core_recovery_handler import RecoveryHandler

        engine_core = Mock()
        vllm_config = Mock()
        vllm_config.parallel_config.local_world_size = mock_kwargs.get("local_world_size", 1)
        vllm_config.parallel_config.data_parallel_rank = mock_kwargs.get("dp_rank", 0)
        vllm_config.needs_dp_coordinator = mock_kwargs.get("needs_dp", False)

        with patch("vllm_ascend.recovery.engine_core_recovery_handler.zmq.Context"):
            handler = RecoveryHandler(engine_core, vllm_config)
        return handler, engine_core, vllm_config

    def test_init_worker_count_and_index(self):
        handler, _, _ = self._create_handler(local_world_size=4, dp_rank=2)
        self.assertEqual(handler._worker_count, 4)
        self.assertEqual(handler._engine_index, 2)

    def test_begin_recovery_state(self):
        handler, _, _ = self._create_handler()
        handler.is_recovering = False
        handler._recovery_done_event.set()
        handler._begin_recovery()
        self.assertTrue(handler.is_recovering)
        self.assertFalse(handler._recovery_success)
        self.assertFalse(handler._recovery_done_event.is_set())

    def test_finish_recovery_success_state(self):
        handler, _, _ = self._create_handler()
        with patch.object(handler, "_dispatch_worker_step"):
            handler._finish_recovery(True)
        self.assertTrue(handler._recovery_success)
        self.assertFalse(handler.is_recovering)
        self.assertTrue(handler._recovery_done_event.is_set())

    def test_finish_recovery_failure_state(self):
        handler, _, _ = self._create_handler()
        handler._finish_recovery(False)
        self.assertFalse(handler._recovery_success)
        self.assertFalse(handler.is_recovering)
        self.assertTrue(handler._recovery_done_event.is_set())

    def test_finish_recovery_success_sends_completed_step(self):
        handler, _, _ = self._create_handler()
        with patch.object(handler, "_dispatch_worker_step") as mock_dispatch:
            handler._finish_recovery(True)
        mock_dispatch.assert_called_once()

    def test_wait_for_recovery_timeout(self):
        handler, _, _ = self._create_handler()
        result = handler.wait_for_recovery(timeout=0.1)
        self.assertFalse(result)

    def test_wait_for_recovery_after_success(self):
        handler, _, _ = self._create_handler()
        with patch.object(handler, "_dispatch_worker_step"):
            handler._finish_recovery(True)
        result = handler.wait_for_recovery(timeout=0.1)
        self.assertTrue(result)

    def test_get_recovery_success_initial(self):
        handler, _, _ = self._create_handler()
        self.assertFalse(handler.get_recovery_success())

    def test_duplicate_fault_report_blocked(self):
        handler, _, _ = self._create_handler()
        handler.is_recovering = True
        handler._recover_report_pull_sock = Mock()
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="507057")
        plan = RecoveryPlan(name="test_plan", timeout_s=300)
        report = FaultReport(worker_rank=0, engine_index=0, exp=exc, plan=plan)
        buffer = msgspec.msgpack.encode(report)
        handler._recover_report_pull_sock.recv.return_value = buffer
        handler._handle_worker_msg()
        self.assertTrue(handler.is_recovering)

    def test_full_recovery_lifecycle(self):
        handler, _, _ = self._create_handler()
        self.assertFalse(handler.is_recovering)
        handler._begin_recovery()
        self.assertTrue(handler.is_recovering)
        with patch.object(handler, "_dispatch_worker_step"):
            handler._finish_recovery(True)
        self.assertFalse(handler.is_recovering)
        self.assertTrue(handler._recovery_success)
        self.assertTrue(handler._recovery_done_event.is_set())

    @patch("vllm_ascend.recovery.engine_core_recovery_handler.make_zmq_socket")
    def test_setup_recover_sockets(self, mock_make_zmq):
        handler, _, _ = self._create_handler()
        handler.setup_recover_sockets("ipc://pub", "ipc://report", "ipc://result")
        self.assertEqual(mock_make_zmq.call_count, 3)

    def test_wait_for_worker_subscriptions(self):
        handler, _, _ = self._create_handler(local_world_size=2)
        handler._recover_step_pub_sock = Mock()
        handler._recover_step_pub_sock.recv.side_effect = [b"\x01", b"\x01"]
        handler.wait_for_worker_subscriptions()

    def test_wait_for_worker_subscriptions_unexpected_msg(self):
        handler, _, _ = self._create_handler(local_world_size=1)
        handler._recover_step_pub_sock = Mock()
        handler._recover_step_pub_sock.recv.side_effect = [b"\xfe"]
        handler.wait_for_worker_subscriptions()

    def test_connect_coordinator(self):
        handler, _, _ = self._create_handler()
        with patch("vllm_ascend.recovery.engine_core_recovery_handler.make_zmq_socket") as mock_make:
            mock_sub = Mock()
            mock_push = Mock()
            mock_make.side_effect = [mock_push, mock_sub]
            handler.connect_coordinator("ipc://sub", "ipc://push")
        self.assertEqual(mock_make.call_count, 2)
        mock_sub.setsockopt_string.assert_called_once()
        self.assertTrue(handler._coord_ready_event.is_set())

    def test_execute_engine_core_step_success(self):
        handler, _, _ = self._create_handler()
        step = Mock()
        step.execute.return_value = ({"k": "v"}, True)
        cfg, success = handler._execute_engine_core_step(step, {})
        self.assertTrue(success)
        self.assertEqual(cfg["k"], "v")

    def test_execute_engine_core_step_failure(self):
        handler, _, _ = self._create_handler()
        step = Mock()
        step.execute.return_value = ({}, False)
        cfg, success = handler._execute_engine_core_step(step, {})
        self.assertFalse(success)

    def test_report_plan_result_with_socket(self):
        handler, _, _ = self._create_handler()
        handler._coord_push_sock = Mock()
        result = RecoveryPlanResult(plan_name="p", engine_index=0, success=True)
        handler._report_plan_result(result)
        handler._coord_push_sock.send.assert_called_once()

    def test_report_plan_result_without_socket(self):
        handler, _, _ = self._create_handler()
        handler._coord_push_sock = None
        result = RecoveryPlanResult(plan_name="p", engine_index=0, success=True)
        handler._report_plan_result(result)

    def test_handle_recovery_complete_success(self):
        handler, ec, _ = self._create_handler()
        ec.current_wave = 0
        ec.step_counter = 10
        ec.engines_running = False
        with patch.object(handler, "_dispatch_worker_step"):
            msg = RecoveryComplete(plan_name="p", success=True, current_wave=5)
            handler._handle_recovery_complete(msg)
        self.assertEqual(ec.current_wave, 5)
        self.assertEqual(ec.step_counter, 0)
        self.assertTrue(ec.engines_running)

    def test_handle_recovery_complete_failure(self):
        handler, ec, _ = self._create_handler()
        ec.current_wave = 0
        with patch.object(handler, "_dispatch_worker_step"):
            msg = RecoveryComplete(plan_name="p", success=False, current_wave=3)
            handler._handle_recovery_complete(msg)
        self.assertEqual(ec.current_wave, 3)

    def test_handle_coord_msg_recoveryplan(self):
        handler, _, _ = self._create_handler()
        handler._coord_sub_sock = Mock()
        plan = RecoveryPlan(name="coord_plan", timeout_s=100)
        msg_data = msgspec.msgpack.encode(("recoveryplan", plan))
        handler._coord_sub_sock.recv.return_value = msg_data
        with patch.object(handler, "_execute_recovery") as mock_exec:
            handler._handle_coord_msg()
        mock_exec.assert_called_once()
        self.assertTrue(handler.is_recovering)

    def test_handle_coord_msg_recoverycomplete(self):
        handler, _, _ = self._create_handler()
        handler._coord_sub_sock = Mock()
        complete = RecoveryComplete(plan_name="p", success=True, current_wave=1)
        msg_data = msgspec.msgpack.encode(("recoverycomplete", complete))
        handler._coord_sub_sock.recv.return_value = msg_data
        with patch.object(handler, "_dispatch_worker_step"):
            handler._handle_coord_msg()
        self.assertFalse(handler.is_recovering)

    def test_handle_coord_msg_unknown_type(self):
        handler, _, _ = self._create_handler()
        handler._coord_sub_sock = Mock()
        handler._coord_sub_sock.recv.return_value = msgspec.msgpack.encode(("unknown", b""))
        handler._handle_coord_msg()

    def test_handle_coord_msg_decode_error(self):
        handler, _, _ = self._create_handler()
        handler._coord_sub_sock = Mock()
        handler._coord_sub_sock.recv.return_value = b"invalid_data"
        handler._handle_coord_msg()

    def test_handle_worker_msg_decode_error(self):
        handler, _, _ = self._create_handler()
        handler._recover_report_pull_sock = Mock()
        handler._recover_report_pull_sock.recv.return_value = b"invalid"
        handler._handle_worker_msg()

    def test_handle_worker_msg_wrong_type(self):
        handler, _, _ = self._create_handler()
        handler._recover_report_pull_sock = Mock()
        handler._recover_report_pull_sock.recv.return_value = msgspec.msgpack.encode(
            ExceptionInfo(exception_type="Test", exception_msg="msg")
        )
        handler._handle_worker_msg()

    def test_handle_worker_msg_no_coordinator(self):
        handler, _, _ = self._create_handler(needs_dp=False)
        handler._recover_report_pull_sock = Mock()
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="507057 fail")
        plan = RecoveryPlan(name="test_plan", timeout_s=300)
        report = FaultReport(worker_rank=0, engine_index=0, exp=exc, plan=plan)
        handler._recover_report_pull_sock.recv.return_value = msgspec.msgpack.encode(report)
        with patch.object(handler, "_execute_recovery") as mock_exec:
            handler._handle_worker_msg()
        mock_exec.assert_called_once_with(plan)

    def test_finalize_recovery_no_coordinator_success(self):
        handler, ec, _ = self._create_handler(needs_dp=False)
        ec.current_wave = 0
        ec.step_counter = 0
        ec.engines_running = False
        with patch.object(handler, "_dispatch_worker_step"):
            plan = RecoveryPlan(name="p", timeout_s=100)
            result = RecoveryPlanResult(plan_name="p", engine_index=0, success=True)
            handler._finalize_recovery(plan, result)
        self.assertEqual(ec.current_wave, 1)
        self.assertTrue(handler._recovery_success)

    def test_finalize_recovery_no_coordinator_failure(self):
        handler, ec, _ = self._create_handler(needs_dp=False)
        ec.current_wave = 0
        with patch.object(handler, "_dispatch_worker_step"):
            plan = RecoveryPlan(name="p", timeout_s=100)
            result = RecoveryPlanResult(plan_name="p", engine_index=0, success=False)
            handler._finalize_recovery(plan, result)
        self.assertFalse(handler._recovery_success)

    def test_finalize_recovery_coordinator_failure(self):
        handler, _, _ = self._create_handler(needs_dp=True)
        handler._coord_push_sock = Mock()
        handler._recover_step_pub_sock = Mock()
        handler._recover_step_result_pull_sock = Mock()

        plan = RecoveryPlan(name="p", timeout_s=100)
        result = RecoveryPlanResult(plan_name="p", engine_index=0, success=False)
        handler._finalize_recovery(plan, result)
        self.assertFalse(handler._recovery_success)

    def test_execute_recovery_all_steps_success(self):
        handler, _, _ = self._create_handler()
        step = RecoveryStep(name="ec_step", target="engine_core", actions=[])
        plan = RecoveryPlan(name="p", timeout_s=100, steps=[step])

        with patch.object(handler, "_execute_engine_core_step", return_value=({}, True)), \
             patch.object(handler, "_finalize_recovery") as mock_finalize:
            handler._execute_recovery(plan)
        mock_finalize.assert_called_once()
        call_args = mock_finalize.call_args[0]
        self.assertTrue(call_args[1].success)

    def test_execute_recovery_step_failure_aborts(self):
        handler, _, _ = self._create_handler()
        step1 = RecoveryStep(name="s1", target="engine_core", actions=[])
        step2 = RecoveryStep(name="s2", target="engine_core", actions=[])
        plan = RecoveryPlan(name="p", timeout_s=100, steps=[step1, step2])

        with patch.object(handler, "_execute_engine_core_step", side_effect=[({}, True), ({}, False)]), \
             patch.object(handler, "_finalize_recovery") as mock_finalize:
            handler._execute_recovery(plan)
        call_args = mock_finalize.call_args[0]
        self.assertFalse(call_args[1].success)

    def test_execute_recovery_unknown_target(self):
        handler, _, _ = self._create_handler()
        step = RecoveryStep(name="bad", target="unknown", actions=[])
        plan = RecoveryPlan(name="p", timeout_s=100, steps=[step])

        with patch.object(handler, "_finalize_recovery") as mock_finalize:
            handler._execute_recovery(plan)
        call_args = mock_finalize.call_args[0]
        self.assertFalse(call_args[1].success)

    def test_execute_recovery_worker_step(self):
        handler, _, _ = self._create_handler()
        step = RecoveryStep(name="w_step", target="worker", actions=[])
        plan = RecoveryPlan(name="p", timeout_s=100, steps=[step])

        with patch.object(handler, "_dispatch_worker_step",
                          return_value=({"k": "v"}, True)), \
             patch.object(handler, "_finalize_recovery") as mock_finalize:
            handler._execute_recovery(plan)
        call_args = mock_finalize.call_args[0]
        self.assertTrue(call_args[1].success)

    def test_finalize_recovery_coordinator_success(self):
        handler, _, _ = self._create_handler(needs_dp=True)
        handler._coord_push_sock = Mock()
        handler._begin_recovery()
        plan = RecoveryPlan(name="p", timeout_s=100)
        result = RecoveryPlanResult(plan_name="p", engine_index=0, success=True)
        handler._finalize_recovery(plan, result)
        self.assertTrue(handler.is_recovering)

    def test_run_with_worker_msg(self):
        handler, _, _ = self._create_handler(needs_dp=False)
        handler._recover_report_pull_sock = Mock()
        handler._recover_report_pull_sock.recv.return_value = msgspec.msgpack.encode(
            FaultReport(
                worker_rank=0, engine_index=0,
                exp=ExceptionInfo(exception_type="E", exception_msg="err"),
                plan=RecoveryPlan(name="p", timeout_s=100),
            )
        )

        import itertools
        itertools.count()

        with patch("zmq.Poller") as mock_poller_cls, \
             patch.object(handler, "_execute_recovery") as mock_exec:
            mock_poller = Mock()
            mock_poller_cls.return_value = mock_poller
            handler._recover_report_pull_sock.fileno.return_value = -1

            call_count = [0]

            def poll_effect(timeout=None):
                call_count[0] += 1
                if call_count[0] == 1:
                    return {handler._recover_report_pull_sock: zmq.POLLIN}
                raise StopIteration

            mock_poller.poll.side_effect = poll_effect

            try:
                handler._run()
            except StopIteration:
                pass
        mock_exec.assert_called_once()

    @patch("vllm_ascend.recovery.engine_core_recovery_handler.threading.Thread")
    def test_start_creates_thread(self, mock_thread_cls):
        handler, _, _ = self._create_handler()
        handler.start()
        mock_thread_cls.assert_called_once()
        mock_thread_cls.return_value.start.assert_called_once()

    def test_handle_worker_msg_with_coordinator(self):
        handler, _, _ = self._create_handler(needs_dp=True)
        handler._recover_report_pull_sock = Mock()
        handler._coord_push_sock = Mock()
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="507057")
        plan = RecoveryPlan(name="test_plan", timeout_s=300)
        report = FaultReport(worker_rank=0, engine_index=0, exp=exc, plan=plan)
        handler._recover_report_pull_sock.recv.return_value = msgspec.msgpack.encode(report)
        handler._handle_worker_msg()
        handler._coord_push_sock.send.assert_called_once()

    def test_handle_worker_msg_coordinator_no_connection(self):
        handler, _, _ = self._create_handler(needs_dp=True)
        handler._recover_report_pull_sock = Mock()
        handler._coord_push_sock = None
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="507057")
        plan = RecoveryPlan(name="test_plan", timeout_s=300)
        report = FaultReport(worker_rank=0, engine_index=0, exp=exc, plan=plan)
        handler._recover_report_pull_sock.recv.return_value = msgspec.msgpack.encode(report)
        handler._handle_worker_msg()
