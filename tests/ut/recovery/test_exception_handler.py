# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from tests.ut.base import TestBase
from vllm_ascend.recovery.exception_handler import (
    ExceptionHandlerFactory,
    NetworkExceptionHandler,
)
from vllm_ascend.recovery.types import ExceptionInfo


class TestExceptionHandler(TestBase):

    def setUp(self):
        self.handler = NetworkExceptionHandler()

    def _plan(self):
        return self.handler.generate_plan(
            ExceptionInfo(exception_type="RuntimeError", exception_msg="507057"),
            Mock(),
        )

    def test_network_handler_matches_code(self):
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="error[507057] test")
        self.assertTrue(self.handler.can_handle(exc))

    def test_network_handler_matches_embedded(self):
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="communication failed code:507057 timeout")
        self.assertTrue(self.handler.can_handle(exc))

    def test_network_handler_no_match(self):
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="out of memory")
        self.assertFalse(self.handler.can_handle(exc))

    def test_network_handler_empty_msg_no_match(self):
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="")
        self.assertFalse(self.handler.can_handle(exc))

    def test_generate_plan_name(self):
        self.assertEqual(self._plan().name, "network_recover_plan")

    def test_generate_plan_step_count(self):
        self.assertEqual(len(self._plan().steps), 4)

    def test_generate_plan_step_names(self):
        step_names = [s.name for s in self._plan().steps]
        self.assertEqual(
            step_names,
            ["device_recovery", "clean_engine_cache", "clean_worker_cache", "re_capture_graph"],
        )

    def test_generate_plan_device_step_actions(self):
        action_names = [a.name for a in self._plan().steps[0].actions]
        self.assertEqual(action_names, ["recovery_begin", "stop_device", "restart_device", "reinit_process_group"])

    def test_generate_plan_engine_step_actions(self):
        action_names = [a.name for a in self._plan().steps[1].actions]
        self.assertEqual(action_names, ["label_dirty_requests", "clean_batch_queue", "recompute_dirty_requests"])

    def test_generate_plan_worker_clean_step_actions(self):
        action_names = [a.name for a in self._plan().steps[2].actions]
        self.assertEqual(action_names, ["worker_clean_dirty_requests_cache"])

    def test_generate_plan_recapture_step_actions(self):
        action_names = [a.name for a in self._plan().steps[3].actions]
        self.assertEqual(action_names, ["worker_rebuild_cpu_group", "worker_recapture_graph"])

    def test_generate_plan_default_cfg(self):
        plan = self._plan()
        self.assertEqual(plan.cfg["rebuild_all_resources"], False)
        self.assertIsNone(plan.cfg["group"])
        self.assertEqual(plan.cfg["rebuild_link"], False)

    def test_generate_plan_timeouts(self):
        plan = self._plan()
        self.assertEqual([s.timeout_s for s in plan.steps], [60, 60, 60, 120])
        self.assertEqual(plan.timeout_s, 300)

    def test_factory_get_handler_match(self):
        factory = ExceptionHandlerFactory()
        factory._register_handler(NetworkExceptionHandler())
        handler = factory.get_handler(
            ExceptionInfo(exception_type="RuntimeError", exception_msg="error[507057]")
        )
        self.assertIsInstance(handler, NetworkExceptionHandler)

    def test_factory_get_handler_no_match(self):
        factory = ExceptionHandlerFactory()
        factory._register_handler(NetworkExceptionHandler())
        handler = factory.get_handler(
            ExceptionInfo(exception_type="RuntimeError", exception_msg="out of memory")
        )
        self.assertIsNone(handler)

    def test_factory_first_match_wins(self):
        factory = ExceptionHandlerFactory()
        handler_a = NetworkExceptionHandler()
        handler_b = NetworkExceptionHandler()
        factory._register_handler(handler_a)
        factory._register_handler(handler_b)
        handler = factory.get_handler(
            ExceptionInfo(exception_type="RuntimeError", exception_msg="error[507057]")
        )
        self.assertIs(handler, handler_a)
