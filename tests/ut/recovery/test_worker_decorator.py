# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

from tests.ut.base import TestBase
from vllm_ascend.recovery.worker_decorator import fault_recovery_decorator


class TestWorkerDecorator(TestBase):

    def _make_worker(self, exception_occur=False, in_recovery=False,
                     with_socket=True):
        worker = Mock()
        worker.exception_occur = exception_occur
        worker.in_recovery = in_recovery
        if with_socket:
            worker.worker_input_socket = Mock()
        return worker

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", False)
    def test_skips_when_recovery_disabled(self):
        @fault_recovery_decorator()
        def test_func(worker):
            return "result"

        self.assertEqual(test_func(Mock()), "result")

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_normal_execution_returns_result(self):
        @fault_recovery_decorator()
        def test_func(worker):
            return "result"

        self.assertEqual(test_func(self._make_worker()), "result")

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_skips_when_exception_occur(self):
        @fault_recovery_decorator()
        def test_func(worker):
            return "should_not_be_called"

        self.assertIsNone(test_func(self._make_worker(exception_occur=True)))

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_skips_when_in_recovery(self):
        @fault_recovery_decorator()
        def test_func(worker):
            return "should_not_be_called"

        self.assertIsNone(test_func(self._make_worker(in_recovery=True)))

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_exception_sets_flags(self):
        @fault_recovery_decorator()
        def test_func(worker):
            raise RuntimeError("test error")

        worker = self._make_worker()
        with self.assertRaises(RuntimeError):
            test_func(worker)
        self.assertTrue(worker.exception_occur)
        self.assertTrue(worker.in_recovery)

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_exception_sends_to_socket(self):
        @fault_recovery_decorator()
        def test_func(worker):
            raise RuntimeError("test error")

        worker = self._make_worker()
        with self.assertRaises(RuntimeError):
            test_func(worker)
        worker.worker_input_socket.send.assert_called_once()

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_exception_re_raises(self):
        @fault_recovery_decorator()
        def test_func(worker):
            raise RuntimeError("test error")

        with self.assertRaisesRegex(RuntimeError, "test error"):
            test_func(self._make_worker())

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_decorator_preserves_func_attrs(self):
        @fault_recovery_decorator()
        def test_func(worker):
            """Test docstring."""
            return "result"

        self.assertEqual(test_func.__name__, "test_func")
        self.assertEqual(test_func.__doc__, "Test docstring.")
