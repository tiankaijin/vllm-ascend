# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, Mock, patch

import msgspec.msgpack
import zmq

from tests.ut.base import TestBase
from vllm_ascend.recovery.types import ExceptionInfo, RecoveryStep, WorkerStepDispatch


class FakeWorker:
    pass


class TestWorkerMonitor(TestBase):

    @staticmethod
    def _ctx_sock():
        sock = MagicMock()
        sock.__enter__.return_value = sock
        return sock

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_RECOVERY": "0"})
    def test_create_skips_when_disabled(self):
        from vllm_ascend.recovery.worker_monitor import create_worker_monitor

        worker = FakeWorker()
        create_worker_monitor(worker, Mock())
        self.assertFalse(hasattr(worker, "in_recovery"))
        self.assertFalse(hasattr(worker, "worker_monitor"))

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_RECOVERY": "1"})
    def test_create_skips_if_already_exists(self):
        from vllm_ascend.recovery.worker_monitor import create_worker_monitor

        worker = Mock()
        existing_monitor = Mock()
        worker.worker_monitor = existing_monitor

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.make_zmq_socket"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            create_worker_monitor(worker, Mock())
        self.assertIs(worker.worker_monitor, existing_monitor)

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_RECOVERY": "1"})
    def test_create_sets_worker_state(self):
        from vllm_ascend.recovery.worker_monitor import create_worker_monitor

        worker = Mock()
        worker.worker_monitor = None
        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 0

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.make_zmq_socket"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            create_worker_monitor(worker, vllm_config)
        self.assertFalse(worker.in_recovery)
        self.assertFalse(worker.exception_occur)
        self.assertFalse(worker.device_stopped)

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_RECOVERY": "1"})
    def test_create_sets_socket(self):
        from vllm_ascend.recovery.worker_monitor import create_worker_monitor

        worker = Mock()
        worker.worker_monitor = None
        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 0

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.make_zmq_socket"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            create_worker_monitor(worker, vllm_config)
        self.assertIsNotNone(worker.worker_input_socket)
        self.assertIsNotNone(worker.worker_monitor)

    def test_build_factory_has_network_handler(self):
        from vllm_ascend.recovery.worker_monitor import WorkerMonitor
        from vllm_ascend.recovery.exception_handler import NetworkExceptionHandler

        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 0

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            monitor = WorkerMonitor(vllm_config, Mock(), Mock())
        self.assertEqual(len(monitor.exception_handler_factory.handlers), 1)
        self.assertIsInstance(
            monitor.exception_handler_factory.handlers[0], NetworkExceptionHandler
        )

    def test_init_engine_index(self):
        from vllm_ascend.recovery.worker_monitor import WorkerMonitor

        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 2

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            monitor = WorkerMonitor(vllm_config, Mock(), Mock())
        self.assertEqual(monitor.engine_index, 2)

    def test_init_decoder_types(self):
        from vllm_ascend.recovery.worker_monitor import WorkerMonitor

        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 0

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            monitor = WorkerMonitor(vllm_config, Mock(), Mock())

        exc_info = msgspec.msgpack.encode(
            ExceptionInfo(exception_type="RuntimeError", exception_msg="test")
        )
        decoded = monitor._exception_decoder.decode(exc_info)
        self.assertIsInstance(decoded, ExceptionInfo)

        dispatch = msgspec.msgpack.encode(
            WorkerStepDispatch(
                step=RecoveryStep(name="test_step", target="worker"), cfg={}
            )
        )
        decoded = monitor._recovery_decoder.decode(dispatch)
        self.assertIsInstance(decoded, WorkerStepDispatch)

    @patch("threading.Thread")
    def test_start_creates_thread(self, mock_thread_cls):
        from vllm_ascend.recovery.worker_monitor import WorkerMonitor

        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 0

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            monitor = WorkerMonitor(vllm_config, Mock(), Mock())
        monitor.start()
        mock_thread_cls.assert_called_once()
        mock_thread_cls.return_value.start.assert_called_once()

    @patch("vllm_ascend.recovery.worker_monitor.make_zmq_socket")
    @patch("zmq.Poller")
    def test_run_monitor_worker_exception(self, mock_poller_cls, mock_make_zmq):
        from vllm_ascend.recovery.worker_monitor import WorkerMonitor

        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 0

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            worker = Mock()
            worker.rank = 0
            monitor = WorkerMonitor(vllm_config, worker, Mock())

        w = self._ctx_sock()
        ci = self._ctx_sock()
        cr = self._ctx_sock()
        cq = self._ctx_sock()
        mock_make_zmq.side_effect = [w, ci, cr, cq]

        w.recv.return_value = msgspec.msgpack.encode(
            ExceptionInfo(exception_type="RuntimeError", exception_msg="507057")
        )

        p = MagicMock()
        mock_poller_cls.return_value = p
        cnt = [0]
        def fx(timeout=None):
            cnt[0] += 1
            if cnt[0] == 1:
                return {w: zmq.POLLIN}
            raise StopIteration
        p.poll.side_effect = fx

        try:
            monitor._run_monitor()
        except StopIteration:
            pass

        ci.send.assert_called_once_with(b"\x01")
        cr.send.assert_called_once()

    @patch("vllm_ascend.recovery.worker_monitor.make_zmq_socket")
    @patch("zmq.Poller")
    def test_run_monitor_recovery_step(self, mock_poller_cls, mock_make_zmq):
        from vllm_ascend.recovery.worker_monitor import WorkerMonitor

        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 0
        worker = Mock()
        worker.rank = 0

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            monitor = WorkerMonitor(vllm_config, worker, Mock())

        w = self._ctx_sock()
        ci = self._ctx_sock()
        cr = self._ctx_sock()
        cq = self._ctx_sock()
        mock_make_zmq.side_effect = [w, ci, cr, cq]

        ci.recv.return_value = msgspec.msgpack.encode(
            WorkerStepDispatch(
                step=RecoveryStep(name="test_step", target="worker", actions=[]),
                cfg={},
            )
        )

        p = MagicMock()
        mock_poller_cls.return_value = p
        cnt = [0]
        def fx(timeout=None):
            cnt[0] += 1
            if cnt[0] == 1:
                return {ci: zmq.POLLIN}
            raise StopIteration
        p.poll.side_effect = fx

        try:
            monitor._run_monitor()
        except StopIteration:
            pass

        cq.send.assert_called_once()

    @patch("vllm_ascend.recovery.worker_monitor.make_zmq_socket")
    @patch("zmq.Poller")
    def test_run_monitor_decode_error(self, mock_poller_cls, mock_make_zmq):
        from vllm_ascend.recovery.worker_monitor import WorkerMonitor

        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 0

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            monitor = WorkerMonitor(vllm_config, Mock(), Mock())

        w = self._ctx_sock()
        ci = self._ctx_sock()
        cr = self._ctx_sock()
        cq = self._ctx_sock()
        mock_make_zmq.side_effect = [w, ci, cr, cq]

        w.recv.return_value = b"invalid"

        p = MagicMock()
        mock_poller_cls.return_value = p
        cnt = [0]
        def fx(timeout=None):
            cnt[0] += 1
            if cnt[0] == 1:
                return {w: zmq.POLLIN}
            raise StopIteration
        p.poll.side_effect = fx

        try:
            monitor._run_monitor()
        except StopIteration:
            pass

        cr.send.assert_not_called()

    @patch("vllm_ascend.recovery.worker_monitor.make_zmq_socket")
    @patch("zmq.Poller")
    def test_run_monitor_non_recoverable(self, mock_poller_cls, mock_make_zmq):
        from vllm_ascend.recovery.worker_monitor import WorkerMonitor

        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 0

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            monitor = WorkerMonitor(vllm_config, Mock(), Mock())

        w = self._ctx_sock()
        ci = self._ctx_sock()
        cr = self._ctx_sock()
        cq = self._ctx_sock()
        mock_make_zmq.side_effect = [w, ci, cr, cq]

        w.recv.return_value = msgspec.msgpack.encode(
            ExceptionInfo(exception_type="RuntimeError", exception_msg="non-recoverable")
        )

        p = MagicMock()
        mock_poller_cls.return_value = p
        cnt = [0]
        def fx(timeout=None):
            cnt[0] += 1
            if cnt[0] == 1:
                return {w: zmq.POLLIN}
            raise StopIteration
        p.poll.side_effect = fx

        try:
            monitor._run_monitor()
        except StopIteration:
            pass

        cr.send.assert_not_called()

    @patch("vllm_ascend.recovery.worker_monitor.make_zmq_socket")
    @patch("zmq.Poller")
    def test_run_monitor_core_decode_error(self, mock_poller_cls, mock_make_zmq):
        from vllm_ascend.recovery.worker_monitor import WorkerMonitor

        vllm_config = Mock()
        vllm_config.parallel_config.data_parallel_rank = 0

        with patch("vllm_ascend.recovery.worker_monitor.zmq.Context"), \
             patch("vllm_ascend.recovery.worker_monitor.get_open_zmq_ipc_path",
                   return_value="ipc:///tmp/test"):
            monitor = WorkerMonitor(vllm_config, Mock(), Mock())

        w = self._ctx_sock()
        ci = self._ctx_sock()
        cr = self._ctx_sock()
        cq = self._ctx_sock()
        mock_make_zmq.side_effect = [w, ci, cr, cq]

        ci.recv.return_value = b"invalid"

        p = MagicMock()
        mock_poller_cls.return_value = p
        cnt = [0]
        def fx(timeout=None):
            cnt[0] += 1
            if cnt[0] == 1:
                return {ci: zmq.POLLIN}
            raise StopIteration
        p.poll.side_effect = fx

        try:
            monitor._run_monitor()
        except StopIteration:
            pass

        cq.send.assert_not_called()
