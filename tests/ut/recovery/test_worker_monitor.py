# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.recovery.worker_monitor import WorkerMonitor, create_worker_monitor


class TestWorkerMonitor:
    def test_build_exception_handler_factory(self):
        vllm_config = MagicMock()
        worker = MagicMock()
        ctx = MagicMock()
        monitor = WorkerMonitor(vllm_config, worker, ctx)
        factory = monitor.build_exception_handler_factory()
        assert len(factory.handlers) == 1

    def test_init_sets_addresses(self):
        vllm_config = MagicMock()
        vllm_config.parallel_config.data_parallel_rank = 2
        worker = MagicMock()
        ctx = MagicMock()
        monitor = WorkerMonitor(vllm_config, worker, ctx)
        assert monitor.engine_index == 2
        assert monitor.worker_input_address is not None
        assert monitor.core_input_address is not None
        assert monitor.core_report_address is not None
        assert monitor.core_result_address is not None


class TestCreateWorkerMonitor:
    @patch("vllm_ascend.recovery.worker_monitor.VLLM_ASCEND_ENABLE_RECOVERY", False)
    def test_recovery_disabled_returns_none(self):
        worker = MagicMock()
        vllm_config = MagicMock()
        result = create_worker_monitor(worker, vllm_config)
        assert result is None

    @patch("vllm_ascend.recovery.worker_monitor.VLLM_ASCEND_ENABLE_RECOVERY", True)
    @patch("vllm_ascend.recovery.worker_monitor.make_zmq_socket")
    @patch("vllm_ascend.recovery.worker_monitor.zmq")
    def test_creates_monitor_on_worker(self, mock_zmq, mock_make_socket):
        mock_make_socket.return_value = MagicMock()
        worker = MagicMock()
        worker.worker_monitor = None
        vllm_config = MagicMock()
        vllm_config.parallel_config.data_parallel_rank = 0
        create_worker_monitor(worker, vllm_config)
        assert worker.worker_monitor is not None
        assert worker.in_recovery is False
        assert worker.exception_occur is False
        assert worker.device_stopped is False

    @patch("vllm_ascend.recovery.worker_monitor.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_skips_if_monitor_already_exists(self):
        worker = MagicMock()
        worker.worker_monitor = MagicMock()
        vllm_config = MagicMock()
        create_worker_monitor(worker, vllm_config)
