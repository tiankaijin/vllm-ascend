# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import AsyncModelRunnerOutput


class TestPatchedEnqueueOutput:
    def test_success_path(self):
        from vllm_ascend.patch.platform.patch_recovery_worker import enqueue_output

        mock_self = MagicMock()
        mock_self.worker.worker.exception_occur = False
        mock_self.worker.worker.in_recovery = False
        mock_self.worker_response_mq = MagicMock()

        output = {"key": "value"}
        enqueue_output(mock_self, output)

        mock_self.worker_response_mq.enqueue.assert_called_once()
        result = mock_self.worker_response_mq.enqueue.call_args[0][0]
        assert result[0] == WorkerProc.ResponseStatus.SUCCESS

    def test_exception_path_sets_flags_and_sends_failure(self):
        from vllm_ascend.patch.platform.patch_recovery_worker import enqueue_output

        mock_self = MagicMock()
        mock_self.worker.worker.exception_occur = False
        mock_self.worker.worker.in_recovery = False
        mock_self.worker.worker_input_socket = MagicMock()
        mock_self.worker_response_mq = MagicMock()

        async_output = MagicMock(spec=AsyncModelRunnerOutput)
        async_output.get_output.side_effect = RuntimeError("NPU error 507057")

        enqueue_output(mock_self, async_output)

        assert mock_self.worker.worker.exception_occur is True
        assert mock_self.worker.worker.in_recovery is True
        mock_self.worker.worker_input_socket.send.assert_called_once()
        mock_self.worker_response_mq.enqueue.assert_called_once()
        result = mock_self.worker_response_mq.enqueue.call_args[0][0]
        assert result[0] == WorkerProc.ResponseStatus.FAILURE

    def test_exception_in_recovery_does_not_resend(self):
        from vllm_ascend.patch.platform.patch_recovery_worker import enqueue_output

        mock_self = MagicMock()
        mock_self.worker.worker.exception_occur = True
        mock_self.worker.worker.in_recovery = True
        mock_self.worker.worker_input_socket = MagicMock()
        mock_self.worker_response_mq = MagicMock()

        async_output = MagicMock(spec=AsyncModelRunnerOutput)
        async_output.get_output.side_effect = RuntimeError("another error")

        enqueue_output(mock_self, async_output)

        mock_self.worker.worker_input_socket.send.assert_not_called()
        result = mock_self.worker_response_mq.enqueue.call_args[0][0]
        assert result[0] == WorkerProc.ResponseStatus.FAILURE

    def test_plain_exception_enqueued_as_failure(self):
        from vllm_ascend.patch.platform.patch_recovery_worker import enqueue_output

        mock_self = MagicMock()
        mock_self.worker_response_mq = MagicMock()

        exc = ValueError("bad value")
        enqueue_output(mock_self, exc)

        result = mock_self.worker_response_mq.enqueue.call_args[0][0]
        assert result[0] == WorkerProc.ResponseStatus.FAILURE
        assert "bad value" in result[1]
