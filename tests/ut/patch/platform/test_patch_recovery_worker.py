# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from tests.ut.base import TestBase
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm_ascend.patch.platform.patch_recovery_worker import enqueue_output


class _FakeResponseStatus:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class _FakeAsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def get_output(self):
        if self._exc is not None:
            raise self._exc
        return self._result


def _make_worker_proc_mock(response_mq=None):
    mp = MagicMock()
    mp.worker.worker.exception_occur = False
    mp.worker.worker.in_recovery = False
    mp.worker.worker_input_socket.send = MagicMock()
    mp.worker_response_mq = response_mq
    return mp


class TestPatchRecoveryWorker(TestBase):

    def setUp(self):
        super().setUp()
        self._orig_resp_status = WorkerProc.ResponseStatus
        WorkerProc.ResponseStatus = _FakeResponseStatus()

    def tearDown(self):
        WorkerProc.ResponseStatus = self._orig_resp_status
        super().tearDown()

    def test_async_output_success(self):
        mp = _make_worker_proc_mock(response_mq=MagicMock())
        output = _FakeAsyncOutput(result={"tokens": [1, 2, 3]})
        enqueue_output(mp, output)

        mp.worker_response_mq.enqueue.assert_called_once()
        result = mp.worker_response_mq.enqueue.call_args[0][0]
        self.assertEqual(result[0], "SUCCESS")
        self.assertEqual(result[1], {"tokens": [1, 2, 3]})

    def test_async_output_exception_sets_flags_and_sends(self):
        mp = _make_worker_proc_mock(response_mq=MagicMock())
        exc = ValueError("test error")
        output = _FakeAsyncOutput(exc=exc)
        enqueue_output(mp, output)

        self.assertTrue(mp.worker.worker.exception_occur)
        self.assertTrue(mp.worker.worker.in_recovery)
        mp.worker.worker_input_socket.send.assert_called_once()
        self.assertEqual(
            mp.worker_response_mq.enqueue.call_args[0][0][0], "FAILURE"
        )

    def test_async_output_exception_already_in_recovery_no_resend(self):
        mp = _make_worker_proc_mock(response_mq=MagicMock())
        mp.worker.worker.in_recovery = True
        exc = ValueError("second error")
        output = _FakeAsyncOutput(exc=exc)
        enqueue_output(mp, output)

        self.assertTrue(mp.worker.worker.exception_occur)
        mp.worker.worker_input_socket.send.assert_not_called()

    def test_direct_exception_output(self):
        mp = _make_worker_proc_mock(response_mq=MagicMock())
        exc = RuntimeError("direct error")
        enqueue_output(mp, exc)

        result = mp.worker_response_mq.enqueue.call_args[0][0]
        self.assertEqual(result[0], "FAILURE")
        self.assertIn("direct error", result[1])

    def test_plain_value_output(self):
        mp = _make_worker_proc_mock(response_mq=MagicMock())
        output = {"key": "value"}
        enqueue_output(mp, output)

        result = mp.worker_response_mq.enqueue.call_args[0][0]
        self.assertEqual(result[0], "SUCCESS")
        self.assertEqual(result[1], {"key": "value"})

    def test_response_mq_none_no_enqueue(self):
        mp = _make_worker_proc_mock(response_mq=None)
        output = {"key": "value"}
        enqueue_output(mp, output)

        self.assertIsNone(mp.worker_response_mq)
