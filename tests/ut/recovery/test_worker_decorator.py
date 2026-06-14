# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.recovery.worker_decorator import fault_recovery_decorator


class TestFaultRecoveryDecorator:
    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", False)
    def test_recovery_disabled_calls_func_directly(self):
        decorator = fault_recovery_decorator()

        @decorator
        def my_func(self):
            return "result"

        mock_self = MagicMock()
        result = my_func(mock_self)
        assert result == "result"

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_exception_occur_returns_none(self):
        decorator = fault_recovery_decorator()

        @decorator
        def my_func(self):
            return "result"

        mock_self = MagicMock()
        mock_self.exception_occur = True
        mock_self.in_recovery = False
        result = my_func(mock_self)
        assert result is None

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_in_recovery_returns_none(self):
        decorator = fault_recovery_decorator()

        @decorator
        def my_func(self):
            return "result"

        mock_self = MagicMock()
        mock_self.exception_occur = False
        mock_self.in_recovery = True
        result = my_func(mock_self)
        assert result is None

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_normal_execution_returns_result(self):
        decorator = fault_recovery_decorator()

        @decorator
        def my_func(self):
            return "ok"

        mock_self = MagicMock()
        mock_self.exception_occur = False
        mock_self.in_recovery = False
        result = my_func(mock_self)
        assert result == "ok"

    @patch("vllm_ascend.recovery.worker_decorator.VLLM_ASCEND_ENABLE_RECOVERY", True)
    def test_exception_sets_flags_and_raises(self):
        decorator = fault_recovery_decorator()

        @decorator
        def my_func(self):
            raise RuntimeError("NPU error")

        mock_self = MagicMock()
        mock_self.exception_occur = False
        mock_self.in_recovery = False
        mock_self.worker_input_socket = MagicMock()
        with pytest.raises(RuntimeError, match="NPU error"):
            my_func(mock_self)
        assert mock_self.exception_occur is True
        assert mock_self.in_recovery is True
        mock_self.worker_input_socket.send.assert_called_once()
