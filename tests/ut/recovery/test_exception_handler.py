# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from vllm_ascend.recovery.exception_handler import (
    ExceptionHandlerFactory,
    NetworkExceptionHandler,
)
from vllm_ascend.recovery.types import ExceptionInfo, RecoveryPlan


class TestNetworkExceptionHandler:
    def test_can_handle_matching_error_code(self):
        handler = NetworkExceptionHandler()
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="NPU error 507057 link down")
        assert handler.can_handle(exc) is True

    def test_can_handle_non_matching_error_code(self):
        handler = NetworkExceptionHandler()
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="NPU error 999999 unknown")
        assert handler.can_handle(exc) is False

    def test_generate_plan_returns_recovery_plan(self):
        handler = NetworkExceptionHandler()
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="507057")
        vllm_config = MagicMock()
        plan = handler.generate_plan(exc, vllm_config)
        assert isinstance(plan, RecoveryPlan)
        assert plan.name == "network_recover_plan"
        assert len(plan.steps) == 4
        assert plan.timeout_s == 300


class TestExceptionHandlerFactory:
    def test_get_handler_returns_matching_handler(self):
        factory = ExceptionHandlerFactory()
        handler = NetworkExceptionHandler()
        factory._register_handler(handler)
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="507057 link error")
        result = factory.get_handler(exc)
        assert result is handler

    def test_get_handler_returns_none_when_no_match(self):
        factory = ExceptionHandlerFactory()
        handler = NetworkExceptionHandler()
        factory._register_handler(handler)
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="unrelated error")
        result = factory.get_handler(exc)
        assert result is None

    def test_get_handler_with_no_handlers(self):
        factory = ExceptionHandlerFactory()
        exc = ExceptionInfo(exception_type="RuntimeError", exception_msg="507057")
        result = factory.get_handler(exc)
        assert result is None
