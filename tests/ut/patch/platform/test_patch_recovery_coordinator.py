# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec.msgpack
import zmq

from tests.ut.base import TestBase
from vllm_ascend.patch.platform.patch_recovery_coordinator import (
    _patched_dp_coordinator_init
)
from vllm_ascend.recovery.types import (
    ExceptionInfo,
    FaultReport,
    RecoveryComplete,
    RecoveryPlan,
    RecoveryPlanResult,
)


def _noop(*a, **kw):
    pass


class TestPatchRecoveryCoordinator(TestBase):

    def test_patched_init_stores_recovery_addresses(self):
        pc = SimpleNamespace(
            data_parallel_size=2,
            data_parallel_master_ip="127.0.0.1",
            local_engines_only=False,
            data_parallel_size_local=2,
            enable_elastic_ep=False,
        )
        addrs = ["pub", "back_out", "back_pub", "recv_pub", "recv_pull"]
        with patch(
            "vllm_ascend.patch.platform.patch_recovery_coordinator"
            ".get_engine_client_zmq_addr",
            side_effect=addrs,
        ), patch(
            "vllm_ascend.patch.platform.patch_recovery_coordinator"
            ".get_mp_context",
        ) as mock_ctx:
            mock_ctx.return_value.Process.return_value.start = _noop
            self_mock = MagicMock()
            _patched_dp_coordinator_init(self_mock, pc)
            self.assertEqual(self_mock.recovery_pub_address, "recv_pub")
            self.assertEqual(self_mock.recovery_pull_address, "recv_pull")

    def test_patched_init_passes_recovery_kwargs_to_process(self):
        pc = SimpleNamespace(
            data_parallel_size=2,
            data_parallel_master_ip="127.0.0.1",
            local_engines_only=False,
            data_parallel_size_local=2,
            enable_elastic_ep=False,
        )
        with patch(
            "vllm_ascend.patch.platform.patch_recovery_coordinator"
            ".get_engine_client_zmq_addr",
            return_value="addr",
        ), patch(
            "vllm_ascend.patch.platform.patch_recovery_coordinator"
            ".get_mp_context",
        ) as mock_ctx:
            mock_ctx.return_value.Process.return_value.start = _noop
            self_mock = MagicMock()
            _patched_dp_coordinator_init(self_mock, pc)
            kwargs = mock_ctx.return_value.Process.call_args[1]["kwargs"]
            self.assertEqual(kwargs["recovery_pub_address"], "addr")
            self.assertEqual(kwargs["recovery_pull_address"], "addr")

    def test_patched_run_coordinator_forwards_recovery_addresses(self):
        with patch(
            "vllm_ascend.patch.platform.patch_recovery_coordinator"
            ".DPCoordinatorProc",
        ) as mock_proc:
            inst = mock_proc.return_value
            from vllm_ascend.patch.platform.patch_recovery_coordinator \
                import _patched_run_coordinator as run_c
            run_c(
                2, "front", "back_out", "back_pub",
                recovery_pub_address="rpub",
                recovery_pull_address="rpull",
            )
            inst.process_input_socket.assert_called_once_with(
                "front", "back_out", "back_pub",
                recovery_pub_address="rpub",
                recovery_pull_address="rpull",
            )


class TestRecoveryCoordinatorSocketLifecycle(TestBase):

    def _zmq_sock(self):
        sock = MagicMock()
        sock.__enter__ = lambda s: s
        sock.__exit__ = lambda s, *a: None
        return sock

    def _run_one_pass(self, recv_pull, recv_pub, pub_back,
                      pub_front, out_back, poll_events,
                      extra_patches=None, enable_wave=False):
        from vllm_ascend.patch.platform.patch_recovery_coordinator \
            import _patched_process_input_socket

        sockmap = {
            "rpub": recv_pub,
            "rpull": recv_pull,
            "back_pub": pub_back,
            "front_pub": pub_front,
            "back_out": out_back,
        }

        def make_sock(path=None, ctx=None, socket_type=None, bind=None):
            return sockmap.get(path, self._zmq_sock())

        poller = MagicMock()
        events_iter = iter(poll_events)
        poller.poll.side_effect = lambda timeout: dict(next(events_iter))

        patches = [
            patch(
                "vllm_ascend.patch.platform.patch_recovery_coordinator"
                ".make_zmq_socket",
                side_effect=make_sock,
            ),
            patch(
                "vllm_ascend.patch.platform.patch_recovery_coordinator"
                ".zmq.Poller", return_value=poller,
            ),
            patch(
                "vllm_ascend.patch.platform.patch_recovery_coordinator"
                ".MsgpackDecoder",
            ),
        ]
        if extra_patches:
            patches.extend(extra_patches)

        for p in patches:
            p.start()
        try:
            self_mock = MagicMock()
            self_mock.engines = [MagicMock(), MagicMock()]
            self_mock.ctx = MagicMock()
            self_mock.stats_update_interval_ms = 100
            self_mock.enable_wave_coordination = enable_wave
            self_mock._send_start_wave = _noop
            self_mock._get_engine_counts.return_value = []

            try:
                _patched_process_input_socket(
                    self_mock, "front_pub", "back_out", "back_pub",
                    recovery_pub_address="rpub",
                    recovery_pull_address="rpull",
                )
            except StopIteration:
                pass
        finally:
            for p in reversed(patches):
                p.stop()

    def test_faultreport_broadcasts_plan(self):
        plan = RecoveryPlan(name="test_plan", timeout_s=30, steps=[])
        exp = ExceptionInfo(exception_type="ValueError", exception_msg="boom")
        report = FaultReport(worker_rank=0, engine_index=1, exp=exp, plan=plan)

        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        fr_msg = msgspec.msgpack.encode(("faultreport", report))
        plan_msg = msgspec.msgpack.encode(("recoveryplan", plan))

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pull.recv.side_effect = [fr_msg, StopIteration]

        poll_events = [
            {},
            {recv_pull: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )
        recv_pub.send.assert_any_call(plan_msg)

    def test_faultreport_ignored_during_active_recovery(self):
        plan = RecoveryPlan(name="test_plan", timeout_s=30, steps=[])
        exp = ExceptionInfo(exception_type="ValueError", exception_msg="boom")
        report = FaultReport(worker_rank=0, engine_index=1, exp=exp, plan=plan)

        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        fr_msg = msgspec.msgpack.encode(("faultreport", report))

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pull.recv.side_effect = [fr_msg, fr_msg, StopIteration]

        poll_events = [
            {},
            {recv_pull: zmq.POLLIN},
            {},
            {recv_pull: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )
        plan_msg = msgspec.msgpack.encode(
            ("recoveryplan", plan)
        )
        self.assertEqual(recv_pub.send.call_count, 1)

    def test_recoveryplanresult_all_engines_success(self):
        plan = RecoveryPlan(name="plan_a", timeout_s=30, steps=[])
        exp = ExceptionInfo(exception_type="ValueError", exception_msg="boom")
        report = FaultReport(worker_rank=0, engine_index=0, exp=exp, plan=plan)
        rpr_0 = RecoveryPlanResult(
            plan_name="plan_a", engine_index=0, success=True, step_results=[]
        )
        rpr_1 = RecoveryPlanResult(
            plan_name="plan_a", engine_index=1, success=True, step_results=[]
        )

        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        fr_msg = msgspec.msgpack.encode(("faultreport", report))
        rpr0_msg = msgspec.msgpack.encode(
            ("recoveryplanresult", rpr_0)
        )
        rpr1_msg = msgspec.msgpack.encode(
            ("recoveryplanresult", rpr_1)
        )

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pull.recv.side_effect = [fr_msg, rpr0_msg, rpr1_msg, StopIteration]

        poll_events = [
            {},
            {recv_pull: zmq.POLLIN},
            {},
            {recv_pull: zmq.POLLIN},
            {},
            {recv_pull: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )
        complete_msg = msgspec.msgpack.encode(
            ("recoverycomplete", RecoveryComplete(
                plan_name="plan_a", success=True, current_wave=0,
            ))
        )
        recv_pub.send.assert_any_call(complete_msg)

    def test_recoveryplanresult_failure(self):
        plan = RecoveryPlan(name="plan_a", timeout_s=30, steps=[])
        exp = ExceptionInfo(exception_type="ValueError", exception_msg="boom")
        report = FaultReport(worker_rank=0, engine_index=0, exp=exp, plan=plan)
        rpr_0 = RecoveryPlanResult(
            plan_name="plan_a", engine_index=0, success=False, step_results=[]
        )

        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        fr_msg = msgspec.msgpack.encode(("faultreport", report))
        rpr0_msg = msgspec.msgpack.encode(
            ("recoveryplanresult", rpr_0)
        )

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pull.recv.side_effect = [fr_msg, rpr0_msg, StopIteration]

        poll_events = [
            {},
            {recv_pull: zmq.POLLIN},
            {},
            {recv_pull: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )
        fail_msg = msgspec.msgpack.encode(
            ("recoverycomplete", RecoveryComplete(
                plan_name="plan_a", success=False, current_wave=0,
            ))
        )
        recv_pub.send.assert_any_call(fail_msg)

    def test_recovery_plan_timeout(self):
        plan = RecoveryPlan(name="plan_b", timeout_s=0, steps=[])
        exp = ExceptionInfo(exception_type="ValueError", exception_msg="boom")
        report = FaultReport(worker_rank=0, engine_index=0, exp=exp, plan=plan)

        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        fr_msg = msgspec.msgpack.encode(("faultreport", report))

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pull.recv.side_effect = [fr_msg, StopIteration]

        poll_events = [
            {},
            {recv_pull: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )
        timeout_msg = msgspec.msgpack.encode(
            ("recoverycomplete", RecoveryComplete(
                plan_name="", success=False, current_wave=0,
            ))
        )
        recv_pub.send.assert_any_call(timeout_msg)

    def test_malformed_recovery_message(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        bad_msg = b"\xff\xfe\x00"

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pull.recv.side_effect = [bad_msg, StopIteration]

        poll_events = [
            {},
            {recv_pull: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )
        plan_msg = msgspec.msgpack.encode(
            ("recoveryplan", RecoveryPlan(name="ignored", timeout_s=0, steps=[]))
        )
        for c in recv_pub.send.call_args_list:
            self.assertNotEqual(c[0][0], plan_msg)

    def test_poll_timeout_publishes_stats(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]

        poll_events = [
            {},
            {},  # no events → stats publish
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )
        self.assertTrue(pub_front.send.call_count >= 1)

    def test_publish_back_non_sub_returns(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        pub_back.recv.side_effect = [b"\x01", b"bad_sub", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]

        poll_events = [
            {},
            {},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )

    def test_publish_back_bad_subscription_returns(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        pub_back.recv.side_effect = [b"\x01", b"\x01", b"weird", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]

        poll_events = [
            {},
            {pub_back: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )

    def test_publish_front_subscription_ignored(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        pub_front.recv.side_effect = [b"\x01", StopIteration]

        poll_events = [
            {},
            {pub_front: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )

    def test_publish_front_sends_start_wave(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        wave_msg = (-1, 0)

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        pub_front.recv.side_effect = [
            msgspec.msgpack.encode(wave_msg), StopIteration,
        ]

        poll_events = [
            {},
            {pub_front: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )

    def test_output_back_stats_update(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        fake_output = SimpleNamespace(
            outputs=[],
            utility_output=None,
            engine_index=0,
            scheduler_stats=SimpleNamespace(
                step_counter=1,
                current_wave=0,
                num_waiting_reqs=5,
                num_running_reqs=3,
            ),
            wave_complete=None,
            start_wave=None,
        )

        class FakeDecoder:
            def decode(self, buf):
                return fake_output

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        out_back.recv.return_value = b"fake"

        poll_events = [
            {},
            {out_back: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
            extra_patches=[
                patch(
                    "vllm_ascend.patch.platform.patch_recovery_coordinator"
                    ".MsgpackDecoder", return_value=FakeDecoder(),
                ),
            ],
        )

    def test_unknown_recovery_msg_type(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        unknown_msg = msgspec.msgpack.encode(("weird_type", {}))
        recv_pull.recv.side_effect = [unknown_msg, StopIteration]

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]

        poll_events = [
            {},
            {recv_pull: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )

    def test_run_coordinator_keyboard_interrupt(self):
        from vllm_ascend.patch.platform.patch_recovery_coordinator \
            import _patched_run_coordinator as run_c

        class FakeProc:
            process_input_socket = MagicMock(
                side_effect=KeyboardInterrupt
            )

        with patch(
            "vllm_ascend.patch.platform.patch_recovery_coordinator"
            ".DPCoordinatorProc", return_value=FakeProc(),
        ):
            run_c(2, "front", "back_out", "back_pub",
                  recovery_pub_address="rpub",
                  recovery_pull_address="rpull")

    def test_recovery_pub_bad_subscription_returns(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"bad", StopIteration]

        poll_events = [
            {},
            {},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )

    def test_publish_front_elastic_ep_scale_up(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        scale_msg = msgspec.msgpack.encode(("SCALE_ELASTIC_EP", 3))

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        pub_front.recv.side_effect = [scale_msg, StopIteration]

        poll_events = [
            {},
            {pub_front: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )

    def test_publish_front_elastic_ep_scale_down(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        scale_msg = msgspec.msgpack.encode(("SCALE_ELASTIC_EP", 1))

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        pub_front.recv.side_effect = [scale_msg, StopIteration]

        poll_events = [
            {},
            {pub_front: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
        )

    def test_output_back_wave_complete_advances_wave(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        fake_output = SimpleNamespace(
            outputs=[],
            utility_output=None,
            engine_index=0,
            scheduler_stats=None,
            wave_complete=1,
            start_wave=None,
        )

        class FakeDecoder:
            def decode(self, buf):
                return fake_output

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        out_back.recv.return_value = b"fake"

        poll_events = [
            {},
            {out_back: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
            enable_wave=True,
            extra_patches=[
                patch(
                    "vllm_ascend.patch.platform.patch_recovery_coordinator"
                    ".MsgpackDecoder", return_value=FakeDecoder(),
                ),
            ],
        )

    def test_output_back_start_wave_triggers_send(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        fake_output = SimpleNamespace(
            outputs=[],
            utility_output=None,
            engine_index=0,
            scheduler_stats=None,
            wave_complete=None,
            start_wave=1,
        )

        class FakeDecoder:
            def decode(self, buf):
                return fake_output

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        out_back.recv.return_value = b"fake"

        poll_events = [
            {},
            {out_back: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
            enable_wave=True,
            extra_patches=[
                patch(
                    "vllm_ascend.patch.platform.patch_recovery_coordinator"
                    ".MsgpackDecoder", return_value=FakeDecoder(),
                ),
            ],
        )

    def test_stats_out_of_order_warning(self):
        recv_pub = self._zmq_sock()
        recv_pull = self._zmq_sock()
        pub_back = self._zmq_sock()
        pub_front = self._zmq_sock()
        out_back = self._zmq_sock()

        first_stats = SimpleNamespace(
            outputs=[],
            utility_output=None,
            engine_index=0,
            scheduler_stats=SimpleNamespace(
                step_counter=2,
                current_wave=0,
                num_waiting_reqs=5,
                num_running_reqs=3,
            ),
            wave_complete=None,
            start_wave=None,
        )
        second_stats = SimpleNamespace(
            outputs=[],
            utility_output=None,
            engine_index=0,
            scheduler_stats=SimpleNamespace(
                step_counter=0,
                current_wave=0,
                num_waiting_reqs=2,
                num_running_reqs=1,
            ),
            wave_complete=None,
            start_wave=None,
        )

        class FakeDecoder:
            def __init__(self):
                self.calls = 0

            def decode(self, buf):
                self.calls += 1
                if self.calls == 1:
                    return first_stats
                return second_stats

        pub_back.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        recv_pub.recv.side_effect = [b"\x01", b"\x01", StopIteration]
        out_back.recv.return_value = b"fake"

        poll_events = [
            {},
            {out_back: zmq.POLLIN},
            {},
            {out_back: zmq.POLLIN},
            {},
        ]

        self._run_one_pass(
            recv_pull, recv_pub, pub_back, pub_front, out_back,
            poll_events,
            extra_patches=[
                patch(
                    "vllm_ascend.patch.platform.patch_recovery_coordinator"
                    ".MsgpackDecoder", return_value=FakeDecoder(),
                ),
            ],
        )
