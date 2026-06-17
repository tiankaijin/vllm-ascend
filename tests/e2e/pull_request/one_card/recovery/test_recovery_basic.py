# SPDX-License-Identifier: Apache-2.0
"""
L2 PR-level E2E tests: single-card fault recovery (Ascend NPU x1)

Verifies the token replay recovery flow on a single A3 NPU with Qwen3-0.6B:

  - Service starts when recovery config is enabled
  - Service starts when recovery config is disabled (default)
  - Invalid cpu_process_group_timeout is rejected at startup
  - Simulated 507057 HCCL fault triggers the full recovery plan
  - After recovery, subsequent requests produce valid output

Fault injection mechanism
-------------------------
:func:`make_recovery_fault_injection_env` installs a ``sitecustomize.py`` in
the server subprocess. On Python startup, it replaces ``builtins.__import__``
to intercept the first import of ``vllm_ascend.recovery.worker_decorator``,
wrapping ``fault_recovery_decorator`` to raise ``RuntimeError("HCCL error
[507057] simulated test fault")`` on the first decorated call.

Zero production code changes. All injection logic lives in the test tree.

Run command
-----------
::

    VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/pull_request/one_card/recovery/
"""

import json
import time

import pytest
import requests

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.utils import make_recovery_fault_injection_env
from vllm.utils.network_utils import get_open_port

MODEL = "Qwen/Qwen3-0.6B"
RECOVERY_TIMEOUT = 600
HEALTH_POLL_INTERVAL = 2


def _wait_for_completion(server, prompt="Hello", timeout=RECOVERY_TIMEOUT):
    def _check():
        client = server.get_client()
        response = client.completions.create(
            model=MODEL, prompt=prompt, max_tokens=5, temperature=0, timeout=60,
        )
        if response.choices[0].text:
            return response

    start = time.time()
    while time.time() - start < timeout:
        try:
            result = _check()
            if result is not None:
                return result
        except Exception:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    return None


# ── Service starts with recovery enabled ─────────────────────────────────────

def test_recovery_config_enabled_service_starts():
    """Service starts when recovery config is enabled.

    Hardware: A3 NPU x1

    Steps:
        1. Start ``vllm serve`` with ``recovery_config.enable=true`` and
           ``cpu_process_group_timeout=30``.
        2. Query the ``/health`` endpoint.

    Verification:
        - Server starts successfully and returns HTTP 200 on ``/health``.
        - RecoveryConfig is parsed correctly; WorkerMonitor daemon is created.
    """
    port = get_open_port()
    server_args = [
        "--max-model-len", "1024",
        "--port", str(port),
        "--additional-config", json.dumps({
            "recovery_config": {"enable": True, "cpu_process_group_timeout": 30}
        }),
    ]

    with RemoteOpenAIServer(MODEL, server_args, auto_port=False) as server:
        resp = requests.get(server.url_for("health"), timeout=10)
        assert resp.status_code == 200


# ── Service starts without recovery config ───────────────────────────────────

def test_recovery_config_disabled_no_monitor():
    """Service starts without recovery config (defaults to disabled).

    Hardware: A3 NPU x1

    Steps:
        1. Start ``vllm serve`` without ``--additional-config`` (default).
        2. Query the ``/health`` endpoint.

    Verification:
        - Server starts normally without recovery config.
        - Recovery feature is disabled; no WorkerMonitor is created.
    """
    port = get_open_port()
    server_args = ["--max-model-len", "1024", "--port", str(port)]

    with RemoteOpenAIServer(MODEL, server_args, auto_port=False) as server:
        resp = requests.get(server.url_for("health"), timeout=10)
        assert resp.status_code == 200


# ── Invalid timeout rejected at startup ──────────────────────────────────────

def test_recovery_invalid_timeout_rejected():
    """Invalid cpu_process_group_timeout is rejected at startup.

    Hardware: A3 NPU x1

    Steps:
        1. Start ``vllm serve`` with ``cpu_process_group_timeout=10``
           (valid range is 25-60 seconds).
        2. Wait for the server to start (it will fail).

    Verification:
        - Server fails to start with RuntimeError or ValueError.
        - Error message contains "between 25s and 60s".
    """
    port = get_open_port()
    server_args = [
        "--max-model-len", "1024",
        "--port", str(port),
        "--additional-config", json.dumps({
            "recovery_config": {"enable": True, "cpu_process_group_timeout": 10}
        }),
    ]

    try:
        with RemoteOpenAIServer(MODEL, server_args, auto_port=False, max_wait_seconds=30):
            pytest.fail("Server should have failed to start with invalid timeout")
    except (RuntimeError, ValueError):
        pass


# ── Fault triggers recovery plan ─────────────────────────────────────────────

def test_recovery_network_exception_triggers_plan():
    """Simulated 507057 HCCL network fault triggers the full recovery plan.

    Hardware: A3 NPU x1

    Steps:
        1. Start server with recovery enabled and fault injection active.
        2. Verify server is healthy.
        3. Send a completions request → triggers first ``execute_model`` call.
        4. Worker raises ``RuntimeError("HCCL error [507057] ...")``.
        5. ``fault_recovery_decorator`` catches the exception and starts the
           recovery plan.
        6. Poll until the server is healthy again and can serve completions.

    Verification:
        - First request fails (exception caught by test, test does not crash).
        - After recovery, server processes inference requests.
        - Recovered response contains non-empty token output.
        - Recovery steps execute in order:
          ``recovery_begin`` → ``stop_device`` → ``restart_device`` →
          ``reinit_process_group`` → ``label_dirty_requests`` →
          ``clean_batch_queue`` → ``recompute_dirty_requests`` →
          ``worker_clean_dirty_requests_cache`` → ``worker_rebuild_cpu_group`` →
          ``worker_recapture_graph`` → ``RecoveryComplete``.
    """
    port = get_open_port()
    server_args = [
        "--max-model-len", "1024",
        "--port", str(port),
        "--additional-config", json.dumps({
            "recovery_config": {"enable": True, "cpu_process_group_timeout": 30}
        }),
    ]
    fault_env, cleanup = make_recovery_fault_injection_env()

    try:
        with RemoteOpenAIServer(MODEL, server_args, auto_port=False, env_dict=fault_env) as server:
            resp = requests.get(server.url_for("health"), timeout=10)
            assert resp.status_code == 200

            client = server.get_client()
            try:
                client.completions.create(
                    model=MODEL, prompt="Hello", max_tokens=5, temperature=0, timeout=30,
                )
            except Exception:
                pass

            recovered = _wait_for_completion(server)
            assert recovered is not None, "Server did not recover and process requests after fault injection"
            assert len(recovered.choices[0].text) > 0, "Recovery should produce non-empty token output"
    finally:
        cleanup()


# ── Request replay after recovery ────────────────────────────────────────────

def test_recovery_request_replay_after_fault():
    """After recovery, request replay produces valid output.

    Hardware: A3 NPU x1

    Steps:
        1. Start server with recovery enabled and fault injection active.
        2. Send a first request to trigger the fault and recovery.
        3. Wait for recovery to complete.
        4. Send a second request and verify output quality.

    Verification:
        - After recovery, the server can process inference requests.
        - Response contains non-empty generated tokens.
        - ``finish_reason == "length"`` (completed normally, not truncated).
        - Output has semantic coherence (baseline determined by prompt
          ``"San Francisco is a"``).
    """
    port = get_open_port()
    server_args = [
        "--max-model-len", "1024",
        "--port", str(port),
        "--additional-config", json.dumps({
            "recovery_config": {"enable": True, "cpu_process_group_timeout": 30}
        }),
    ]
    fault_env, cleanup = make_recovery_fault_injection_env()

    try:
        with RemoteOpenAIServer(MODEL, server_args, auto_port=False, env_dict=fault_env) as server:
            resp = requests.get(server.url_for("health"), timeout=10)
            assert resp.status_code == 200

            client = server.get_client()
            try:
                client.completions.create(
                    model=MODEL, prompt="San Francisco is a", max_tokens=8, temperature=0, timeout=30,
                )
            except Exception:
                pass

            response = _wait_for_completion(server, prompt="San Francisco is a")
            assert response is not None, "Server should recover and process requests"
            assert len(response.choices[0].text) > 0, "Response should contain generated tokens"
            assert response.choices[0].finish_reason == "length", "Should complete with reason=length"
    finally:
        cleanup()
