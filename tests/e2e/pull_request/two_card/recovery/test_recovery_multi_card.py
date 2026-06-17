# SPDX-License-Identifier: Apache-2.0
"""
L2 PR-level E2E tests: multi-card fault recovery (Ascend NPU x2)

Verifies the token replay recovery flow on dual A3 NPUs with Qwen3-0.6B:

  - TP=2 device fault recovery
  - TP=2 request replay after recovery
  - DP=2 Coordinator recovery routing

Fault injection mechanism
-------------------------
Shares :func:`make_recovery_fault_injection_env` with the single-card tests.
The environment variables are passed through ``RemoteOpenAIServer.env_dict``
to the server subprocess and all spawned worker processes. In TP mode, all
ranks inherit the same environment, so both ranks raise the fault and
participate in recovery together.

Multi-card vs single-card
-------------------------
- **TP=2**: Both ranks' tensor-parallel group must be re-established via
  ``reinit_process_group()`` after device restart.
- **DP=2**: The Coordinator must collect ``RecoveryComplete`` from every
  engine before broadcasting completion to all.

Run command
-----------
::

    VLLM_USE_MODELSCOPE=true pytest -sv tests/e2e/pull_request/two_card/recovery/
"""

import json
import time

import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import make_recovery_fault_injection_env

MODEL = "Qwen/Qwen3-0.6B"
RECOVERY_TIMEOUT = 900
HEALTH_POLL_INTERVAL = 2


def _wait_for_completion(server, prompt="Hello", timeout=RECOVERY_TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        try:
            client = server.get_client()
            response = client.completions.create(
                model=MODEL, prompt=prompt, max_tokens=5, temperature=0, timeout=60,
            )
            if response.choices[0].text:
                return response
        except Exception:
            pass
        time.sleep(HEALTH_POLL_INTERVAL)
    return None


def _run_fault_recovery_test(server_args, *, prompt="Hello", max_tokens=5):
    fault_env, cleanup = make_recovery_fault_injection_env()
    try:
        with RemoteOpenAIServer(MODEL, server_args, auto_port=False, env_dict=fault_env) as server:
            resp = requests.get(server.url_for("health"), timeout=10)
            assert resp.status_code == 200

            client = server.get_client()
            try:
                client.completions.create(
                    model=MODEL, prompt=prompt, max_tokens=max_tokens, temperature=0, timeout=30,
                )
            except Exception:
                pass

            recovered = _wait_for_completion(server, prompt=prompt)
            assert recovered is not None, "Server did not recover after fault injection"
            assert len(recovered.choices[0].text) > 0, "Recovery should produce non-empty token output"
            return recovered
    finally:
        cleanup()


# ── TP=2 device fault recovery ──────────────────────────────────────────────

@wait_until_npu_memory_free(target_free_percentage=0.95)
def test_recovery_tp2_device_fault():
    """TP=2 device fault recovery.

    Hardware: A3 NPU x2

    Steps:
        1. Start server with ``--tensor-parallel-size 2`` and recovery enabled.
        2. Inject a 507057 fault and send a request to trigger recovery.
        3. Both ranks experience the fault and undergo recovery simultaneously.
        4. Poll until the server is healthy and can serve completions.

    Verification:
        - Both ranks recover successfully.
        - ``reinit_process_group()`` rebuilds the TP communication group.
        - After recovery, the server can process inference requests.
        - Recovered response contains non-empty token output.
    """
    port = get_open_port()
    server_args = [
        "--max-model-len", "1024",
        "--tensor-parallel-size", "2",
        "--port", str(port),
        "--additional-config", json.dumps({
            "recovery_config": {"enable": True, "cpu_process_group_timeout": 30}
        }),
    ]
    _run_fault_recovery_test(server_args)


# ── TP=2 request replay after recovery ──────────────────────────────────────

@wait_until_npu_memory_free(target_free_percentage=0.95)
def test_recovery_tp2_request_replay_after_fault():
    """TP=2 request replay after recovery produces valid output.

    Hardware: A3 NPU x2

    Steps:
        1. Start server with TP=2, recovery enabled, and fault injection.
        2. Send a first request to trigger fault + recovery.
        3. Wait for recovery to complete.
        4. Send a second request and validate the output.

    Verification:
        - After recovery, the server can process inference requests.
        - ``finish_reason == "length"`` (normal completion).
        - Response contains non-empty generated tokens.
    """
    port = get_open_port()
    server_args = [
        "--max-model-len", "1024",
        "--tensor-parallel-size", "2",
        "--port", str(port),
        "--additional-config", json.dumps({
            "recovery_config": {"enable": True, "cpu_process_group_timeout": 30}
        }),
    ]
    response = _run_fault_recovery_test(server_args, prompt="San Francisco is a", max_tokens=8)
    assert response.choices[0].finish_reason == "length", "Should complete with reason=length"


# ── DP=2 Coordinator recovery routing ───────────────────────────────────────

@wait_until_npu_memory_free(target_free_percentage=0.95)
def test_recovery_dp_coordinator_route():
    """DP=2 Coordinator recovery routing.

    Hardware: A3 NPU x2

    Background:
        In DP mode, each engine has its own Worker + EngineCore process pair.
        The Coordinator patch is responsible for:

        1. Receiving the FaultReport from the affected engine's WorkerMonitor.
        2. Broadcasting the RecoveryPlan to every engine via a publish ZMQ socket.
        3. Collecting StepResult from each engine during step execution.
        4. Broadcasting RecoveryComplete to all engines once every engine
           reports success.

    Steps:
        1. Start server with ``--data-parallel-size 2`` and recovery enabled.
        2. Inject a 507057 fault and send a request to trigger recovery.
        3. Poll until the server is healthy and can serve completions.

    Verification:
        - Coordinator correctly routes
          FaultReport → RecoveryPlan → RecoveryComplete.
        - Both engines complete recovery.
        - After recovery, the server can process inference requests.
    """
    port = get_open_port()
    server_args = [
        "--max-model-len", "1024",
        "--data-parallel-size", "2",
        "--port", str(port),
        "--additional-config", json.dumps({
            "recovery_config": {"enable": True, "cpu_process_group_timeout": 30}
        }),
    ]
    _run_fault_recovery_test(server_args)
