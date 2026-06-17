# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tests.e2e.conftest import RemoteOpenAIServer
from vllm.utils.network_utils import get_open_port

RECOVERY_MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="function")
def recovery_enabled_server():
    port = get_open_port()
    server_args = [
        "--max-model-len", "1024",
        "--port", str(port),
        "--additional-config", json.dumps({
            "recovery_config": {"enable": True, "cpu_process_group_timeout": 30}
        }),
    ]

    with RemoteOpenAIServer(RECOVERY_MODEL, server_args, auto_port=False) as server:
        yield server
