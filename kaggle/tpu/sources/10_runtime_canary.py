"""Inspect real Kaggle TPU topology and PyTorch/XLA feasibility."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_backend(name, code):
    environment = os.environ.copy()
    environment.setdefault("PJRT_DEVICE", "TPU")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("SLAO_TPU_BACKEND "):
            payload = json.loads(line.removeprefix("SLAO_TPU_BACKEND "))
            break
    return {
        "name": name,
        "returncode": completed.returncode,
        "payload": payload,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


jax_probe = run_backend(
    "jax",
    """
import json
import jax
devices = jax.devices("tpu")
print("SLAO_TPU_BACKEND " + json.dumps({
    "backend": jax.default_backend(),
    "device_count": jax.device_count(),
    "local_device_count": jax.local_device_count(),
    "tpu_device_count": len(devices),
    "devices": [str(device) for device in devices],
}, sort_keys=True))
""",
)

torch_xla_probe = run_backend(
    "torch_xla",
    """
import json
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
supported = xm.get_xla_supported_devices()
device = xm.xla_device()
print("SLAO_TPU_BACKEND " + json.dumps({
    "torch_version": torch.__version__,
    "torch_xla_version": getattr(torch_xla, "__version__", None),
    "device_type": xr.device_type(),
    "supported_devices": supported,
    "supported_device_count": len(supported),
    "selected_device": str(device),
}, sort_keys=True))
""",
)

jax_payload = jax_probe.get("payload") or {}
xla_payload = torch_xla_probe.get("payload") or {}
jax_tpu_count = int(jax_payload.get("tpu_device_count") or 0)
xla_device_type = str(xla_payload.get("device_type") or "").upper()
xla_device_count = int(xla_payload.get("supported_device_count") or 0)
real_tpu_observed = jax_tpu_count > 0 or (xla_device_type == "TPU" and xla_device_count > 0)
pytorch_xla_ready = (
    torch_xla_probe["returncode"] == 0
    and xla_device_type == "TPU"
    and xla_device_count > 0
)

report = {
    "status": (
        "pytorch_xla_ready"
        if pytorch_xla_ready
        else "real_tpu_but_pytorch_xla_unavailable"
        if real_tpu_observed
        else "no_real_tpu"
    ),
    "evidence_class": "development_tpu_canary",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "requested_shape": "TpuV5E8",
    "real_tpu_observed": real_tpu_observed,
    "pytorch_xla_ready": pytorch_xla_ready,
    "jax": jax_probe,
    "torch_xla": torch_xla_probe,
}
output = Path("/kaggle/working/tpu_runtime_canary.json")
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("SLAO_TPU_RUNTIME_CANARY " + json.dumps(report, sort_keys=True))
if not real_tpu_observed:
    raise RuntimeError("TpuV5E8 was requested but no TPU backend exposed devices")
