#!/usr/bin/env bash
# Prepare the GTX 1080 box for RT2021 pretraining. Linux, native, non-interactive.
#
# Intended to be run over SSH by an orchestrating agent, so it never prompts,
# is safe to re-run after a partial failure, and reports state in a parseable
# form: every fact is a "windml KEY=VALUE" line and the last line is either
# "RESULT OK" or "RESULT FAIL <reason>". Exit status matches.
#
# The load-bearing part is the Pascal gate. Recent PyTorch wheels dropped the
# sm_61 kernels a GTX 1080 needs, and the failure is quiet: torch imports,
# torch.cuda.is_available() returns True, and then real kernels fail or fall
# back. So rather than trusting any particular wheel to be correct, this runs
# an actual CUDA matmul and checks sm_61 is in the compiled arch list. Better
# to fail here in seconds than three hours into a pretraining run.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

VENV="${WINDML_VENV:-$REPO/.venv-gpu}"
TORCH_INDEX="${WINDML_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
MIN_DISK_GB="${WINDML_MIN_DISK_GB:-150}"

say()  { echo "windml $*"; }
fail() { echo "RESULT FAIL $*"; exit 1; }

say "repo=$REPO"
say "venv=$VENV"

# --- preflight: hardware -----------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found; install the NVIDIA driver"
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
[ -n "$gpu_name" ] || fail "nvidia-smi returned no GPU"
say "gpu=$gpu_name"
say "driver=$driver"
say "vram_mb=$vram"

# --- preflight: disk ---------------------------------------------------------
# The CMIP archive is the reason for the threshold. Exact size is unknown (the
# TUM server sends no Content-Length for these zips); the estimate is 40-120 GB
# at 5.625 degrees, and unpacked netCDF needs headroom on top of the zips.
avail_gb=$(df -PBG "$REPO" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
say "disk_avail_gb=$avail_gb"
if [ "${avail_gb:-0}" -lt "$MIN_DISK_GB" ]; then
  fail "only ${avail_gb}GB free, want >=${MIN_DISK_GB}GB for CMIP (override WINDML_MIN_DISK_GB, or fetch fewer years with scripts/fetch_cmip.py --years)"
fi

# --- toolchain ---------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  say "installing_uv=1"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || fail "uv install failed"
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || fail "uv still not on PATH after install (try: export PATH=\$HOME/.local/bin:\$PATH)"
say "uv=$(uv --version 2>/dev/null)"

[ -d "$VENV" ] || uv venv "$VENV" >/dev/null 2>&1 || fail "could not create venv at $VENV"
PY="$VENV/bin/python"

say "installing_torch_from=$TORCH_INDEX"
uv pip install --python "$PY" --index-url "$TORCH_INDEX" torch >/dev/null 2>&1 \
  || fail "torch install failed from $TORCH_INDEX"
uv pip install --python "$PY" -e ".[dev]" >/dev/null 2>&1 \
  || uv pip install --python "$PY" -e . >/dev/null 2>&1 \
  || fail "editable install of windml failed"

# --- the Pascal gate ---------------------------------------------------------
"$PY" - <<'PYGATE'
import sys
import torch

print(f"windml torch={torch.__version__}")
print(f"windml cuda_build={torch.version.cuda}")

if not torch.cuda.is_available():
    print("RESULT FAIL torch cannot see the GPU (driver/CUDA mismatch)")
    sys.exit(1)

arch_list = torch.cuda.get_arch_list()
cap = torch.cuda.get_device_capability(0)
print(f"windml arch_list={','.join(arch_list)}")
print(f"windml device_capability=sm_{cap[0]}{cap[1]}")

needed = f"sm_{cap[0]}{cap[1]}"
if needed not in arch_list:
    # This is the silent killer: it imports fine and reports a GPU, then dies
    # or silently misbehaves on the first real kernel.
    print(
        f"RESULT FAIL this torch has no {needed} kernels (built for "
        f"{','.join(arch_list)}). Reinstall from a cu126 or older index: "
        f"uv pip install --index-url https://download.pytorch.org/whl/cu126 torch"
    )
    sys.exit(1)

# Trust nothing: run a real kernel and check the numbers.
try:
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    got = (a @ b).cpu()
    torch.testing.assert_close(got, a.cpu() @ b.cpu(), atol=1e-3, rtol=1e-3)
except Exception as exc:  # noqa: BLE001 - want the message, whatever it is
    print(f"RESULT FAIL CUDA matmul failed on this GPU: {type(exc).__name__}: {exc}")
    sys.exit(1)

free, total = torch.cuda.mem_get_info()
print(f"windml cuda_matmul=ok")
print(f"windml vram_free_gb={free/2**30:.1f}")
PYGATE
[ $? -eq 0 ] || exit 1

# --- correctness -------------------------------------------------------------
"$PY" -m pytest -q >/tmp/windml-pytest.log 2>&1 \
  || { tail -20 /tmp/windml-pytest.log; fail "pytest failed (full log /tmp/windml-pytest.log)"; }
say "pytest=$(tail -2 /tmp/windml-pytest.log | tr -d '\n' | tail -c 60)"

say "python=$PY"
echo "RESULT OK"
