"""What does a Kaggle TPU actually look like to torch_xla? Diagnostic kernel.

`xmp.spawn(nprocs=None)` -- the documented way to get one process per core --
died here with:

    TPU initialization failed: Invalid --<id>_slice_builder_worker_addresses
    specified. Expected 8 worker addresses, got 1.

That is libtpu topology configuration, not anything in the training code, so
the next step is to find out what this host reports rather than try another
spawn incantation and hope. Cheap: a few minutes of the ~20 h weekly quota.

Prints the runtime's own view (version, device count, addressable devices, the
TPU_*/PJRT_*/XLA_* environment), then tries three ways to get work onto the
chips, each isolated so one failure does not hide the next:

  A. single process, one device        -- the floor; if this fails nothing works
  B. single process, ALL devices       -- 8 independent models in one process,
                                          which would suit this job better than
                                          spawn anyway (no multiprocessing, and
                                          XLA enqueues to devices asynchronously)
  C. xmp.spawn                         -- the path that just failed, re-run here
                                          with the error captured rather than
                                          taking down the kernel
"""

import os
import traceback

import torch

print("=" * 70, flush=True)
for k in sorted(os.environ):
    if any(k.startswith(p) for p in ("TPU", "PJRT", "XLA", "LIBTPU", "XRT",
                                     "CLOUD_TPU", "GRPC")):
        print(f"windml env {k}={os.environ[k]}", flush=True)
print("=" * 70, flush=True)

import torch_xla
import torch_xla.core.xla_model as xm

print(f"windml torch={torch.__version__} torch_xla={torch_xla.__version__}", flush=True)

try:
    import torch_xla.runtime as xr
    print(f"windml global_device_count={xr.global_runtime_device_count()} "
          f"addressable={xr.addressable_runtime_device_count()} "
          f"world={xr.world_size()} ordinal={xr.global_ordinal()}", flush=True)
except Exception:
    print("windml runtime_query_failed:", flush=True)
    traceback.print_exc()

try:
    print(f"windml xla_supported_devices={xm.get_xla_supported_devices()}", flush=True)
except Exception:
    print("windml get_xla_supported_devices_failed:", flush=True)
    traceback.print_exc()


def attempt(name, fn):
    print("-" * 70, flush=True)
    print(f"windml attempt={name}", flush=True)
    try:
        fn()
        print(f"windml attempt={name} RESULT=ok", flush=True)
    except Exception as exc:
        print(f"windml attempt={name} RESULT=fail {type(exc).__name__}: "
              f"{str(exc)[:400]}", flush=True)


def one_device():
    dev = torch_xla.device()
    a = torch.randn(256, 256, device=dev)
    b = (a @ a).sum()
    xm.mark_step()
    print(f"windml   device={dev} matmul_sum={float(b):.3f}", flush=True)


def all_devices():
    """8 tiny independent models, one per device, stepped from one process.

    This is the shape the ensemble run wants: no gradient is shared, so nothing
    needs multiprocessing -- it only needs eight devices reachable from here.
    """
    devs = xm.get_xla_supported_devices()
    print(f"windml   devices={devs}", flush=True)
    nets, opts = [], []
    for d in devs:
        n = torch.nn.Sequential(torch.nn.Conv2d(4, 8, 3, padding=1),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(8, 2, 3, padding=1)).to(d)
        nets.append(n)
        opts.append(torch.optim.Adam(n.parameters(), lr=1e-4))
    for step in range(3):
        losses = []
        for d, n, o in zip(devs, nets, opts):
            x = torch.randn(4, 4, 32, 64, device=d)
            y = torch.randn(4, 2, 32, 64, device=d)
            loss = ((n(x) - y) ** 2).mean()
            o.zero_grad(set_to_none=True)
            loss.backward()
            o.step()
            losses.append(loss)
        xm.mark_step()
        print(f"windml   step={step} losses="
              f"{[round(float(l), 4) for l in losses]}", flush=True)


def spawned():
    import torch_xla.distributed.xla_multiprocessing as xmp

    def body(rank):
        dev = torch_xla.device()
        a = torch.randn(64, 64, device=dev)
        xm.mark_step()
        print(f"windml   spawn rank={rank} dev={dev} ok", flush=True)

    xmp.spawn(body, args=())


attempt("A_single_device", one_device)
attempt("B_all_devices_one_process", all_devices)
attempt("C_xmp_spawn", spawned)
print("=" * 70, flush=True)
print("RESULT OK")
