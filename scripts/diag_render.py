"""Throwaway diagnostic for the base-group render-staleness bug (not part of
the test suite or the public API) -- delete once the bug is resolved.

Writes two well-separated `base`-group states directly through
`IsaacSceneBackend`, bypassing pytest, and prints read-back + frame stats for
both so we can tell whether the physics write reaches Fabric (`READ_DIFF`)
independently of whether it reaches the renderer (`FRAME_DIFF`).

Run: /workspace/isaaclab/isaaclab.sh -p scripts/diag_render.py 2>&1 | grep -E "READ_|FRAME_"
"""

import argparse
import sys

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
saved_argv = sys.argv
sys.argv = saved_argv[:1]
app = AppLauncher(args).app
sys.argv = saved_argv

from idtb.latents import Handle, LatentSpec  # noqa: E402
from idtb.sim.scene import IsaacSceneBackend  # noqa: E402

spec = LatentSpec(
    (
        Handle("arm.j0", 0.0, 1.5, group="base"),
        Handle("arm.j1", 0.0, 1.0, group="base"),
        Handle("arm.j2", -1.5, 1.5, group="base"),
        Handle("arm.j3", 1.8, 1.8, group="base"),
        Handle("gripper.aperture", 0.02, 0.02, group="base"),
        Handle("cube.x", 0.0, 0.3, group="base"),
        Handle("cube.y", 0.0, 0.3, group="base"),
    )
)

backend = IsaacSceneBackend()
backend.bind(spec)

phi_a = spec.squash(torch.zeros(1, spec.n))
phi_b = spec.squash(torch.ones(1, spec.n))

backend.write_state(phi_a)
read_a = backend.read_state()
frame_a = backend.render(0)["cam0"]["rgb"].clone()

backend.write_state(phi_b)
read_b = backend.read_state()
frame_b = backend.render(1)["cam0"]["rgb"].clone()

print("READ_A", read_a, flush=True)
print("READ_B", read_b, flush=True)
print("READ_DIFF", (read_a - read_b).abs().max().item(), flush=True)
print("FRAME_A stats", frame_a.float().mean().item(), frame_a.float().std().item(), flush=True)
print("FRAME_B stats", frame_b.float().mean().item(), frame_b.float().std().item(), flush=True)
print("FRAME_DIFF", (frame_a.float() - frame_b.float()).abs().max().item(), flush=True)
