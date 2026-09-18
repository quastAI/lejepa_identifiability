"""Throwaway diagnostic for the base-group render-staleness bug (not part of
the test suite or the public API) -- delete once the bug is resolved.

The isolated single-dim hue-only + sim.step() test just PASSED cleanly
(max=113, mean=0.074 -- real signal, close to the original no-step
baseline of 120/0.148). So sim.step() + material write are NOT
fundamentally incompatible. The earlier 4-case sequential test (base ->
arm_only -> cube_only -> hue_only, full 8-dim spec, each case a real
sim.step()) regressed hue to near-zero. This test isolates *why*: does
having arm+cube dims in the same spec as hue matter (spec composition),
or does it take multiple prior sim.step() calls to degrade the signal
(call-count accumulation)? Two cases only: base -> hue_only, same 8-dim
spec, skipping arm_only/cube_only entirely.

Run: /workspace/isaaclab/isaaclab.sh -p scripts/diag_render.py 2>&1 | grep -E "READ_|FRAME_"
"""

import argparse

import torch
from isaaclab.app import AppLauncher

from idtb.sim.app import launch

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app = launch(args)

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
        Handle("cube.hue", 0.5, 0.5, group="full"),
    )
)

backend = IsaacSceneBackend()
backend.bind(spec)

n = spec.n
base_z = torch.zeros(1, n)
hue_z = base_z.clone()
hue_z[0, 7] = 1.0

phi_base = spec.squash(base_z)
phi_hue = spec.squash(hue_z)

backend.write_state(phi_base)
backend._rig.sim.step(render=False)
read_base = backend.read_state()
backend.render(0)
frame_base = backend.render(1)["cam0"]["rgb"].clone()

backend.write_state(phi_hue)
backend._rig.sim.step(render=False)
read_hue = backend.read_state()
backend.render(0)
frame_hue = backend.render(1)["cam0"]["rgb"].clone()

print("READ_base", read_base, flush=True)
print("READ_hue", read_hue, flush=True)
d = (frame_base.float() - frame_hue.float()).abs()
print(f"FRAME_DIFF max={d.max().item():.4f} mean={d.mean().item():.6f}", flush=True)
