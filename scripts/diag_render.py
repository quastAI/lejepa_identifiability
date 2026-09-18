"""Throwaway diagnostic for the base-group render-staleness bug (not part of
the test suite or the public API) -- delete once the bug is resolved.

Isolated hue-only + sim.step() probe: earlier multi-case runs regressed
`cube.hue`'s signal to near-zero once `sim.step()` was added, but that
test cycled through base/arm/cube/hue sequentially, each write resetting
every other dim back to center -- by the 4th case there's a chain of
reversions in play, not a clean single-variable comparison. This uses a
spec with *only* `cube.hue` bound, two states, nothing else touched, to
tell whether `sim.step()` genuinely kills the material signal or whether
that was a cross-case artifact.

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

spec = LatentSpec((Handle("cube.hue", 0.5, 0.5, group="full"),))

backend = IsaacSceneBackend()
backend.bind(spec)

phi_a = spec.squash(torch.zeros(1, 1))
phi_b = spec.squash(torch.ones(1, 1))

backend.write_state(phi_a)
backend._rig.sim.step(render=False)
read_a = backend.read_state()
backend.render(0)
frame_a = backend.render(1)["cam0"]["rgb"].clone()

backend.write_state(phi_b)
backend._rig.sim.step(render=False)
read_b = backend.read_state()
backend.render(0)
frame_b = backend.render(1)["cam0"]["rgb"].clone()

print("READ_A", read_a, flush=True)
print("READ_B", read_b, flush=True)
d = (frame_a.float() - frame_b.float()).abs()
print(f"FRAME_DIFF max={d.max().item():.4f} mean={d.mean().item():.6f}", flush=True)
