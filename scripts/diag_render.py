"""Throwaway diagnostic for the base-group render-staleness bug (not part of
the test suite or the public API) -- delete once the bug is resolved.

Per isaac-sim/IsaacLab#6394: "poses written through the PhysX tensor API
only reach the renderer during a physics step" -- `forward()`'s
`update_articulations_kinematic()` + `_update_fabric(0.0, 0.0)` is
apparently not equivalent to what `step()`'s `_physx_sim.simulate(dt, 0.0)`
+ `_physx_sim.fetch_results()` does. This tests a real `sim.step()` and
measures how much it drifts `read_state()` away from what we wrote (the
cube is `kinematic_enabled=True` so gravity-immune; the arm's PD control
is the one at risk of correcting toward a stale target).

Run: /workspace/isaaclab/isaaclab.sh -p scripts/diag_render.py 2>&1 | grep -E "READ_|FRAME_|DRIFT"
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
arm_z = base_z.clone()
arm_z[0, 0:5] = 1.0
cube_z = base_z.clone()
cube_z[0, 5:7] = 1.0
hue_z = base_z.clone()
hue_z[0, 7] = 1.0

cases = {"base": base_z, "arm_only": arm_z, "cube_only": cube_z, "hue_only": hue_z}
frames = {}
for name, z in cases.items():
    phi = spec.squash(z)
    backend.write_state(phi)
    before = backend.read_state()
    backend._rig.sim.step(render=False)
    after = backend.read_state()
    drift = (after - before).abs().max().item()
    print(f"DRIFT_{name}", drift, flush=True)
    print(f"READ_{name}_after_step", after, flush=True)
    for _ in range(7):  # extra discarded settle renders (material recompile?)
        backend.render(0)
    frames[name] = backend.render(1)["cam0"]["rgb"].clone()

base_frame = frames["base"].float()
for name in ("arm_only", "cube_only", "hue_only"):
    d = (frames[name].float() - base_frame).abs()
    print(f"FRAME_{name}_vs_base max={d.max().item():.4f} mean={d.mean().item():.6f}", flush=True)
