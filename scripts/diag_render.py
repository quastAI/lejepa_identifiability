"""Throwaway diagnostic for the base-group render-staleness bug (not part of
the test suite or the public API) -- delete once the bug is resolved.

Splits the previous all-dims-at-once probe into three cases -- arm-only,
cube-only, and both -- to tell whether the articulation (arm) and the
kinematic-enabled RigidObject (cube) are stuck for the same reason or two
different ones: `update_articulations_kinematic()` (found by grepping
isaaclab_physx's `physx_manager.py`) is articulation-specific and gated on
`sim.is_playing()`, so it may fix the arm without touching the cube at all.

Run: /workspace/isaaclab/isaaclab.sh -p scripts/diag_render.py 2>&1 | grep -E "READ_|FRAME_|PLAYING"
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
    )
)

backend = IsaacSceneBackend()
backend.bind(spec)
print("PLAYING", backend._rig.sim.is_playing(), flush=True)

base_z = torch.zeros(1, spec.n)
arm_z = base_z.clone()
arm_z[0, 0:5] = 1.0  # arm.j0..j3 + gripper.aperture
cube_z = base_z.clone()
cube_z[0, 5:7] = 1.0  # cube.x, cube.y
both_z = base_z.clone()
both_z[0, :] = 1.0

cases = {"base": base_z, "arm_only": arm_z, "cube_only": cube_z, "both": both_z}
frames = {}
reads = {}
for name, z in cases.items():
    phi = spec.squash(z)
    backend.write_state(phi)
    reads[name] = backend.read_state()
    frames[name] = backend.render(0)["cam0"]["rgb"].clone()
    print(f"READ_{name}", reads[name], flush=True)

base_frame = frames["base"].float()
for name in ("arm_only", "cube_only", "both"):
    diff = (frames[name].float() - base_frame).abs().max().item()
    print(f"FRAME_DIFF_{name}_vs_base", diff, flush=True)
