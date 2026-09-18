"""Throwaway diagnostic for the base-group render-staleness bug (not part of
the test suite or the public API) -- delete once the bug is resolved.

Isolated cube+hue is dead (max=0.0) while arm+hue works (max=86). The
cube's kinematic root-pose write is the culprit, not the arm -- matching
IsaacLab PR #7587 ("shared kinematic rigid-object renderer contract"),
which exists because kinematic RigidObjects need different renderer-sync
handling than articulations. Testing whether writing hue *after* the
cube's pose write + step() (instead of before, which is what
write_latent_state() currently does) survives -- bypassing write_state()
to control the exact order by hand.

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
from idtb.sim.writer import _write_hue  # noqa: E402

spec = LatentSpec(
    (
        Handle("cube.x", 0.0, 0.3, group="base"),
        Handle("cube.y", 0.0, 0.3, group="base"),
        Handle("cube.hue", 0.5, 0.5, group="full"),
    )
)

backend = IsaacSceneBackend()
backend.bind(spec)
rig = backend._rig


def write_cube_pose_then_step(x: float, y: float) -> None:
    root = rig.cube.data.default_root_state.clone()
    root[:, 0] = x
    root[:, 1] = y
    root[:, 2] = rig.ground_z + 0.5 * rig.default_cube_edge_m
    root[:, 0:3] += rig.scene.env_origins
    root[:, 7:] = 0.0
    rig.cube.write_root_pose_to_sim(root[:, :7])
    rig.cube.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))
    rig.scene.write_data_to_sim()
    rig.sim.step(render=False)


write_cube_pose_then_step(0.0, 0.0)
_write_hue(rig.cube_shader, 0.5)
backend.render(0)
frame_base = backend.render(1)["cam0"]["rgb"].clone()

write_cube_pose_then_step(0.0, 0.0)  # same pose again, matching a real sample
_write_hue(rig.cube_shader, 0.8808)
backend.render(0)
frame_hue = backend.render(1)["cam0"]["rgb"].clone()

d = (frame_hue.float() - frame_base.float()).abs()
print(f"FRAME_DIFF max={d.max().item():.4f} mean={d.mean().item():.6f}", flush=True)
