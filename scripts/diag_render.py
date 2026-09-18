"""Throwaway diagnostic for the base-group render-staleness bug (not part of
the test suite or the public API) -- delete once the bug is resolved.

arm_only/cube_only are now fixed (real sim.step() in write_latent_state)
but hue_only still collapses to near-zero signal in the full 8-dim spec,
even with the anim-time epsilon hack removed. Isolating which role is
the actual culprit: cube.x/cube.y (root_state) alone with hue, vs
arm joints (joint) alone with hue -- using bind() to rebind the SAME
backend/rig to two different mini-specs, no need to rebuild the scene.

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

backend = IsaacSceneBackend()

cube_hue_spec = LatentSpec(
    (
        Handle("cube.x", 0.0, 0.3, group="base"),
        Handle("cube.y", 0.0, 0.3, group="base"),
        Handle("cube.hue", 0.5, 0.5, group="full"),
    )
)
arm_hue_spec = LatentSpec(
    (
        Handle("arm.j0", 0.0, 1.5, group="base"),
        Handle("arm.j1", 0.0, 1.0, group="base"),
        Handle("arm.j2", -1.5, 1.5, group="base"),
        Handle("arm.j3", 1.8, 1.8, group="base"),
        Handle("gripper.aperture", 0.02, 0.02, group="base"),
        Handle("cube.hue", 0.5, 0.5, group="full"),
    )
)


def run_case(spec, hue_index):
    backend.bind(spec)
    base_z = torch.zeros(1, spec.n)
    hue_z = base_z.clone()
    hue_z[0, hue_index] = 1.0

    backend.write_state(spec.squash(base_z))
    print("READ_base", backend.read_state(), flush=True)
    backend.render(0)
    frame_base = backend.render(1)["cam0"]["rgb"].clone()

    backend.write_state(spec.squash(hue_z))
    print("READ_hue", backend.read_state(), flush=True)
    backend.render(0)
    frame_hue = backend.render(1)["cam0"]["rgb"].clone()

    d = (frame_hue.float() - frame_base.float()).abs()
    print(f"FRAME_DIFF max={d.max().item():.4f} mean={d.mean().item():.6f}", flush=True)


print("=== cube+hue (no arm) ===", flush=True)
run_case(cube_hue_spec, hue_index=2)

print("=== arm+hue (no cube) ===", flush=True)
run_case(arm_hue_spec, hue_index=5)
