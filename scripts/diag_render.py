"""Throwaway diagnostic for the base-group render-staleness bug (not part of
the test suite or the public API) -- delete once the bug is resolved.

Adds a `cube.hue` (attribute-write) case alongside the physics-write cases,
and reports *mean* abs diff, not just max -- `style_sensitivity_gate`'s
`noise_floor=0.0` path only checks `mad > 0.0`, so a real content change
must be checked against genuine magnitude (tens of grey levels for a full
hue flip), not merely "nonzero", before trusting that any write -- physics
or attribute -- is really reaching the renderer.

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

import carb.settings  # noqa: E402

_settings = carb.settings.get_settings()
print("updateToUsd_before_scene_build", _settings.get("/physics/updateToUsd"), flush=True)
_settings.set_bool("/physics/updateToUsd", True)
_settings.set_bool("/physics/updateVelocitiesToUsd", True)

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
print("PLAYING", backend._rig.sim.is_playing(), flush=True)
print("updateToUsd_after_scene_build", _settings.get("/physics/updateToUsd"), flush=True)

n = spec.n
base_z = torch.zeros(1, n)
arm_z = base_z.clone()
arm_z[0, 0:5] = 1.0  # arm.j0..j3 + gripper.aperture
cube_z = base_z.clone()
cube_z[0, 5:7] = 1.0  # cube.x, cube.y
both_z = base_z.clone()
both_z[0, 0:7] = 1.0
hue_z = base_z.clone()
hue_z[0, 7] = 1.0  # cube.hue only -- attribute write, known-working reference

cases = {"base": base_z, "arm_only": arm_z, "cube_only": cube_z, "both": both_z, "hue_only": hue_z}
frames = {}
reads = {}
for name, z in cases.items():
    phi = spec.squash(z)
    backend.write_state(phi)
    reads[name] = backend.read_state()
    backend.render(0)  # discarded settle render
    frames[name] = backend.render(1)["cam0"]["rgb"].clone()
    print(f"READ_{name}", reads[name], flush=True)

base_frame = frames["base"].float()
for name in ("arm_only", "cube_only", "both", "hue_only"):
    d = (frames[name].float() - base_frame).abs()
    print(f"FRAME_{name}_vs_base max={d.max().item():.4f} mean={d.mean().item():.6f}", flush=True)
