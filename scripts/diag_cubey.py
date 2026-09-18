"""Throwaway diagnostic (not part of the test suite) -- delete once resolved.

pytest's --showlocals truncates the `responses` dict repr regardless of
`CI=1`/`-vv` (that only affects assertion-rewrite truncation, not
saferepr's own size cap), so the exact mad_vs_base values for cube.x/
cube.y under `full+style` were never actually visible. Reusing the same
spec/gate/capture helpers the real test uses, printing every value
directly instead of relying on a repr.

Run: /workspace/isaaclab/isaaclab.sh -p /idtb/repo/scripts/diag_cubey.py 2>&1 | grep -E "^MAD_|^READ_"
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
from idtb.gates import mean_abs_diff  # noqa: E402
from idtb.sim.scene import IsaacSceneBackend  # noqa: E402

FULL_STYLE_SPEC = LatentSpec(
    (
        Handle("arm.j0", 0.0, 1.5, group="base"),
        Handle("arm.j1", 0.0, 1.0, group="base"),
        Handle("arm.j2", -1.5, 1.5, group="base"),
        Handle("arm.j3", 1.8, 1.8, group="base"),
        Handle("gripper.aperture", 0.02, 0.02, group="base"),
        Handle("cube.x", 0.0, 0.3, group="base"),
        Handle("cube.y", 0.0, 0.3, group="base"),
        Handle("cube.size", 0.05, 0.02, group="full"),
        Handle("cube.hue", 0.5, 0.45, group="full"),
        Handle("light.intensity", 0.0, 1.0, group="style"),
        Handle("light.warmth", 0.0, 1.0, group="style"),
        Handle("cam.jitter.x", 0.0, 0.5, group="style"),
        Handle("cam.jitter.y", 0.0, 0.5, group="style"),
        Handle("table.albedo", 0.0, 1.0, group="style"),
        Handle("exposure", 0.0, 1.0, group="style"),
    )
)

backend = IsaacSceneBackend()
backend.bind(FULL_STYLE_SPEC)

n = FULL_STYLE_SPEC.n
base_z = torch.zeros(1, n)
base_phi = FULL_STYLE_SPEC.squash(base_z)

backend.write_state(base_phi)
print("READ_base", backend.read_state(), flush=True)
base_frame = backend.render(0)["cam0"]["rgb"].clone()

base_dims = FULL_STYLE_SPEC.dims("base")
for dim in base_dims:
    handle = FULL_STYLE_SPEC.handles[dim]
    phi = base_phi.clone()
    phi[0, dim] = handle.center + 0.8 * handle.radius
    backend.write_state(phi)
    read = backend.read_state()
    frame = backend.render(0)["cam0"]["rgb"].clone()
    mad = mean_abs_diff(base_frame, frame)
    print(f"MAD_{handle.role} = {mad}", flush=True)
    print(f"READ_{handle.role} cube.x={read[0, 5].item():.4f} cube.y={read[0, 6].item():.4f}", flush=True)
