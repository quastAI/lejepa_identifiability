"""Throwaway diagnostic for the base-group render-staleness bug (not part of
the test suite or the public API) -- delete once the bug is resolved.

Any tensor-API cube pose write (write_root_pose_to_sim) permanently
poisons the cube's material rendering for the rest of the session,
regardless of write order relative to sim.step() -- confirmed by two
separate tests. "arm+hue" (no cube tensor writes at all) works fine, so
the fix under test here: write cube.x/cube.y as a pure USD translate
Xform op on the cube prim instead, the same mechanism `cube.size`
already uses reliably, bypassing the tensor API entirely for cube
position.

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


def write_cube_xy_via_usd(x: float, y: float) -> None:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(rig.cube_prim)
    op = None
    for existing in xformable.GetOrderedXformOps():
        if existing.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op = existing
            break
    if op is None:
        op = xformable.AddTranslateOp()
    z = rig.ground_z + 0.5 * rig.default_cube_edge_m
    op.Set(Gf.Vec3d(x, y, z))


def read_cube_xy_via_usd() -> tuple[float, float]:
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(rig.cube_prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            return float(v[0]), float(v[1])
    return 0.0, 0.0


write_cube_xy_via_usd(0.0, 0.0)
_write_hue(rig.cube_shader, 0.5)
rig.sim.step(render=False)  # still needed for other groups' arm dims
backend.render(0)
frame_base = backend.render(1)["cam0"]["rgb"].clone()
print("READ_base_xy", read_cube_xy_via_usd(), flush=True)

write_cube_xy_via_usd(0.2285, 0.2285)
_write_hue(rig.cube_shader, 0.8808)
rig.sim.step(render=False)
backend.render(0)
frame_moved = backend.render(1)["cam0"]["rgb"].clone()
print("READ_moved_xy", read_cube_xy_via_usd(), flush=True)

d = (frame_moved.float() - frame_base.float()).abs()
print(f"FRAME_DIFF max={d.max().item():.4f} mean={d.mean().item():.6f}", flush=True)
