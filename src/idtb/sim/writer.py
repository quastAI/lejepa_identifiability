"""README §6.3's write dispatch: one path per write mechanism, not one indexed
assignment over the whole latent vector -- a joint write, a root-state write,
and a USD/carb attribute write have different failure modes and each needs
its own read-back.

Every underlying call is one Spikes 1-5 verified (README §4.4, §7.5); this
file is what combines them into the single per-sample capture sequence §4.5
describes. Current scope is **`B=1` only**: the joint and root-state paths
are natively batched by Isaac Lab, but batching the attribute paths
(`cube.size`/`cube.hue`/`light.*`/`cam.jitter`/`exposure`/`table.albedo`) has
never been spiked -- each currently writes to one shared prim, not one per
env -- so this is a documented, honest limit, not a silent assumption.

Verified on the pod against :class:`idtb.sim.scene.Rig`, not here (this
repo has no Isaac runtime) -- see `write_latent_state`'s and
`_write_cube_position`'s docstrings for the two real defects that
composition found beyond what the spikes verified in isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from idtb.sim.backend import HandleInfo, WritePath

if TYPE_CHECKING:
    from idtb.latents import LatentSpec
    from idtb.sim.scene import Rig

Tensor = torch.Tensor

#: One entry per role this writer knows how to handle, and which of README
#: §6.3's three write paths carries it. A role missing here is a role no
#: spike has verified (`light.azimuth`/`light.elevation`, `table.roughness`)
#: -- refused at `bind()` time, never silently dropped (README Phase 4).
ROLE_WRITE_PATHS: dict[str, WritePath] = {
    "arm.j0": "joint",
    "arm.j1": "joint",
    "arm.j2": "joint",
    "arm.j3": "joint",
    "gripper.aperture": "joint",
    "cube.x": "root_state",
    "cube.y": "root_state",
    "cube.size": "attribute",
    "cube.hue": "attribute",
    "light.intensity": "attribute",
    "light.warmth": "attribute",
    "cam.jitter.x": "attribute",
    "cam.jitter.y": "attribute",
    "table.albedo": "attribute",
    "exposure": "attribute",
}

_ARM_ROLE_ORDER: tuple[str, ...] = ("arm.j0", "arm.j1", "arm.j2", "arm.j3")


def handle_infos(spec: LatentSpec) -> dict[str, HandleInfo]:
    return {role: HandleInfo(role, ROLE_WRITE_PATHS[role]) for role in spec.roles}


# ---------------------------------------------------------------------------
# Pure layer: no Isaac import. Unit-tested locally.
# ---------------------------------------------------------------------------


def resolve_cube_edge_m(spec: LatentSpec, phi: Tensor, *, default_edge_m: float) -> float:
    """`cube.size`'s edge length for this sample, or the fixed default if
    `cube.size` isn't part of the active spec (README §5.2.1's "held at
    handle centre" rule, applied to the one dimension the writer needs a
    concrete number from to derive the cube's resting height)."""
    if "cube.size" not in spec.roles:
        return default_edge_m
    return float(phi[0, spec.index("cube.size")].item())


def split_camera_jitter(spec: LatentSpec, phi: Tensor) -> tuple[float, float]:
    """`(dx, dy)` for this sample -- `0.0` on any axis not in the active spec."""
    dx = float(phi[0, spec.index("cam.jitter.x")].item()) if "cam.jitter.x" in spec.roles else 0.0
    dy = float(phi[0, spec.index("cam.jitter.y")].item()) if "cam.jitter.y" in spec.roles else 0.0
    return dx, dy


# ---------------------------------------------------------------------------
# Isaac layer: every import lives inside a function, after SimulationApp
# exists (README §4.2).
# ---------------------------------------------------------------------------


def write_latent_state(rig: Rig, spec: LatentSpec, phi: Tensor) -> None:
    """Teleport to squashed physical state `phi`, shape `[1, n]`.

    Dispatches by write path (README §6.3): joint columns and the root pose
    are staged into batched tensors and written once each; every attribute
    write applies immediately, since each is its own USD/carb call with no
    batched equivalent.

    Confirmed on the pod (and matching isaac-sim/IsaacLab#6394): a plain
    `sim.forward()` is not enough to make a joint write show up in a render
    -- Fabric-mirrored transforms only get republished by PhysX's own
    `simulate()`/`fetch_results()` cycle, which only `sim.step()` calls.
    README §5.5's original "no physics stepping" policy assumed
    `forward()` was sufficient; it wasn't. A real `sim.step(render=False)`
    is currently the *only* public path that republishes a tensor-API
    write to the renderer (IsaacLab#7138, which would add a lighter
    zero-dynamics alternative, is open/unmerged). We also set the joint
    position *target*, not just the state, so the PD controller has zero
    error at the moment of that step and doesn't drift the write during it
    -- confirmed by measurement, zero drift observed.

    `cube.x`/`cube.y` go through a translate xform op instead, exactly
    like `cube.size`'s scale op -- also confirmed on the pod: *any*
    tensor-API root-pose write on the cube permanently kills that same
    process's ability to render later `cube.hue`/`cube.size` attribute
    edits, regardless of write order relative to `sim.step()`. No official
    example combines a kinematic pose write with a live material edit in
    one loop (the tutorials only ever set materials once, before
    `sim.reset()`); a pure-USD position write sidesteps the interaction
    entirely rather than working around it.
    """
    if phi.shape[0] != 1:
        raise ValueError(f"write_latent_state supports B=1 only, got shape {tuple(phi.shape)}")

    joint_pos = rig.robot.data.default_joint_pos.clone()
    cube_x, cube_y = rig.default_cube_xy

    for i, handle in enumerate(spec.handles):
        role = handle.role
        value = float(phi[0, i].item())
        if role in _ARM_ROLE_ORDER:
            joint_pos[:, rig.arm_joint_ids[_ARM_ROLE_ORDER.index(role)]] = value
        elif role == "gripper.aperture":
            for joint_id in rig.finger_joint_ids:
                joint_pos[:, joint_id] = value
        elif role == "cube.x":
            cube_x = value
        elif role == "cube.y":
            cube_y = value
        elif role == "cube.size":
            _write_cube_scale(rig, value)
        elif role == "cube.hue":
            _write_hue(rig.cube_shader, value)
        elif role == "table.albedo":
            _write_grey_albedo(rig.table_shader, value)
        elif role == "light.intensity":
            _write_light_intensity(rig.light_prim, value)
        elif role == "light.warmth":
            _write_light_warmth(rig.light_prim, value)
        elif role == "exposure":
            from idtb.sim.render import write_exposure

            write_exposure(value)
        elif role not in ("cam.jitter.x", "cam.jitter.y"):
            raise AssertionError(f"unhandled role {role!r} -- bind() should have refused it")

    edge_m = resolve_cube_edge_m(spec, phi, default_edge_m=rig.default_cube_edge_m)
    _write_cube_position(rig, cube_x, cube_y, edge_m)

    rig.robot.set_joint_position_target(joint_pos)
    rig.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

    dx, dy = split_camera_jitter(spec, phi)
    _aim_camera(rig, jitter_xy=(dx, dy))

    rig.scene.write_data_to_sim()
    rig.sim.step(render=False)  # republishes the arm's write to Fabric (see docstring)


def read_latent_state(rig: Rig, spec: LatentSpec) -> Tensor:
    """`[1, n]`, for the §6.3 read-back gate -- one value per handle, read
    back through whichever path wrote it."""
    values = torch.empty(1, spec.n, dtype=torch.float32)
    joint_pos = rig.robot.data.joint_pos
    cube_x, cube_y = read_cube_position(rig)

    for i, handle in enumerate(spec.handles):
        role = handle.role
        if role in _ARM_ROLE_ORDER:
            values[0, i] = joint_pos[0, rig.arm_joint_ids[_ARM_ROLE_ORDER.index(role)]]
        elif role == "gripper.aperture":
            values[0, i] = joint_pos[0, rig.finger_joint_ids[0]]
        elif role == "cube.x":
            values[0, i] = cube_x
        elif role == "cube.y":
            values[0, i] = cube_y
        elif role == "cube.size":
            values[0, i] = _read_cube_edge(rig)
        elif role == "cube.hue":
            values[0, i] = _read_hue(rig.cube_shader)
        elif role == "table.albedo":
            values[0, i] = _read_grey_albedo(rig.table_shader)
        elif role == "light.intensity":
            values[0, i] = _read_light_intensity(rig.light_prim)
        elif role == "light.warmth":
            values[0, i] = _read_light_warmth(rig.light_prim)
        elif role == "exposure":
            from idtb.sim.render import read_exposure

            values[0, i] = read_exposure()
        elif role in ("cam.jitter.x", "cam.jitter.y"):
            values[0, i] = rig.pending_jitter[0 if role == "cam.jitter.x" else 1]
        else:
            raise AssertionError(f"unhandled role {role!r} -- bind() should have refused it")
    return values


def _write_cube_position(rig: Rig, x: float, y: float, edge_m: float) -> None:
    """`cube.x`/`cube.y` via a translate xform op (see `write_latent_state`'s
    docstring for why this isn't a `write_root_pose_to_sim` call) -- z comes
    along for free from the resting-height rule (README §4.4), local frame
    (no `env_origins` offset), matching how `read_cube_position` reads it
    back."""
    from pxr import Gf, UsdGeom

    z = rig.ground_z + 0.5 * edge_m
    op = _find_or_add_xform_op(rig.cube_prim, UsdGeom.XformOp.TypeTranslate)
    op.Set(Gf.Vec3d(x, y, z))


def read_cube_position(rig: Rig) -> tuple[float, float]:
    """`(x, y)`, local frame -- the counterpart to `_write_cube_position`,
    also used by `IsaacSceneBackend.diagnostics()`'s collision proxy since
    the cube's root-pose tensor buffer is never written any more."""
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(rig.cube_prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            value = op.Get()
            return float(value[0]), float(value[1])
    return rig.default_cube_xy


def _write_cube_scale(rig: Rig, edge_m: float) -> None:
    """`cube.size` via an xform scale multiplier on top of the authored edge
    (README §5.2.2) -- not a root-state write."""
    from pxr import Gf, UsdGeom

    factor = edge_m / rig.default_cube_edge_m
    op = _find_or_add_xform_op(rig.cube_prim, UsdGeom.XformOp.TypeScale)
    op.Set(Gf.Vec3f(factor, factor, factor))


def _read_cube_edge(rig: Rig) -> float:
    from pxr import UsdGeom

    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_], useExtentsHint=False)
    size = cache.ComputeWorldBound(rig.cube_prim).ComputeAlignedRange().GetSize()
    return float(size[2])


def _write_hue(shader: Any, hue: float) -> None:
    import colorsys

    from pxr import Gf

    rgb = colorsys.hsv_to_rgb(hue % 1.0, 0.85, 0.85)
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(*rgb))


def _read_hue(shader: Any) -> float:
    import colorsys

    value = shader.GetInput("diffuseColor").Get()
    r, g, b = float(value[0]), float(value[1]), float(value[2])
    hue, _saturation, _value = colorsys.rgb_to_hsv(r, g, b)
    return hue


def _write_grey_albedo(shader: Any, albedo: float) -> None:
    from pxr import Gf

    shader.GetInput("diffuseColor").Set(Gf.Vec3f(albedo, albedo, albedo))


def _read_grey_albedo(shader: Any) -> float:
    return float(shader.GetInput("diffuseColor").Get()[0])


def _write_light_intensity(light_prim: Any, intensity: float) -> None:
    from pxr import UsdLux

    UsdLux.LightAPI(light_prim).GetIntensityAttr().Set(intensity)


def _read_light_intensity(light_prim: Any) -> float:
    from pxr import UsdLux

    return float(UsdLux.LightAPI(light_prim).GetIntensityAttr().Get())


def _write_light_warmth(light_prim: Any, kelvin: float) -> None:
    from pxr import UsdLux

    api = UsdLux.LightAPI(light_prim)
    api.GetEnableColorTemperatureAttr().Set(True)
    api.GetColorTemperatureAttr().Set(kelvin)


def _read_light_warmth(light_prim: Any) -> float:
    from pxr import UsdLux

    return float(UsdLux.LightAPI(light_prim).GetColorTemperatureAttr().Get())


def _aim_camera(rig: Rig, *, jitter_xy: tuple[float, float]) -> None:
    """Re-aim the camera at the cube every capture, offset by `jitter_xy`
    (README §5.2.3) -- a different usage pattern from aiming once at boot,
    per §7.5's `camera_jitter_per_capture` check."""
    dx, dy = jitter_xy
    origin = rig.scene.env_origins[0]
    eye_xyz = [rig.camera_eye[0] + dx, rig.camera_eye[1] + dy, rig.camera_eye[2]]
    eye = origin + torch.tensor(eye_xyz, device=origin.device)
    target = origin + torch.tensor(list(rig.camera_target), device=origin.device)
    rig.camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
    rig.pending_jitter = jitter_xy


def _find_or_add_xform_op(prim: Any, op_type: Any) -> Any:
    """Find an existing xform op of `op_type` on `prim`, or add one --
    `UsdGeom.XformCommonAPI` fails silently on a pattern it doesn't expect
    (README §7.5's `cube.size` finding). Keeps a fixed [translate, scale]
    order regardless of which role happens to be written first this
    session, so translate is always the outermost op (applied after
    scale) -- scaling around the cube's own origin, then moving it, not
    the other way round."""
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    if op_type == UsdGeom.XformOp.TypeTranslate:
        new_op = xformable.AddTranslateOp()
    elif op_type == UsdGeom.XformOp.TypeScale:
        new_op = xformable.AddScaleOp()
    else:
        raise ValueError(f"no add-op helper wired up for {op_type!r}")
    ordered = sorted(
        xformable.GetOrderedXformOps(),
        key=lambda op: 0 if op.GetOpType() == UsdGeom.XformOp.TypeTranslate else 1,
    )
    xformable.SetXformOpOrder(ordered)
    return new_op
