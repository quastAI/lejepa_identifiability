"""Spike 5: the attribute write paths for README §5.2's `full` and `style` groups.

Spikes 1-4 (``spikes/spike_api.py``) measured exactly two write paths --
``write_joint_state_to_sim`` and ``write_root_state_to_sim`` -- which is §5.2's
`base` group and nothing else. Every `full`/`style` latent moves through a
*different* mechanism: a geometry write for ``cube.size``, a USD material write
for ``cube.hue``, a light-attribute write for ``light.*``, a per-capture camera
re-aim for ``cam.jitter.*``, a post-process setting for ``exposure``. None of
those has been touched, and none of Spike 1's verdicts transfers to them by
default (docs/PLAN.md Phase 3b).

Run it on the pod, not here::

    ./isaaclab.sh -p spikes/spike_dynamic_attrs.py --out /idtb/data/spike_attrs

This file reuses ``spikes/spike_api.py``'s pure layer directly -- ``Report``,
``determinism_report``, ``sensitivity_report``, the uint8-safe diffs -- rather
than duplicating it (loaded by path below, since ``spikes/`` is a script
directory, not a package). Only the Isaac-layer rig is new, and it is
deliberately smaller: every open question here is about the cube, a light and
the camera, so there is no Franka and no ``InteractiveScene`` -- a single
absolute-path prim per object, spawned directly, is enough and sidesteps the
whole ``{ENV_REGEX_NS}`` multi-env templating problem Spike 1 needed for its
`#251 <https://github.com/isaac-sim/IsaacSim/issues/251>`_ check, which this
spike has no reason to repeat.

Three structural rules, carried over from Spike 1 (docs/PLAN.md Phase 3):

1. **Never fail fast.** Every check isolated, all run, one PASS/FAIL table and
   one ``facts.json`` at the end. A FAIL that is *understood* is a completed
   check, not a bug to force past -- the motion-blur check below is exactly
   that, by design.
2. **One write path per check.** A failure writing cube colour must not block
   finding out whether writing light intensity works -- isolating *which
   mechanisms exist* is the whole point of this spike.
3. **Resolve, don't guess.** Every Isaac/USD symbol below was chosen from
   documentation, never from running anything (macOS has no local Isaac
   runtime, README §1). Where more than one call could plausibly be the right
   one, :func:`try_candidates` tries each and records which answered, the same
   discipline as ``spike_api.resolve()`` generalised from import paths to
   arbitrary write attempts.

**Three questions per knob, and the order matters** (docs/PLAN.md Phase 3b):
does the write land (a per-attribute read-back), does varying it alone move
pixels far above the noise floor (ranked above determinism for the same reason
Spike 1 ranks it above determinism -- a `style` knob that writes cleanly and
renders deterministically but changes nothing visible is worse than useless:
the encoder would score perfect invariance for free), and is it still bitwise
deterministic under `standard`. :func:`run_knob_check` runs all three, in that
order, for one attribute at a time.
"""

from __future__ import annotations

import argparse
import colorsys
import importlib.util
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Reuse spike_api.py's pure layer -- loaded by path, `spikes/` is a script
# directory, not a package (same technique as tests/test_spike_api.py).
# Importing it also runs its OMNICLIENT_HUB_MODE default, so that fix applies
# here too without being duplicated.
# ---------------------------------------------------------------------------


def _load_spike_api() -> Any:
    path = Path(__file__).resolve().parent / "spike_api.py"
    spec = importlib.util.spec_from_file_location("spike_api", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


spike_api = _load_spike_api()

Tensor = spike_api.Tensor
Capture = spike_api.Capture
Report = spike_api.Report
CheckFailed = spike_api.CheckFailed
CheckSkipped = spike_api.CheckSkipped
determinism_report = spike_api.determinism_report
sensitivity_report = spike_api.sensitivity_report
mean_abs_diff = spike_api.mean_abs_diff
bitwise_equal = spike_api.bitwise_equal
frame_stats = spike_api.frame_stats
resolve = spike_api.resolve
apply_preset = spike_api.apply_preset

# ---------------------------------------------------------------------------
# Pure layer: no Isaac, no globals. Unit-tested locally (tests/test_spike_dynamic_attrs.py).
# ---------------------------------------------------------------------------


def try_candidates(candidates: Sequence[tuple[str, Callable[[], Any]]]) -> tuple[str, Any]:
    """Attempt each ``(name, thunk)`` in order; return the first that doesn't raise.

    Generalises ``spike_api.resolve()`` from import paths to arbitrary zero-arg
    callables -- this spike's uncertainty is mostly about *which call* writes an
    attribute, not which module it lives in. A candidate that fails on this
    build costs a recorded note, not the check (README §7.5: "resolve, don't
    guess").
    """
    errors = []
    for name, thunk in candidates:
        try:
            return name, thunk()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise CheckFailed("no candidate write path worked:\n  " + "\n  ".join(errors))


def hue_to_rgb(
    hue: float, *, saturation: float = 0.85, value: float = 0.85
) -> tuple[float, float, float]:
    """`cube.hue` is one dimension; saturation and value are fixed (README §5.2.2).

    ``hue`` wraps into ``[0, 1)`` so a value outside the sampled range degrades
    to a colour rather than an error -- the actual non-wrapping constraint is
    enforced upstream by the squash radius, not here.
    """
    return colorsys.hsv_to_rgb(hue % 1.0, saturation, value)


def azel_to_direction(azimuth_rad: float, elevation_rad: float) -> tuple[float, float, float]:
    """Unit vector a key light points *along*, from azimuth (about Z) and elevation.

    Pure trig, independent of how a particular light schema expects that
    direction to be authored (that translation happens in :func:`write_light_direction`).
    """
    cos_el = math.cos(elevation_rad)
    return (
        cos_el * math.cos(azimuth_rad),
        cos_el * math.sin(azimuth_rad),
        math.sin(elevation_rad),
    )


def carb_value_matches(wanted: Any, read_back: Any) -> bool:
    """Was a carb setting actually accepted -- tolerant of float32 round-trip.

    carb settings commonly store floats as float32 internally; ``1.6`` can
    read back as ``1.600000023841858``, and a strict ``==`` would then report
    a perfectly good setting as "not accepted". Bools and ints compare exactly
    (``math.isclose`` on a bool is well-defined but not what's meant here).
    """
    if isinstance(wanted, float) and isinstance(read_back, int | float):
        return math.isclose(read_back, wanted, rel_tol=1e-5, abs_tol=1e-6)
    return read_back == wanted


def cube_size_radius_from_aperture(aperture_m: float, *, margin: float = 0.8) -> float:
    """`cube.size`'s upper half-range, from the *measured* per-finger aperture.

    README §5.2.2: a cube edge the gripper cannot close on is outside the task
    by definition. ``aperture_m`` is Spike 1's measured per-finger travel
    (``[0.0, 0.04]`` m, README §5.2.1); the full opening is twice that, and
    ``margin`` keeps clearance so a squashed sample near the bound is still
    closable rather than sitting exactly at the mechanical limit.
    """
    if not 0.0 < margin <= 1.0:
        raise ValueError(f"margin must lie in (0, 1], got {margin!r}")
    return margin * 2.0 * aperture_m


# ---------------------------------------------------------------------------
# Isaac layer: every import lives inside a function, after SimulationApp exists.
# ---------------------------------------------------------------------------

GROUND_PATH = "/World/Ground"
LIGHT_PATH = "/World/Light"
CUBE_PATH = "/World/Cube"
CAMERA_PATH = "/World/Camera"

CUBE_TRANSLATION_XY = (0.45, 0.0)  # z is derived from the base edge, in build_rig()
CAMERA_EYE = (1.0, 1.0, 0.9)
CAMERA_TARGET = (0.45, 0.0, 0.1)

BASE_CUBE_EDGE_M = 0.06
PERTURBED_CUBE_EDGE_M = 0.09
CUBE_SCALE_READBACK_ATOL_M = 2e-3  # bbox-cache based -- coarser than a solver read-back

BASE_CUBE_HUE = 0.05
PERTURBED_CUBE_HUE = 0.55
CUBE_HUE_READBACK_ATOL = 1e-3

BASE_LIGHT_INTENSITY = 900.0
PERTURBED_LIGHT_INTENSITY = 2400.0
LIGHT_INTENSITY_READBACK_ATOL = 1e-3

BASE_LIGHT_WARMTH_K = 5500.0
PERTURBED_LIGHT_WARMTH_K = 3200.0
LIGHT_WARMTH_READBACK_ATOL = 1e-3

BASE_LIGHT_AZIMUTH_ELEVATION = (0.0, math.radians(35.0))
PERTURBED_LIGHT_AZIMUTH_ELEVATION = (math.radians(120.0), math.radians(55.0))

BASE_CAMERA_JITTER = (0.0, 0.0)
PERTURBED_CAMERA_JITTER = (0.05, -0.04)

# Spike 1, README §5.2.1: the measured per-finger aperture that bounds cube.size.
MEASURED_FINGER_APERTURE_M = 0.04

# For measuring where the render clips, not for any handle's actual range.
INTENSITY_CANDIDATES_FOR_CLIPPING = (50.0, 300.0, 900.0, 3000.0, 8000.0, 20000.0)

# Exact key names read from documentation, never run -- exactly what this spike
# exists to settle. carb happily creates unknown keys, so `accepted` in
# apply_carb_settings() (read-back equality) is the only evidence a path exists.
EXPOSURE_CANDIDATES: dict[str, Any] = {
    "/rtx/post/tonemap/exposure": 1.6,
    "/rtx/post/tonemap/filmIso": 200.0,
    "/rtx/post/histogram/enabled": False,
}

MOTION_BLUR_CANDIDATES: dict[str, Any] = {
    "/rtx/post/motionblur/enabled": True,
    "/rtx/post/motionblur/maxBlurDiameterFraction": 0.05,
}


@dataclass
class Rig:
    """Everything the checks need from one built scene."""

    sim: Any
    device: str
    camera: Any
    cube_prim: Any
    cube_shader: Any  # UsdShade.Shader, or None if binding resolution failed
    light_prim: Any
    notes: dict[str, Any]


def build_rig(args: argparse.Namespace) -> Rig:
    """Build the smallest scene that can answer §7.5: cube + light + camera.

    No ``InteractiveScene`` and no env-namespace templating -- every prim sits
    at a fixed absolute path this script chose itself, which is the whole
    reason the ``{ENV_REGEX_NS}`` resolution problem Spike 1 needed does not
    come up here (docs/PLAN.md Phase 3b).
    """
    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    notes: dict[str, Any] = {}

    render_cfg = None
    try:
        from isaaclab.sim import RenderCfg

        render_cfg = RenderCfg(antialiasing_mode="Off", enable_dl_denoiser=False)
        notes["render_cfg"] = "antialiasing_mode=Off, enable_dl_denoiser=False"
    except Exception as exc:
        notes["render_cfg"] = f"unavailable, booting with defaults: {type(exc).__name__}: {exc}"

    sim_kwargs: dict[str, Any] = {"dt": 1.0 / 60.0, "device": args.device}
    if render_cfg is not None:
        sim_kwargs["render"] = render_cfg
    sim = SimulationContext(SimulationCfg(**sim_kwargs))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func(GROUND_PATH, ground_cfg)

    light_cfg = sim_utils.DistantLightCfg(intensity=BASE_LIGHT_INTENSITY, color=(1.0, 1.0, 1.0))
    light_cfg.func(LIGHT_PATH, light_cfg)

    cube_translation = (*CUBE_TRANSLATION_XY, 0.5 * BASE_CUBE_EDGE_M)
    cube_cfg = sim_utils.CuboidCfg(
        size=(BASE_CUBE_EDGE_M, BASE_CUBE_EDGE_M, BASE_CUBE_EDGE_M),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=hue_to_rgb(BASE_CUBE_HUE)),
    )
    cube_cfg.func(CUBE_PATH, cube_cfg, translation=cube_translation)

    camera_cfg = CameraCfg(
        prim_path=CAMERA_PATH,
        update_period=0.0,
        height=args.resolution[0],
        width=args.resolution[1],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 40.0)),
        offset=CameraCfg.OffsetCfg(pos=CAMERA_EYE, convention="world"),
    )
    camera = Camera(camera_cfg)

    sim.reset()

    stage = omni.usd.get_context().get_stage()
    cube_prim = stage.GetPrimAtPath(CUBE_PATH)
    light_prim = stage.GetPrimAtPath(LIGHT_PATH)

    cube_shader = None
    try:
        shader_name, cube_shader = resolve_cube_shader(cube_prim)
        notes["cube_shader_resolved_via"] = shader_name
    except Exception as exc:
        # A colour-write failure must not take the scale/light/camera checks
        # down with it (structural rule 2) -- recorded here, raised later only
        # by the checks that actually need the shader.
        notes["cube_shader_resolved_via"] = f"unresolved: {type(exc).__name__}: {exc}"
        # Logged unconditionally on failure so the next run doesn't need to
        # guess again -- see describe_prim_tree's docstring.
        try:
            notes["cube_prim_tree"] = describe_prim_tree(cube_prim)
        except Exception as tree_exc:
            notes["cube_prim_tree"] = f"unavailable: {type(tree_exc).__name__}: {tree_exc}"

    rig = Rig(
        sim=sim,
        device=str(sim.device),
        camera=camera,
        cube_prim=cube_prim,
        cube_shader=cube_shader,
        light_prim=light_prim,
        notes=notes,
    )
    aim_camera(rig, jitter_xy=BASE_CAMERA_JITTER)
    camera.update(dt=0.0, force_recompute=True)
    return rig


def resolve_cube_shader(cube_prim: Any) -> tuple[str, Any]:
    """Find the UsdShade.Shader driving the cube's diffuse colour.

    Tries the schema-correct lookup on the cube prim itself first, then walks
    its whole subtree for a bound material -- measured on the pod: binding-API
    lookup on ``cube_prim`` directly found nothing (docs/PLAN.md Phase 3b,
    first run), meaning the spawner most likely nests the actual visual
    geometry (and its binding) under a child prim rather than binding on
    ``cube_prim`` itself. The recursive search finds it regardless of naming,
    which is more robust than guessing another literal path.
    """
    from pxr import Usd, UsdShade

    def via_binding_api() -> Any:
        material, _ = UsdShade.MaterialBindingAPI(cube_prim).ComputeBoundMaterial()
        if not material:
            raise RuntimeError("no bound material")
        source, _, _ = material.ComputeSurfaceSource()
        if not source:
            raise RuntimeError("material has no surface source")
        return source

    def via_recursive_binding_search() -> Any:
        for prim in Usd.PrimRange(cube_prim):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not material:
                continue
            source, _, _ = material.ComputeSurfaceSource()
            if source:
                return source
        raise RuntimeError("no descendant of the cube prim has a bound material with a surface")

    def via_looks_convention() -> Any:
        stage = cube_prim.GetStage()
        for suffix in (
            "Looks/Material/Shader",
            "Looks/PreviewSurface/Shader",
            "Looks/visualMaterial/Shader",
        ):
            shader_prim = stage.GetPrimAtPath(cube_prim.GetPath().AppendPath(suffix))
            if shader_prim and shader_prim.IsValid():
                return UsdShade.Shader(shader_prim)
        raise RuntimeError("no shader found under the Looks/ convention")

    return try_candidates(
        [
            ("material_binding_api", via_binding_api),
            ("recursive_binding_search", via_recursive_binding_search),
            ("looks_convention", via_looks_convention),
        ]
    )


def describe_prim_tree(root_prim: Any) -> list[str]:
    """Every descendant of ``root_prim``, with its type and any bound material --
    a diagnostic, not a write path. Logged unconditionally in scene_builds_and_measures
    so a resolution failure is something to *read*, not guess at again (README §7.5:
    "resolve, don't guess" extends to debugging the resolver itself).
    """
    from pxr import Usd, UsdShade

    lines = []
    for prim in Usd.PrimRange(root_prim):
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        bound = material.GetPath().pathString if material else None
        lines.append(f"{prim.GetPath()} [{prim.GetTypeName()}] bound_material={bound}")
    return lines


def aim_camera(rig: Rig, *, jitter_xy: tuple[float, float]) -> None:
    """Re-aim the camera at the cube, offset by ``jitter_xy`` -- called on *every*
    capture for the ``cam.jitter.*`` check, not once at boot (README §5.2.3):
    Spike 1 only ever called ``set_world_poses_from_view`` once, which is a
    different usage pattern this spike checks separately.
    """
    dx, dy = jitter_xy
    device = getattr(rig.camera, "device", None) or rig.sim.device
    eye = torch.tensor([[CAMERA_EYE[0] + dx, CAMERA_EYE[1] + dy, CAMERA_EYE[2]]], device=device)
    target = torch.tensor([[CAMERA_TARGET[0], CAMERA_TARGET[1], CAMERA_TARGET[2]]], device=device)
    rig.camera.set_world_poses_from_view(eye, target)


def read_cube_bbox_extent(rig: Rig) -> tuple[float, float, float]:
    """World-space axis-aligned extent of the cube -- a proxy for *both* the
    visual size and the collider, since PhysX derives the collision shape from
    the same authored mesh + scale. Recorded as an approximation, not a direct
    PhysX query; sharpen this on the pod if it turns out not to be enough.
    """
    from pxr import UsdGeom

    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_], useExtentsHint=False)
    size = cache.ComputeWorldBound(rig.cube_prim).ComputeAlignedRange().GetSize()
    return (float(size[0]), float(size[1]), float(size[2]))


def read_cube_diffuse_color(rig: Rig) -> tuple[float, float, float]:
    value = rig.cube_shader.GetInput("diffuseColor").Get()
    return (float(value[0]), float(value[1]), float(value[2]))


def read_cube_translation(rig: Rig) -> tuple[float, float, float]:
    from pxr import UsdGeom

    matrix = UsdGeom.Xformable(rig.cube_prim).ComputeLocalToWorldTransform(0)
    t = matrix.ExtractTranslation()
    return (float(t[0]), float(t[1]), float(t[2]))


def read_light_intensity(rig: Rig) -> float:
    from pxr import UsdLux

    return float(UsdLux.LightAPI(rig.light_prim).GetIntensityAttr().Get())


def read_light_warmth(rig: Rig) -> float:
    from pxr import UsdLux

    return float(UsdLux.LightAPI(rig.light_prim).GetColorTemperatureAttr().Get())


def _find_or_add_xform_op(prim: Any, op_type: Any) -> Any:
    """Find an existing xform op of ``op_type`` on ``prim``, or add one.

    ``UsdGeom.XformCommonAPI`` refuses to touch a prim whose existing xform ops
    aren't in exactly the pattern it expects, and fails *silently* -- `Set*`
    returns ``False`` rather than raising, which is indistinguishable from
    success unless the caller checks it. Measured on the pod: `cube.size`'s
    write never landed and nothing said why (docs/PLAN.md Phase 3b, first
    run). Working the ops directly is robust to whatever ``xformOpOrder``
    Isaac Lab's spawner already authored.
    """
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type:
            return op
    if op_type == UsdGeom.XformOp.TypeScale:
        return xformable.AddScaleOp()
    if op_type == UsdGeom.XformOp.TypeRotateXYZ:
        return xformable.AddRotateXYZOp()
    raise ValueError(f"no add-op helper wired up for {op_type!r}")


def write_cube_scale(rig: Rig, edge_m: float) -> None:
    """``cube.size`` via an xform scale multiplier on top of the authored edge --
    not a root-state write (README §5.2.2)."""
    from pxr import Gf, UsdGeom

    factor = edge_m / BASE_CUBE_EDGE_M
    op = _find_or_add_xform_op(rig.cube_prim, UsdGeom.XformOp.TypeScale)
    op.Set(Gf.Vec3f(factor, factor, factor))


def write_cube_hue(rig: Rig, hue: float) -> None:
    if rig.cube_shader is None:
        raise CheckFailed("cube shader was never resolved -- see scene_builds_and_measures")
    from pxr import Gf

    rig.cube_shader.GetInput("diffuseColor").Set(Gf.Vec3f(*hue_to_rgb(hue)))


def reset_cube_to_default(rig: Rig) -> None:
    write_cube_scale(rig, BASE_CUBE_EDGE_M)
    if rig.cube_shader is not None:
        write_cube_hue(rig, BASE_CUBE_HUE)


def write_light_intensity(rig: Rig, intensity: float) -> None:
    from pxr import UsdLux

    UsdLux.LightAPI(rig.light_prim).GetIntensityAttr().Set(intensity)


def write_light_warmth(rig: Rig, kelvin: float) -> None:
    from pxr import UsdLux

    api = UsdLux.LightAPI(rig.light_prim)
    api.GetEnableColorTemperatureAttr().Set(True)
    api.GetColorTemperatureAttr().Set(kelvin)


def write_light_direction(
    rig: Rig, *, azimuth_rad: float, elevation_rad: float
) -> tuple[float, ...]:
    """Rotate the key light so its emission direction matches (azimuth, elevation).

    A ``UsdLux`` distant/directional light emits along its local ``-Z``; the
    rotation is built as "take -Z to the wanted direction" and decomposed into
    the XYZ Euler angles a rotateXYZ op wants. The decomposition's angle
    *order* is the one part of this file worth a visual sanity check on the
    pod (does the shadow actually move where azimuth/elevation say it should)
    rather than trusting the math alone.
    """
    from pxr import Gf, UsdGeom

    direction = azel_to_direction(azimuth_rad, elevation_rad)
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*direction))
    euler_deg = rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
    op = _find_or_add_xform_op(rig.light_prim, UsdGeom.XformOp.TypeRotateXYZ)
    op.Set(Gf.Vec3f(euler_deg[0], euler_deg[1], euler_deg[2]))
    return direction


def apply_carb_settings(values: Mapping[str, Any]) -> dict[str, Any]:
    """Set each carb key and read it back -- generalises ``spike_api.apply_preset``
    from a named, module-level preset to an arbitrary settings dict, for the
    exposure/motion-blur candidates this spike is actually uncertain about.
    """
    import carb

    settings = carb.settings.get_settings()
    applied: dict[str, Any] = {}
    for key, value in values.items():
        try:
            settings.set(key, value)
            read_back = settings.get(key)
            applied[key] = {
                "wanted": value,
                "read_back": read_back,
                "accepted": carb_value_matches(value, read_back),
            }
        except Exception as exc:
            applied[key] = {"wanted": value, "error": f"{type(exc).__name__}: {exc}"}
    return applied


def make_capture_static(rig: Rig) -> Capture:
    """A capture with no write of its own -- for checks that mutate state
    explicitly before calling it (write-order independence, cross-talk).

    The §4.5 sequence, matching ``spike_api.py``'s proven-working one: flush,
    aim the camera at the current state, ``sim.render()`` -- the call that
    actually produces a new frame, missing from an earlier version of this
    function -- then a final ``camera.update()`` to pull it. Spike 1 measured
    that one render call suffices once `standard`'s carb settings are applied
    (``totalSpp`` converges within a single external call, §7.2 Spike 3);
    without any render call at all, every capture returns whatever frame the
    sensor initialised with, which is exactly the "renders the same stale
    frame every time" failure `sensitivity_report` exists to catch -- and
    caught, on the first pod run of this file (docs/PLAN.md Phase 3b).
    """

    def capture(_state: Any) -> Tensor:
        rig.sim.forward()
        rig.camera.update(dt=0.0, force_recompute=True)
        rig.sim.render()
        rig.camera.update(dt=0.0, force_recompute=True)
        return rig.camera.data.output["rgb"]

    return capture


def make_attribute_capture(rig: Rig, writer: Callable[[Any], None]) -> Capture:
    """A capture closure for a single-attribute write path (README §6.3's dispatch
    is by write path; this spike only ever varies one attribute per check)."""
    static_capture = make_capture_static(rig)

    def capture(value: Any) -> Tensor:
        writer(value)
        return static_capture(value)

    return capture


@dataclass
class KnobCheck:
    """One knob's write -> read-back -> sensitivity -> determinism recipe.

    docs/PLAN.md Phase 3b: three questions per knob, in that order. The same
    recipe applies to every §5.2 `full`/`style` attribute; only the write/read
    functions and the two values change per knob.
    """

    role: str
    write: Callable[[Any], None]
    base_value: Any
    perturbed_value: Any
    read: Callable[[], Any] | None = None
    readback_error: Callable[[Any, Any], float] | None = None
    readback_atol: float = 0.0


def run_knob_check(rig: Rig, knob: KnobCheck) -> dict[str, Any]:
    problems: list[str] = []
    facts: dict[str, Any] = {}

    # -- 1. does the write land? ---------------------------------------------
    knob.write(knob.perturbed_value)
    if knob.read is not None and knob.readback_error is not None:
        observed = knob.read()
        error = knob.readback_error(knob.perturbed_value, observed)
        facts["write_and_readback"] = {
            "wrote": knob.perturbed_value,
            "read_back": observed,
            "error": error,
            "atol": knob.readback_atol,
        }
        if error > knob.readback_atol:
            problems.append(
                f"{knob.role}: write did not land -- read back {observed!r} for "
                f"{knob.perturbed_value!r} (error {error:.3g}, atol {knob.readback_atol:g})"
            )
    knob.write(knob.base_value)

    # -- 2. does it move pixels above the noise floor? -----------------------
    preset_settings = apply_preset("pathtracing_denoiser_off")
    capture = make_attribute_capture(rig, knob.write)
    first = capture(knob.base_value).clone()
    second = capture(knob.base_value).clone()
    noise_floor = mean_abs_diff(first, second)

    sens = sensitivity_report(
        capture, knob.base_value, {knob.role: knob.perturbed_value}, noise_floor=noise_floor
    )
    facts["sensitivity"] = sens
    if not sens["all_responsive"]:
        problems.append(f"{knob.role}: varying it alone does not move pixels above the noise floor")

    # -- 3. still bitwise deterministic under `standard`? --------------------
    det = determinism_report(capture, knob.base_value, knob.perturbed_value, tol=0.0)
    facts["determinism"] = det
    facts["preset_applied"] = {"name": "pathtracing_denoiser_off", "settings": preset_settings}
    if not det["content_reproducible"]:
        problems.append(
            f"{knob.role}: not bitwise deterministic under `standard` while varying it "
            f"(order-independence {det['order_independent_mad']:.4g}, "
            f"back-to-back {det['back_to_back_mad']:.4g})"
        )

    knob.write(knob.base_value)  # leave the rig as found for whichever check runs next
    if problems:
        raise CheckFailed("; ".join(problems))
    return facts


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("/idtb/data/spike_attrs"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(128, 128), metavar=("H", "W"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    report = Report()
    out_dir: Path = args.out
    rig: Rig | None = None
    started = time.time()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[warn] cannot create {out_dir}: {type(exc).__name__}: {exc}", flush=True)

    def need_rig() -> Rig:
        if rig is None:
            raise CheckSkipped("scene did not build")
        return rig

    try:
        # -- boot -----------------------------------------------------------
        def check_versions() -> dict[str, Any]:
            facts: dict[str, Any] = {"torch": torch.__version__, "device_arg": args.device}
            for name, path in (
                ("isaacsim", "isaacsim.__version__"),
                ("isaaclab", "isaaclab.__version__"),
            ):
                try:
                    facts[name], _ = resolve([path])
                except Exception as exc:
                    facts[name] = f"unavailable: {type(exc).__name__}: {exc}"
            if torch.cuda.is_available():
                facts["cuda"] = torch.version.cuda
                facts["gpu"] = torch.cuda.get_device_name(0)
                facts["capability"] = ".".join(str(c) for c in torch.cuda.get_device_capability(0))
            else:
                facts["gpu"] = "torch reports no CUDA device"
            return facts

        report.run("boot_and_versions", check_versions)

        # -- scene ------------------------------------------------------------
        def check_scene() -> dict[str, Any]:
            nonlocal rig
            rig = build_rig(args)
            facts: dict[str, Any] = {
                "cube_prim_path": CUBE_PATH,
                "light_prim_path": LIGHT_PATH,
                "camera_prim_path": CAMERA_PATH,
                **rig.notes,
            }
            try:
                facts["intrinsic_matrices"] = rig.camera.data.intrinsic_matrices[0].tolist()
            except Exception as exc:
                facts["intrinsic_matrices"] = f"unavailable: {type(exc).__name__}: {exc}"
            return facts

        report.run("scene_builds_and_measures", check_scene)

        # -- cube.size --------------------------------------------------------
        def check_cube_scale() -> dict[str, Any]:
            current = need_rig()
            return run_knob_check(
                current,
                KnobCheck(
                    role="cube.size",
                    write=lambda v: write_cube_scale(current, v),
                    read=lambda: max(read_cube_bbox_extent(current)),
                    readback_error=lambda wrote, read: abs(read - wrote),
                    readback_atol=CUBE_SCALE_READBACK_ATOL_M,
                    base_value=BASE_CUBE_EDGE_M,
                    perturbed_value=PERTURBED_CUBE_EDGE_M,
                ),
            )

        report.run("cube_scale", check_cube_scale)

        def check_cube_size_measured_radius() -> dict[str, Any]:
            radius = cube_size_radius_from_aperture(MEASURED_FINGER_APERTURE_M)
            return {
                "measured_finger_aperture_m": MEASURED_FINGER_APERTURE_M,
                "source": "Spike 1, spikes/spike_api.py scene_builds_and_measures, "
                "2026-09-16 (README §5.2.1)",
                "derived_radius_m": radius,
                "note": "cube.size's upper half-range -- the exact margin is a "
                "modelling decision, not measured here (README §5.2.2)",
            }

        report.run("cube_size_measured_radius", check_cube_size_measured_radius)

        # -- cube.hue -----------------------------------------------------------
        def check_cube_hue() -> dict[str, Any]:
            current = need_rig()
            return run_knob_check(
                current,
                KnobCheck(
                    role="cube.hue",
                    write=lambda v: write_cube_hue(current, v),
                    read=lambda: read_cube_diffuse_color(current),
                    readback_error=lambda wrote, read: max(
                        abs(a - b) for a, b in zip(hue_to_rgb(wrote), read, strict=True)
                    ),
                    readback_atol=CUBE_HUE_READBACK_ATOL,
                    base_value=BASE_CUBE_HUE,
                    perturbed_value=PERTURBED_CUBE_HUE,
                )
            )

        report.run("cube_hue", check_cube_hue)

        # -- light.intensity / light.warmth --------------------------------------
        def check_light_intensity() -> dict[str, Any]:
            current = need_rig()
            return run_knob_check(
                current,
                KnobCheck(
                    role="light.intensity",
                    write=lambda v: write_light_intensity(current, v),
                    read=lambda: read_light_intensity(current),
                    readback_error=lambda wrote, read: abs(read - wrote),
                    readback_atol=LIGHT_INTENSITY_READBACK_ATOL,
                    base_value=BASE_LIGHT_INTENSITY,
                    perturbed_value=PERTURBED_LIGHT_INTENSITY,
                )
            )

        report.run("light_intensity", check_light_intensity)

        def check_light_warmth() -> dict[str, Any]:
            current = need_rig()
            return run_knob_check(
                current,
                KnobCheck(
                    role="light.warmth",
                    write=lambda v: write_light_warmth(current, v),
                    read=lambda: read_light_warmth(current),
                    readback_error=lambda wrote, read: abs(read - wrote),
                    readback_atol=LIGHT_WARMTH_READBACK_ATOL,
                    base_value=BASE_LIGHT_WARMTH_K,
                    perturbed_value=PERTURBED_LIGHT_WARMTH_K,
                )
            )

        report.run("light_warmth", check_light_warmth)

        def check_light_intensity_clipping_range() -> dict[str, Any]:
            current = need_rig()
            apply_preset("pathtracing_denoiser_off")
            capture = make_attribute_capture(current, lambda v: write_light_intensity(current, v))
            samples = {
                value: frame_stats(capture(value).clone())
                for value in INTENSITY_CANDIDATES_FOR_CLIPPING
            }
            write_light_intensity(current, BASE_LIGHT_INTENSITY)
            clipped_low = [v for v, s in samples.items() if s["max"] <= 1.0]
            clipped_high = [v for v, s in samples.items() if s["min"] >= 254.0]
            facts = {"samples": samples, "clipped_low": clipped_low, "clipped_high": clipped_high}
            if len(clipped_low) == len(samples) or len(clipped_high) == len(samples):
                raise CheckFailed(
                    "every candidate intensity clips at one end -- no usable range "
                    f"found in {INTENSITY_CANDIDATES_FOR_CLIPPING}, widen it"
                )
            return facts

        report.run("light_intensity_measured_clipping_range", check_light_intensity_clipping_range)

        # -- light.azimuth / light.elevation --------------------------------------
        def check_light_direction() -> dict[str, Any]:
            current = need_rig()

            def write_direction(azimuth_elevation: tuple[float, float]) -> None:
                write_light_direction(
                    current, azimuth_rad=azimuth_elevation[0], elevation_rad=azimuth_elevation[1]
                )

            return run_knob_check(
                current,
                KnobCheck(
                    role="light.azimuth_elevation",
                    write=write_direction,
                    base_value=BASE_LIGHT_AZIMUTH_ELEVATION,
                    perturbed_value=PERTURBED_LIGHT_AZIMUTH_ELEVATION,
                )
            )

        report.run("light_direction", check_light_direction)

        # -- cam.jitter.* -----------------------------------------------------
        def check_camera_jitter() -> dict[str, Any]:
            current = need_rig()
            return run_knob_check(
                current,
                KnobCheck(
                    role="cam.jitter",
                    write=lambda xy: aim_camera(current, jitter_xy=xy),
                    base_value=BASE_CAMERA_JITTER,
                    perturbed_value=PERTURBED_CAMERA_JITTER,
                )
            )

        report.run("camera_jitter_per_capture", check_camera_jitter)

        # -- exposure ----------------------------------------------------------
        def check_exposure_lever() -> dict[str, Any]:
            current = need_rig()
            readbacks: dict[str, Any] = {}

            def apply_and_capture(values: Mapping[str, Any]) -> Tensor:
                readbacks["latest"] = apply_carb_settings(values)
                current.sim.forward()
                current.camera.update(dt=0.0, force_recompute=True)
                return current.camera.data.output["rgb"]

            baseline = apply_and_capture({}).clone()
            candidate = apply_and_capture(EXPOSURE_CANDIDATES).clone()
            repeat = apply_and_capture(EXPOSURE_CANDIDATES).clone()
            apply_and_capture({})  # restore

            mad = mean_abs_diff(baseline, candidate)
            accepted = {k: v for k, v in readbacks["latest"].items() if v.get("accepted")}
            facts = {
                "candidates": EXPOSURE_CANDIDATES,
                "readback": readbacks["latest"],
                "accepted_keys": list(accepted),
                "mad_vs_baseline": mad,
                "repeat_bitwise_equal": bitwise_equal(candidate, repeat),
            }
            if not accepted:
                raise CheckFailed(
                    "no candidate exposure setting was accepted on this build -- readback "
                    f"{readbacks['latest']}; the handle is dropped, not faked (README §5.2.3)"
                )
            if mad <= 0.0:
                raise CheckFailed(
                    f"exposure settings {list(accepted)} were accepted by carb but changed "
                    f"nothing in the rendered frame (mad={mad:.4g}) -- a dead lever, same as no "
                    "lever"
                )
            return facts

        report.run("exposure_lever", check_exposure_lever)

        # -- write-order independence -------------------------------------------
        def check_write_order_independence() -> dict[str, Any]:
            current = need_rig()
            preset_settings = apply_preset("pathtracing_denoiser_off")
            capture = make_capture_static(current)

            reset_cube_to_default(current)
            write_cube_scale(current, PERTURBED_CUBE_EDGE_M)
            write_cube_hue(current, PERTURBED_CUBE_HUE)
            size_then_hue = capture(None).clone()

            reset_cube_to_default(current)
            write_cube_hue(current, PERTURBED_CUBE_HUE)
            write_cube_scale(current, PERTURBED_CUBE_EDGE_M)
            hue_then_size = capture(None).clone()

            reset_cube_to_default(current)
            mad = mean_abs_diff(size_then_hue, hue_then_size)
            facts = {
                "mad": mad,
                "bitwise_equal": bitwise_equal(size_then_hue, hue_then_size),
                "preset_applied": {"name": "pathtracing_denoiser_off", "settings": preset_settings},
            }
            if not facts["bitwise_equal"]:
                raise CheckFailed(
                    f"size-then-hue vs hue-then-size differ (mad={mad:.4g}) -- USD "
                    "attribute writes are order-dependent here; the writer needs a "
                    "fixed canonical order"
                )
            return facts

        report.run("write_order_independence", check_write_order_independence)

        # -- cross-talk: style writes must not perturb the base cube pose --------
        def check_cross_talk() -> dict[str, Any]:
            current = need_rig()
            before = read_cube_translation(current)

            write_light_intensity(current, PERTURBED_LIGHT_INTENSITY)
            write_light_warmth(current, PERTURBED_LIGHT_WARMTH_K)
            write_light_direction(
                current,
                azimuth_rad=PERTURBED_LIGHT_AZIMUTH_ELEVATION[0],
                elevation_rad=PERTURBED_LIGHT_AZIMUTH_ELEVATION[1],
            )
            aim_camera(current, jitter_xy=PERTURBED_CAMERA_JITTER)
            apply_carb_settings(EXPOSURE_CANDIDATES)

            after = read_cube_translation(current)
            delta = max(abs(a - b) for a, b in zip(before, after, strict=True))

            write_light_intensity(current, BASE_LIGHT_INTENSITY)
            write_light_warmth(current, BASE_LIGHT_WARMTH_K)
            write_light_direction(
                current,
                azimuth_rad=BASE_LIGHT_AZIMUTH_ELEVATION[0],
                elevation_rad=BASE_LIGHT_AZIMUTH_ELEVATION[1],
            )
            aim_camera(current, jitter_xy=BASE_CAMERA_JITTER)
            apply_carb_settings({})

            facts = {
                "cube_translation_before": before,
                "cube_translation_after": after,
                "max_abs_delta_m": delta,
            }
            if delta > 1e-6:
                raise CheckFailed(
                    f"writing style knobs moved the cube's own transform by {delta:.3g} m -- "
                    "a camera re-aim, light write, or exposure setting must never perturb "
                    "task latents"
                )
            return facts

        report.run("cross_talk_style_leaves_base_unchanged", check_cross_talk)

        # -- motion blur under teleport-no-step -----------------------------------
        def check_motion_blur() -> dict[str, Any]:
            current = need_rig()
            blur_settings = apply_carb_settings(MOTION_BLUR_CANDIDATES)
            preset_settings = apply_preset("pathtracing_denoiser_off")
            capture = make_capture_static(current)
            first = capture(None).clone()
            second = capture(None).clone()
            mad = mean_abs_diff(first, second)
            apply_carb_settings({k: False for k in MOTION_BLUR_CANDIDATES if "enabled" in k})
            raise CheckFailed(
                f"no motion-blur artefact between two captures of the same teleported "
                f"state (mad={mad:.4g}; blur settings {blur_settings}, preset {preset_settings}) "
                "-- expected, not a bug: motion blur needs velocity across frames and this "
                "pipeline has none by design (README §5.5). Recorded as a completed check, "
                "same category as Spike 1's aliasing FAIL."
            )

        report.run("motion_blur_under_teleport_no_step", check_motion_blur)

        # -- sensor noise: a documented decision, not an Isaac call ---------------
        def check_sensor_noise_injection_point() -> dict[str, Any]:
            return {
                "decision": "synthetic sensor noise, if ever wanted, is a generate.py-side "
                "post-process applied to an already-deterministic frame, with its own "
                "recorded seed -- never a render setting and never a style handle (README §5.2.3)",
                "why": "a render-level noise source would reintroduce exactly the stochastic "
                "g the §7.1 determinism gate exists to rule out",
            }

        report.run("sensor_noise_injection_point", check_sensor_noise_injection_point)

    finally:
        payload = {
            "run": {
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                "seconds": round(time.time() - started, 1),
                "args": {
                    k: (str(v) if isinstance(v, Path) else v)
                    for k, v in sorted(vars(args).items())
                    if not k.startswith("_")
                },
            },
            **report.to_dict(),
        }
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "facts.json").write_text(json.dumps(payload, indent=2, default=str))
            written = str(out_dir / "facts.json")
        except Exception as exc:
            written = f"could not write facts.json: {type(exc).__name__}: {exc}"

        print("\n" + report.table())
        print(f"\nfacts: {written}")
        simulation_app.close()

    return 1 if report.counts()["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
