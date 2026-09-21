"""Spike 5: the attribute write paths for README §5.2's `full` and `style` groups.

Spikes 1-4 (``spikes/spike_api.py``) measured exactly two write paths --
``write_joint_state_to_sim`` and ``write_root_state_to_sim`` -- which is §5.2's
`base` group and nothing else. Every `full`/`style` latent moves through a
*different* mechanism: a geometry write for ``cube.size``, a USD material write
for ``cube.hue`` and ``table.roughness``/``table.albedo``, a light-attribute
write for ``light.*``, a per-capture camera re-aim for ``cam.jitter.*``, a
post-process setting for ``exposure``. None of those has been touched, and
none of Spike 1's verdicts transfers to them by default (docs/PLAN.md Phase 3b).

**``table.roughness``/``table.albedo`` were added later than everything else
in this file** (docs/PLAN.md Phase 4 follow-up) -- every prior round tested
the cube, a light and the camera, but no round ever built a table prim at
all, so README §5.2.3's two table knobs sat undeclared rather than merely
unresolved. ``build_rig()`` now spawns one; the two checks follow the same
three-question recipe as everything else via :func:`run_knob_check`.

**Retargeted at scene v1 (docs/PLAN.md Phase 2b).** Every prior round tested
this spike's own inline-built scene: one ``DistantLight`` and a flat 0.3x0.3
material-test table that was never actually load-bearing. ``build_rig()`` now
references ``scenes/stage_v1_tabletop.usda`` in whole instead -- real PBR
materials, an HDRI dome, two ``RectLight``s (``KeyLight``/``FillLight``) in
place of the single ``DistantLight``. Every read/write helper below turned
out to be generic enough (``UsdLux.LightAPI``, raw shader/xform manipulation,
never a light-type- or Isaac-Lab-asset-specific call) to need no changes of
its own; only ``build_rig()``'s scene construction and
``respawn_light_and_write_direction``'s respawned light type changed. Whether
``light.azimuth_elevation``/``table.roughness`` stay dead knobs against this
real rig, and whether ``cube.y``'s multi-view fix (README §5.3) is even in
scope here (this spike still has one camera), are exactly what this retarget
is for.

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

**Round 2 (docs/answer.md, docs/PLAN.md Phase 3c).** `cube.hue`, `light.*` and
`cam.jitter` came back non-deterministic under `standard` and four targeted
fixes were falsified (see README §7.5). An external research reply proposes a
different mechanism -- path-tracer caches and AA jitter, not sample count --
and names a version-matched candidate root cause
(`IsaacLab#6609 <https://github.com/isaac-sim/IsaacLab/issues/6609>`_). Four
new, independently togglable levers below (``--extra-preset``,
``--reset-pt-accum-on-time-change``, ``--reset-cadence-per-capture``,
``--capture-via``) test that, plus a standing diagnostic
(``render_product_attribute_audit``) that runs every time. Every carb key and
API name here was confirmed by reading real source -- Isaac Lab's own
``isaaclab_physx.renderers.isaac_rtx_renderer_utils.apply_isaac_rtx_determinism_settings``,
``isaaclab_physx.renderers.isaac_rtx_renderer`` (the ``omni:rtx:rendermode`` /
``"Minimal"`` per-product attribute), IsaacLab#6609 itself (the exact
``RenderContext`` method names), and OmniGibson's
``renderer_settings/path_tracing_settings.py`` (the offline-PathTracing carb
keys) -- not guessed, but never run against this scene either. ``none``/unset
on all four preserves the exact Round-1 behaviour that produced README §7.5's
verdict.
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

from idtb.scenegen.stage_v1 import (
    CAMERAS,
    CUBE_PRIM_PATH,
    CUBE_XY,
    DEFAULT_CUBE_EDGE_M,
    KEY_LIGHT_PRIM_PATH,
    TABLE_PRIM_PATH,
    TABLE_TOP_Z,
)

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


def diff_summary(a: Tensor, b: Tensor) -> dict[str, Any]:
    """Where two same-shape frames differ -- localizes a mad number to an actual
    pixel region, so "back-to-back mad=1.87" becomes something you can reason
    about without a GUI: a spatially uniform shift (every pixel off by ~1.87,
    consistent with a global exposure/accumulation artefact) reads completely
    differently from a handful of pixels off by 100+ (an edge/aliasing glitch)
    averaging out to the same mad.
    """
    diff = (a.detach().to(torch.float32) - b.detach().to(torch.float32)).abs()
    flat = diff.flatten()
    max_idx = int(torch.argmax(flat).item())
    coords = torch.unravel_index(torch.tensor(max_idx), diff.shape)
    return {
        "shape": list(diff.shape),
        "mean_abs_diff": float(flat.mean().item()),
        "std_abs_diff": float(flat.std().item()),
        "max_abs_diff": float(flat.max().item()),
        "fraction_pixels_changed_gt_1": float((flat > 1.0).float().mean().item()),
        "max_diff_at_index": [int(c) for c in coords],
        "value_a_at_max_diff": a.flatten()[max_idx].item(),
        "value_b_at_max_diff": b.flatten()[max_idx].item(),
    }


def dump_render_product_rtx_attributes(prim: Any) -> dict[str, Any]:
    """Every ``omni:rtx*``-namespaced attribute on a prim, name -> value.

    Isaac Sim 6.x sets render mode and path-tracing parameters *per
    RenderProduct* as USD attributes (``omni:rtx:rendermode``,
    ``omni:rtx:pt:samplesPerPixel``, ...) -- a separate mechanism from the
    global/deprecated carb keys ``standard`` sets. Reading a carb key back
    only proves carb stored it, not that this camera's RenderProduct used it
    (docs/answer.md, docs/PLAN.md Phase 3c experiment A). Takes any
    ``Usd.Prim``-like object with ``GetAttributes()`` -- duck-typed so this is
    testable without a live stage; the caller resolves the real prim.
    """
    values: dict[str, Any] = {}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if not name.startswith("omni:rtx"):
            continue
        try:
            values[name] = attr.Get()
        except Exception as exc:
            values[name] = f"<unreadable: {type(exc).__name__}: {exc}>"
    return values


# ---------------------------------------------------------------------------
# Isaac layer: every import lives inside a function, after SimulationApp exists.
# ---------------------------------------------------------------------------

# docs/PLAN.md Phase 2b: scene v1's authored stage replaces this spike's own
# Round 1-5 inline-built scene (one DistantLight, a flat 0.3x0.3 material-test
# table that was never actually load-bearing). `stage_v1_tabletop.usda` is
# referenced in whole at SCENE_ROOT (`build_rig`), so its own internal
# `/World/...` paths are recomposed under SCENE_ROOT, not under `/World`
# directly -- `_composed()` does that remapping from the single source of
# truth in `idtb.scenegen.stage_v1` rather than duplicating literal paths
# (imported at the top of the file, next to every other module-level import).

SCENE_ROOT = "/World/SceneV1"


def _composed(world_relative_path: str) -> str:
    assert world_relative_path.startswith("/World"), world_relative_path
    return SCENE_ROOT + world_relative_path[len("/World") :]


GROUND_PATH = "/World/Ground"  # stage_v1_tabletop.usda deliberately excludes one
# (docs/PLAN.md Phase 2's own scope, same reasoning as excluding the Franka) --
# spawned separately here, same as Phase 3 rework's build_rig() will need to.
CUBE_PATH = _composed(CUBE_PRIM_PATH)
TABLE_PATH = _composed(TABLE_PRIM_PATH)
LIGHT_PATH = _composed(KEY_LIGHT_PRIM_PATH)  # KeyLight -- stage_v1.py's own
# light.intensity/warmth/azimuth_elevation target; FillLight is fixed and
# non-latent, untouched here; the dome is ambient-only, not a per-sample knob.
CAMERA_PATH = "/World/Camera"  # freshly spawned, NOT one of the authored
# CameraL/CameraR stereo prims -- Isaac Lab's Camera sensor always spawns its
# own prim from a spawn config rather than wrapping one that already exists
# (confirmed building spikes/spike_scene_v1_view.py, docs/PLAN.md Phase 2).

CAMERA_EYE = CAMERAS[0].eye  # CameraL -- continuity with every prior round's placement
CAMERA_TARGET = CAMERAS[0].target

BASE_CUBE_EDGE_M = DEFAULT_CUBE_EDGE_M  # 0.06 m -- single source of truth now,
# was independently hardcoded to the same value before this retarget
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
LIGHT_DIRECTION_READBACK_ATOL = 1e-3  # unit-vector components

BASE_CAMERA_JITTER = (0.0, 0.0)
PERTURBED_CAMERA_JITTER = (0.05, -0.04)

# Table geometry (size, position) now comes from stage_v1_tabletop.usda
# itself, not a hardcoded constant here -- see build_rig().

BASE_TABLE_ROUGHNESS = 0.5
PERTURBED_TABLE_ROUGHNESS = 0.95
TABLE_ROUGHNESS_READBACK_ATOL = 1e-3

BASE_TABLE_ALBEDO = 0.5
PERTURBED_TABLE_ALBEDO = 0.15
TABLE_ALBEDO_READBACK_ATOL = 1e-3

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

# --- Round 2 (docs/answer.md, docs/PLAN.md Phase 3c) ------------------------
# Every key below was confirmed by reading real source, not guessed -- see the
# module docstring for exactly which file each one came from. Applied on top
# of `pathtracing_denoiser_off` (build_rig), never by default.

# Experiment B: offline PathTracing's own cross-frame caches and AA jitter,
# none of which the four already-falsified fixes (README §7.5) touched.
# Confirmed real carb keys (OmniGibson's renderer_settings/path_tracing_settings.py
# enumerates Isaac Sim's actual PathTracing settings UI, incl. resetPtAccumOnAnimTimeChange
# below) -- `adaptiveSampling/enabled` is the one key in this dict *not* corroborated
# there; apply_carb_settings()'s read-back `accepted` flag is what tells the two apart.
PATHTRACING_CACHES_AND_AA_OFF: dict[str, Any] = {
    "/rtx/pathtracing/cached/enabled": False,
    "/rtx/pathtracing/lightcache/cached/enabled": False,
    "/rtx/pathtracing/adaptiveSampling/enabled": False,
    "/rtx/pathtracing/fireflyFilter/enabled": False,
    "/rtx/pathtracing/aa/op": 0,
    "/rtx/pathtracing/aa/filterRadius": 0.0,
}

# Experiment D (partial) / an independent lead: Isaac Lab's own built-in
# "deterministic rendering" recipe -- isaaclab_physx.renderers.isaac_rtx_renderer_utils.
# apply_isaac_rtx_determinism_settings(), read directly from the isaac-sim/IsaacLab
# source (not the offline `standard` preset's PathTracing mode at all): RealTimePathTracing
# with its own *RTPT* cache namespace disabled -- `/rtx/rtpt/*`, distinct from
# `/rtx/pathtracing/*`. Spike 1 (§7.2) measured as-booted RealTimePathTracing
# non-deterministic, but never with these two caches off -- an untested combination,
# not a rerun of that result.
REALTIME_PATHTRACING_RTPT_CACHES_OFF: dict[str, Any] = {
    "/rtx/rendermode": "RealTimePathTracing",
    "/rtx/rtpt/cached/enabled": False,
    "/rtx/rtpt/lightcache/cached/enabled": False,
}

# Experiment D: the escape hatch. "Minimal" is the exact render-mode string
# Isaac Lab itself sets on a RenderProduct's `omni:rtx:rendermode` attribute
# for its own low-cost shading path (isaaclab_physx.renderers.isaac_rtx_renderer);
# applied here as a global carb value for a first cut, since a plain
# `/rtx/rendermode` write has worked for every other mode switch this spike
# and Spike 1 have tried. No Monte Carlo state, no caches, no accumulation.
MINIMAL_RENDER_MODE: dict[str, Any] = {"/rtx/rendermode": "Minimal"}

EXTRA_PRESETS: dict[str, dict[str, Any]] = {
    "none": {},
    "caches_and_aa_off": PATHTRACING_CACHES_AND_AA_OFF,
    "realtime_rtpt_caches_off": REALTIME_PATHTRACING_RTPT_CACHES_OFF,
    "minimal": MINIMAL_RENDER_MODE,
}

# Experiment C, part 1: forces the path tracer to treat every capture as a
# fresh scene instead of relying on change detection to notice one -- the
# accumulation restart research doc's leading theory says is
# change-detection-driven, i.e. exactly the mechanism that would need an
# actual animation-time change to fire, which this pipeline's zero-`sim.step()`
# policy (README §5.5) never produces.
RESET_ACCUM_ON_TIME_CHANGE: dict[str, Any] = {"/rtx/resetPtAccumOnAnimTimeChange": True}

# Experiment E (added after C1/D2 both independently produced the identical
# dead-knob signature for light.azimuth_elevation -- order-independent AND
# back-to-back mad both exactly 0.0, while light.intensity/warmth on the same
# prim stayed fully responsive under the same two configs). The light's
# rotation is a raw, non-Fabric-tracked USD xform op -- only the kinematic
# cube is Fabric-published (docs/answer.md's own diagnosis of the TypeScale-
# vs-TypeRotateXYZ split, README §7.5). This is the exact warning printed on
# every single boot, flagged as "the single best lead" in the original
# research task and never tested until now: a renderer path that prefers
# Fabric-published transforms may simply never see this prim's update at all
# under `resetPtAccumOnAnimTimeChange`/`minimal`'s faster code paths, freezing
# the light's rotation at scene-build time rather than under-converging it.
DISABLE_FABRIC_TRANSFORM_SYNC: dict[str, Any] = {
    "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False
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
    table_prim: Any
    table_shader: Any  # UsdShade.Shader, or None if binding resolution failed
    notes: dict[str, Any]
    render_tick: Callable[[], None]
    reset_cadence: Callable[[], None] | None = None


def build_rig(args: argparse.Namespace) -> Rig:
    """Build the rig against scene v1's authored stage (docs/PLAN.md Phase 2b):
    table, cube, dome + 2 area lights, referenced in from
    ``scenes/stage_v1_tabletop.usda`` rather than spawned inline -- replacing
    this spike's own Round 1-5 scene (one ``DistantLight`` and a flat 0.3x0.3
    material-test table that was never actually load-bearing).

    Referenced via ``sim_utils.UsdFileCfg`` at ``SCENE_ROOT`` as a whole prim
    reference, not by swapping the live USD stage wholesale
    (``omni.usd...open_stage()``, what ``spikes/spike_scene_v1_view.py`` uses)
    -- this rig needs physics (``SimulationContext``) and a real camera
    *sensor* coexisting with the authored content in one scene, which is also
    exactly how Phase 3 rework will load this file alongside a dynamically
    spawned Franka. No ``InteractiveScene``/env-namespace templating, same as
    every prior round -- this spike isolates `full`/`style` write paths, which
    don't need one.

    A ground plane is spawned separately: ``stage_v1_tabletop.usda``
    deliberately excludes one (docs/PLAN.md Phase 2's own scope, same
    reasoning as excluding the Franka).
    """
    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

    notes: dict[str, Any] = {}

    repo_root = Path(__file__).resolve().parent.parent
    stage_path = repo_root / "scenes" / "stage_v1_tabletop.usda"
    if not stage_path.exists():
        raise FileNotFoundError(f"scene v1 not found: {stage_path}")

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

    scene_cfg = sim_utils.UsdFileCfg(usd_path=str(stage_path))
    scene_cfg.func(SCENE_ROOT, scene_cfg)  # identity transform -- world coords
    # match exactly what's authored (table centered under the cube, etc.)

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
    table_prim = stage.GetPrimAtPath(TABLE_PATH)

    # Never fail fast (structural rule 1): if the reference-composition path
    # guess above is wrong, record what's actually under SCENE_ROOT instead of
    # crashing blind -- this is new integration surface (an external USD file
    # referenced alongside spawned content), not yet verified on the pod.
    if not (cube_prim.IsValid() and table_prim.IsValid() and light_prim.IsValid()):
        notes["composed_paths_valid"] = {
            "cube": cube_prim.IsValid(),
            "table": table_prim.IsValid(),
            "light": light_prim.IsValid(),
        }
        try:
            notes["scene_root_prim_tree"] = describe_prim_tree(stage.GetPrimAtPath(SCENE_ROOT))
        except Exception as exc:
            notes["scene_root_prim_tree"] = f"unavailable: {type(exc).__name__}: {exc}"

    cube_shader = None
    try:
        shader_name, cube_shader = resolve_bound_shader(cube_prim)
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

    table_shader = None
    try:
        shader_name, table_shader = resolve_bound_shader(table_prim)
        notes["table_shader_resolved_via"] = shader_name
    except Exception as exc:
        notes["table_shader_resolved_via"] = f"unresolved: {type(exc).__name__}: {exc}"
        try:
            notes["table_prim_tree"] = describe_prim_tree(table_prim)
        except Exception as tree_exc:
            notes["table_prim_tree"] = f"unavailable: {type(tree_exc).__name__}: {tree_exc}"

    # Applied exactly once, here -- not per check. Measured on the pod: every
    # check after the first re-applied this same preset redundantly (setting
    # carb keys to values they already had), and every one of those *repeat*
    # applications showed the same ~1.8 mad back-to-back noise regardless of
    # which attribute it was actually varying, while the one check that ran
    # before any repeat application (cube.size, the first to run) was clean.
    # That pattern -- reproducible bit-for-bit across separate process runs,
    # so not driver flakiness -- points at the redundant re-application
    # itself, not the individual attribute writes. Testing that directly.
    notes["preset_applied"] = {
        "name": "pathtracing_denoiser_off",
        "settings": apply_preset("pathtracing_denoiser_off"),
    }

    # -- Round 2 (docs/PLAN.md Phase 3c) -- every lever below is opt-in via a
    # CLI flag and defaults to a no-op, so an unflagged run reproduces exactly
    # the Round-1 configuration that produced README §7.5's verdict. --------
    extra_preset_name = getattr(args, "extra_preset", "none")
    if extra_preset_name != "none":
        notes["extra_preset"] = {
            "name": extra_preset_name,
            "settings": apply_carb_settings(EXTRA_PRESETS[extra_preset_name]),
        }

    if getattr(args, "reset_pt_accum_on_time_change", False):
        notes["reset_pt_accum_on_time_change"] = apply_carb_settings(RESET_ACCUM_ON_TIME_CHANGE)

    if getattr(args, "disable_fabric_transform_sync", False):
        notes["disable_fabric_transform_sync"] = apply_carb_settings(DISABLE_FABRIC_TRANSFORM_SYNC)

    if getattr(args, "reset_cadence_per_capture", False):
        cadence_desc, cadence_fn = resolve_cadence_reset(sim)
    else:
        cadence_desc, cadence_fn = "disabled (--reset-cadence-per-capture not set)", None
    notes["reset_cadence_per_capture"] = cadence_desc

    render_tick, render_tick_desc = resolve_render_tick(
        getattr(args, "capture_via", "sim_render"), sim
    )
    notes["capture_via"] = render_tick_desc

    rig = Rig(
        sim=sim,
        device=str(sim.device),
        camera=camera,
        cube_prim=cube_prim,
        cube_shader=cube_shader,
        light_prim=light_prim,
        table_prim=table_prim,
        table_shader=table_shader,
        notes=notes,
        render_tick=render_tick,
        reset_cadence=cadence_fn,
    )
    aim_camera(rig, jitter_xy=BASE_CAMERA_JITTER)
    camera.update(dt=0.0, force_recompute=True)
    return rig


def resolve_bound_shader(prim: Any) -> tuple[str, Any]:
    """Find the UsdShade.Shader driving ``prim``'s diffuse colour (or any other
    PreviewSurface input) -- generalised from cube-only, since the table
    material (README §5.2.3) is resolved the identical way, on a different prim.

    Tries the schema-correct lookup on ``prim`` itself first, then walks its
    whole subtree for a bound material -- measured on the pod: binding-API
    lookup on ``prim`` directly found nothing for the cube (docs/PLAN.md
    Phase 3b, first run), meaning the spawner most likely nests the actual
    visual geometry (and its binding) under a child prim rather than binding
    on ``prim`` itself. The recursive search finds it regardless of naming,
    which is more robust than guessing another literal path.
    """
    from pxr import Usd, UsdShade

    def via_binding_api() -> Any:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            raise RuntimeError("no bound material")
        source, _, _ = material.ComputeSurfaceSource()
        if not source:
            raise RuntimeError("material has no surface source")
        return source

    def via_recursive_binding_search() -> Any:
        for descendant in Usd.PrimRange(prim):
            material, _ = UsdShade.MaterialBindingAPI(descendant).ComputeBoundMaterial()
            if not material:
                continue
            source, _, _ = material.ComputeSurfaceSource()
            if source:
                return source
        raise RuntimeError("no descendant of the prim has a bound material with a surface")

    def via_looks_convention() -> Any:
        stage = prim.GetStage()
        for suffix in (
            "Looks/Material/Shader",
            "Looks/PreviewSurface/Shader",
            "Looks/visualMaterial/Shader",
        ):
            shader_prim = stage.GetPrimAtPath(prim.GetPath().AppendPath(suffix))
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


def get_or_create_shader_input(shader: Any, name: str, sdf_type: Any) -> Any:
    """``GetInput(name)`` if the shader already declares it, else ``CreateInput``.

    Isaac Lab's ``PreviewSurfaceCfg`` only authors ``diffuseColor`` explicitly
    (the pattern ``write_cube_hue`` already relies on) -- a standard
    ``UsdPreviewSurface`` input like ``roughness`` is defined by the schema but
    may not exist on *this* authored shader at all until something creates it.
    Duck-typed (only calls ``GetInput``/``CreateInput``) so it's testable
    without a live stage; the caller passes the real shader and an
    ``Sdf.ValueTypeNames`` constant.
    """
    existing = shader.GetInput(name)
    return existing if existing is not None else shader.CreateInput(name, sdf_type)


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


def resolve_render_tick(mode: str, sim: Any) -> tuple[Callable[[], None], str]:
    """The per-``depth``-iteration render call inside the capture loop.

    ``--capture-via app_update`` is docs/PLAN.md Phase 3c experiment C's third
    lever: swap ``sim.render()`` for ``omni.kit.app.get_app().update()``, in
    case the two tick the render graph differently. ``sim_render`` (the
    default) is exactly what Spike 1 and Round 1 of this spike already used --
    unchanged unless the flag is passed.
    """
    if mode == "app_update":
        import omni.kit.app

        app = omni.kit.app.get_app()
        return app.update, "omni.kit.app.get_app().update()"
    return sim.render, "sim.render()"


def resolve_cadence_reset(sim: Any) -> tuple[str, Callable[[], None] | None]:
    """Locate `IsaacLab#6609 <https://github.com/isaac-sim/IsaacLab/issues/6609>`_'s
    cadence-invalidation call on this build's ``SimulationContext``.

    Confirmed real API, not guessed: the issue's own reproduction reads
    ``env.sim.render_context._last_scene_state_step`` and its fix is
    ``RenderContext.reset_transform_cadence()`` (public since patch1 -- our
    exact Isaac Lab release) or ``reset_scene_state_cadence()`` (the eventual
    upstream fix, possibly not on this build yet). The bug's trigger is
    exactly our capture loop: ``sim.forward()`` with the physics-step count
    never advanced (README §5.5's zero-``sim.step()`` policy). Records which
    name answered, or that neither is reachable, rather than guessing a third.
    """
    render_context = getattr(sim, "render_context", None)
    if render_context is None:
        return "sim.render_context not found on this build", None
    for name in ("reset_transform_cadence", "reset_scene_state_cadence"):
        method = getattr(render_context, name, None)
        if callable(method):
            return f"sim.render_context.{name}()", method
    return (
        "sim.render_context found but neither reset_transform_cadence nor "
        "reset_scene_state_cadence is callable on it",
        None,
    )


def resolve_render_product_path(rig: Rig) -> tuple[str, str]:
    """The USD prim path of the camera's RenderProduct -- for experiment A's audit.

    No single accessor is confirmed across Isaac Lab's renderer backends: the
    PhysX backend stores it at ``camera._render_data.render_product.path``
    (``isaaclab_physx.renderers.isaac_rtx_renderer.IsaacRtxRenderData``), the
    OV/OVRTX backend instead at ``camera._renderer._render_product_paths[0]``
    (``isaaclab_ov.renderers.ovrtx_renderer.OVRTXRenderer``), and some releases
    expose a public ``camera.render_product_paths``. Tried in that order, with
    a schema-level fallback that doesn't depend on internal attribute names at
    all: search the whole stage for a ``UsdRender.Product`` prim, which any
    backend must create for Hydra to render through.
    """
    camera = rig.camera

    def via_public_attr() -> str:
        return camera.render_product_paths[0]

    def via_physx_render_data() -> str:
        return camera._render_data.render_product.path

    def via_ov_renderer_paths() -> str:
        return camera._renderer._render_product_paths[0]

    def via_stage_search() -> str:
        from pxr import Usd, UsdRender

        stage = rig.cube_prim.GetStage()
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            if prim.IsA(UsdRender.Product):
                return prim.GetPath().pathString
        raise RuntimeError("no UsdRender.Product prim found on the stage")

    return try_candidates(
        [
            ("camera.render_product_paths", via_public_attr),
            ("physx_render_data.render_product.path", via_physx_render_data),
            ("ov_renderer._render_product_paths", via_ov_renderer_paths),
            ("stage_search_UsdRender.Product", via_stage_search),
        ]
    )


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


def read_light_direction(rig: Rig) -> tuple[float, float, float]:
    """World-space direction the light emits along (its local -Z, composed
    through the prim's transform) -- the ground-truth read-back for
    :func:`write_light_direction`, added after external research
    (docs/research_task_light_direction_dead_knob.md) flagged that this is the
    one knob in this file whose write was never actually verified to land in
    USD. Every other knob has a `read`/`readback_error` pair in its
    `KnobCheck` (cube.size's bbox, cube.hue's shader input, light.intensity/
    warmth's attributes); `light.azimuth_elevation` did not, so a silently
    orphaned xform op (not in `xformOpOrder`) or a wrong `Gf.Rotation.Decompose`
    angle order could have gone undetected the whole time -- exactly the
    `_find_or_add_xform_op` failure mode already documented for `cube.size`,
    just never checked here.
    """
    from pxr import Gf, UsdGeom

    matrix = UsdGeom.Xformable(rig.light_prim).ComputeLocalToWorldTransform(0)
    world_direction = matrix.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
    return (float(world_direction[0]), float(world_direction[1]), float(world_direction[2]))


def read_table_roughness(rig: Rig) -> float:
    """Read-back for :func:`write_table_roughness` -- ``roughness`` only
    exists on the shader once something has created the input (see
    :func:`get_or_create_shader_input`), which the write path always does
    before this is ever called in :func:`run_knob_check`."""
    value = rig.table_shader.GetInput("roughness").Get()
    return float(value) if value is not None else float("nan")


def read_table_albedo(rig: Rig) -> float:
    """``table.albedo`` is one scalar (README §5.2.3, same reasoning as
    ``cube.hue`` being one hue axis rather than three RGB channels) -- authored
    as a grey ``diffuseColor`` with r == g == b, so reading channel 0 is enough."""
    value = rig.table_shader.GetInput("diffuseColor").Get()
    return float(value[0])


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
    if op_type == UsdGeom.XformOp.TypeOrient:
        return xformable.AddOrientOp()
    raise ValueError(f"no add-op helper wired up for {op_type!r}")


def write_cube_scale(rig: Rig, edge_m: float) -> None:
    """``cube.size`` via an xform scale multiplier on top of the authored edge --
    not a root-state write (README §5.2.2).

    The multiplier is relative to whatever edge length is actually baked into
    the cube's own ``UsdGeom.Cube`` ``size`` attribute, not assumed to equal
    ``BASE_CUBE_EDGE_M``. **Confirmed on the pod (Phase 2b):** the old spike
    scene's `CuboidCfg`-spawned cube bakes its geometry at `BASE_CUBE_EDGE_M`
    directly (its scale op is a pure multiplier on top), but stage_v1's cube
    (`idtb.scenegen.stage_v1::_set_box_prim`) is a *unit* `UsdGeom.Cube`
    (baked size `1.0`) with its own separate scale op that directly encodes
    the world size in meters. Assuming the old scene's convention set the
    scale straight to `edge_m / BASE_CUBE_EDGE_M` (e.g. `1.5` for a wanted
    `0.09` m edge) instead of the intended `0.09` -- read-back caught it
    (`cube_scale`: "read back 1.5 for 0.09").
    """
    from pxr import Gf, UsdGeom

    baked_size = float(UsdGeom.Cube(rig.cube_prim).GetSizeAttr().Get())
    factor = edge_m / baked_size
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


def write_table_roughness(rig: Rig, roughness: float) -> None:
    """``table.roughness`` via the shader's own ``roughness`` input, the same
    technique as ``write_cube_hue``'s ``diffuseColor`` write -- raw ``pxr``,
    not an Isaac-Lab-version-specific convenience wrapper (README §7.5's
    stated preference). Unlike ``diffuseColor``, ``PreviewSurfaceCfg`` never
    authors ``roughness`` explicitly, so :func:`get_or_create_shader_input`
    creates it on first write rather than assuming it already exists.
    """
    if rig.table_shader is None:
        raise CheckFailed("table shader was never resolved -- see scene_builds_and_measures")
    from pxr import Sdf

    get_or_create_shader_input(rig.table_shader, "roughness", Sdf.ValueTypeNames.Float).Set(
        float(roughness)
    )


def write_table_albedo(rig: Rig, albedo: float) -> None:
    if rig.table_shader is None:
        raise CheckFailed("table shader was never resolved -- see scene_builds_and_measures")
    from pxr import Gf

    rig.table_shader.GetInput("diffuseColor").Set(Gf.Vec3f(albedo, albedo, albedo))


def reset_table_to_default(rig: Rig) -> None:
    if rig.table_shader is not None:
        write_table_albedo(rig, BASE_TABLE_ALBEDO)
        write_table_roughness(rig, BASE_TABLE_ROUGHNESS)


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
    rotation is built as "take -Z to the wanted direction" and set directly as
    a quaternion (``TypeOrient``) -- **not** decomposed into XYZ Euler angles.

    This replaces an earlier version that went through
    ``Gf.Rotation.Decompose(XAxis, YAxis, ZAxis)`` into a ``TypeRotateXYZ`` op.
    That was a real, confirmed bug, not a hypothetical one: external research
    (docs/research_task_light_direction_dead_knob.md) flagged
    ``Decompose``'s angle order as not necessarily matching ``TypeRotateXYZ``'s
    application order, and read-back evidence on the pod confirmed it exactly
    -- writing (azimuth=120°, elevation=55°) read back as roughly
    (azimuth=9°, elevation=55°): elevation round-tripped exactly, azimuth was
    scrambled. That's almost certainly why README §7.5's `light_direction`
    "dead knob" was never actually about renderer determinism at all -- the
    real perturbation was much smaller than the intended one, small enough to
    sit at or below whatever noise floor happened to be measuring it. A
    quaternion has no axis-order ambiguity to get wrong: ``ComputeLocalToWorldTransform``
    reading back a `TypeOrient` op is mathematically guaranteed to reproduce
    the exact rotation that was set, by construction.
    """
    from pxr import Gf, UsdGeom

    direction = azel_to_direction(azimuth_rad, elevation_rad)
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*direction))
    quat = rotation.GetQuat()
    op = _find_or_add_xform_op(rig.light_prim, UsdGeom.XformOp.TypeOrient)
    # Precision of an authored `orient` op can be float or double depending on
    # how it was added -- "resolve, don't guess" (README §7.5) applies here
    # too, cheaply, rather than assuming AddOrientOp()'s default forever.
    try_candidates(
        [
            ("Quatf", lambda: op.Set(Gf.Quatf(quat))),
            ("Quatd", lambda: op.Set(Gf.Quatd(quat))),
        ]
    )
    return direction


def respawn_light_and_write_direction(
    rig: Rig, *, azimuth_rad: float, elevation_rad: float
) -> tuple[float, ...]:
    """Destroy and recreate the light prim, then set its rotation on the fresh copy.

    Experiment F (docs/PLAN.md Phase 3c, following up on §7.5's confirmed dead
    knob): tests whether the zero-pixel-effect result under
    `--reset-pt-accum-on-time-change`/`--extra-preset minimal` is a stale
    acceleration structure or shadow cache keyed to the *prim's identity*
    rather than its transform value -- a full respawn forces whatever
    per-light structure the renderer builds to rebuild from scratch, which a
    live `.Set()` on a long-lived prim does not. Deliberately expensive
    (a full USD prim destroy+recreate every capture, not a cheap attribute
    write) -- this is a diagnostic to localize the mechanism, not a candidate
    for the real per-sample write path even if it turns out to work.

    **Retargeted for Phase 2b:** `KeyLight` is now a `RectLight`
    (`idtb.scenegen.stage_v1`), not the `DistantLightCfg` earlier rounds
    respawned -- `isaaclab.sim` has no `RectLightCfg` (docs/PLAN.md Phase 2's
    own docs-research finding), so this authors the raw `pxr.UsdLux.RectLight`
    directly, matching `stage_v1.py::_define_rect_light`'s own parameters and
    local translate exactly (a *local* transform, unaffected by this prim's
    absolute path being under `SCENE_ROOT` now rather than `/World` directly).
    """
    from pxr import Gf, UsdGeom, UsdLux

    stage = rig.light_prim.GetStage()
    light_path = rig.light_prim.GetPath()
    stage.RemovePrim(light_path)

    key_light_local_translate = (CUBE_XY[0] - 0.3, CUBE_XY[1] - 0.3, TABLE_TOP_Z + 1.0)
    light = UsdLux.RectLight.Define(stage, light_path)
    light.CreateIntensityAttr(BASE_LIGHT_INTENSITY)
    light.CreateWidthAttr(0.4)
    light.CreateHeightAttr(0.4)
    light.CreateEnableColorTemperatureAttr(True)
    light.CreateColorTemperatureAttr(BASE_LIGHT_WARMTH_K)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*key_light_local_translate))
    rig.light_prim = stage.GetPrimAtPath(light_path)

    return write_light_direction(rig, azimuth_rad=azimuth_rad, elevation_rad=elevation_rad)


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


def make_capture_static(rig: Rig, *, depth: int = 1) -> Capture:
    """A capture with no write of its own -- for checks that mutate state
    explicitly before calling it (write-order independence, cross-talk).

    The §4.5 sequence, matching ``spike_api.py``'s proven-working one: flush,
    ``sim.render()`` ``depth`` times -- the call that actually produces a new
    frame, missing from an earlier version of this function -- then a final
    ``camera.update()`` to pull it. Spike 1 measured that one render call
    suffices for *physics-state* writes once `standard`'s carb settings are
    applied (§7.2 Spike 3); that measurement never covered material/light
    attribute writes, and README §7.4's Known-limit note already predicted N
    "almost certainly won't transfer" -- measured on the pod: every attribute
    write showed non-zero back-to-back noise at ``depth=1`` while an unchanged
    physics state didn't, consistent with a shader/material rebuild not
    finishing within a single render call. ``depth`` is a CLI flag
    (``--render-depth``) for exactly that reason -- it's an open question,
    not a constant.

    The render tick itself (``rig.render_tick``, ``sim.render()`` by default)
    and an optional cadence-invalidation call before it (``rig.reset_cadence``)
    are both resolved once in :func:`build_rig` from CLI flags -- docs/PLAN.md
    Phase 3c experiment C.
    """

    def capture(_state: Any) -> Tensor:
        rig.sim.forward()
        if rig.reset_cadence is not None:
            # Experiment C (docs/PLAN.md Phase 3c): invalidate IsaacLab#6609's
            # scene-state cadence on every capture, matching the bug's exact
            # trigger -- forward() with no sim.step() to advance the count.
            rig.reset_cadence()
        rig.camera.update(dt=0.0, force_recompute=True)
        for _ in range(depth):
            rig.render_tick()
        rig.camera.update(dt=0.0, force_recompute=True)
        return rig.camera.data.output["rgb"]

    return capture


def make_attribute_capture(rig: Rig, writer: Callable[[Any], None], *, depth: int = 1) -> Capture:
    """A capture closure for a single-attribute write path (README §6.3's dispatch
    is by write path; this spike only ever varies one attribute per check).

    A discard-the-first-render "warm-up" variant was tried and reverted: it
    did not fix the ~1.8 mad back-to-back noise on material/light/camera
    writes, and it *regressed* cube.size, which had been bitwise-clean
    (0.0/0.0) at depth=8 with a single render. That a change to the capture
    sequence made a previously-clean case worse, rather than leaving it
    alone, is itself evidence against a capture-sequence explanation --
    consistent with spikes/spike_api.py's independently confirmed finding
    that this exact driver/renderer is flaky run-to-run (an identical
    ``--num-envs 1`` run failed once and passed once, unchanged). See
    docs/PLAN.md Phase 3b for the reproducibility test this predicts.
    """
    static_capture = make_capture_static(rig, depth=depth)

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


def run_knob_check(rig: Rig, knob: KnobCheck, *, depth: int = 1) -> dict[str, Any]:
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
    # The preset is applied once, at scene build -- not here. See build_rig's
    # note on why a *repeat* application was itself the likely noise source.
    capture = make_attribute_capture(rig, knob.write, depth=depth)
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
    facts["preset_applied"] = rig.notes.get("preset_applied")
    if not det["content_reproducible"]:
        problems.append(
            f"{knob.role}: not bitwise deterministic under `standard` while varying it "
            f"(order-independence {det['order_independent_mad']:.4g}, "
            f"back-to-back {det['back_to_back_mad']:.4g})"
        )

    knob.write(knob.base_value)  # leave the rig as found for whichever check runs next
    if problems:
        # `facts` attached: a FAIL is exactly when the sensitivity/determinism
        # numbers behind the message matter most, and Report.run() previously
        # discarded them on every failing check -- fixed at the source
        # (spike_api.CheckFailed), not patched around here.
        raise CheckFailed("; ".join(problems), facts=facts)
    return facts


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("/idtb/data/spike_attrs"))
    parser.add_argument("--resolution", type=int, nargs=2, default=(128, 128), metavar=("H", "W"))
    parser.add_argument(
        "--render-depth",
        type=int,
        default=8,
        help="sim.render() calls per capture -- Spike 1 found 1 suffices for physics-state "
        "writes; measured on the pod that material/light attribute writes show non-zero "
        "back-to-back noise at depth=1, consistent with README §7.4's Known-limit note "
        "that N 'almost certainly won't transfer' to a different write path",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="save the a1/b1/b2/a2 frames determinism_report compares, for a few "
        "representative knobs, as <out>/frames/*.pt -- so the back-to-back noise found "
        "on cube.hue/light.*/cam.jitter can be inspected directly instead of guessed at",
    )
    parser.add_argument(
        "--extra-preset",
        choices=sorted(EXTRA_PRESETS),
        default="none",
        help="docs/PLAN.md Phase 3c: an additional carb config applied once at scene "
        "build, on top of `pathtracing_denoiser_off`. 'none' (default) reproduces "
        "exactly the Round-1 configuration that produced README §7.5's verdict; "
        "'caches_and_aa_off' is experiment B, 'realtime_rtpt_caches_off' and 'minimal' "
        "are experiment D's two candidates.",
    )
    parser.add_argument(
        "--reset-pt-accum-on-time-change",
        action="store_true",
        help="docs/PLAN.md Phase 3c experiment C: set "
        "/rtx/resetPtAccumOnAnimTimeChange=True once at scene build.",
    )
    parser.add_argument(
        "--reset-cadence-per-capture",
        action="store_true",
        help="docs/PLAN.md Phase 3c experiment C: call IsaacLab#6609's cadence-"
        "invalidation method (sim.render_context.reset_transform_cadence() or "
        "reset_scene_state_cadence()) before every capture's render step.",
    )
    parser.add_argument(
        "--capture-via",
        choices=["sim_render", "app_update"],
        default="sim_render",
        help="docs/PLAN.md Phase 3c experiment C: swap the capture loop's render tick "
        "from sim.render() (default) to omni.kit.app.get_app().update().",
    )
    parser.add_argument(
        "--disable-fabric-transform-sync",
        action="store_true",
        help="docs/PLAN.md Phase 3c experiment E: set "
        "/rtx/hydra/readTransformsFromFabricInRenderDelegate=False once at scene build -- "
        "tests whether a Fabric-preferring transform path is why light.azimuth_elevation "
        "(a non-Fabric-tracked prim) goes fully unresponsive under --extra-preset minimal "
        "and --reset-pt-accum-on-time-change while light.intensity/warmth do not.",
    )
    parser.add_argument(
        "--respawn-light-for-direction",
        action="store_true",
        help="docs/PLAN.md Phase 3c experiment F: for the light_direction check only, "
        "destroy and recreate the light prim before writing its rotation on the fresh "
        "copy, instead of mutating the existing prim's xform op in place -- tests whether "
        "the confirmed zero-pixel-effect result (README §7.5) is a stale per-light "
        "acceleration structure/shadow cache keyed to prim identity, not the transform "
        "value. Diagnostic only: expensive, not a candidate for the real write path.",
    )
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

        # -- experiment A: is `standard` even the config the render path reads? ---
        def check_render_product_audit() -> dict[str, Any]:
            current = need_rig()
            via, product_path = resolve_render_product_path(current)
            stage = current.cube_prim.GetStage()
            prim = stage.GetPrimAtPath(product_path)
            per_product = (
                dump_render_product_rtx_attributes(prim) if prim and prim.IsValid() else {}
            )
            facts = {
                "render_product_path": product_path,
                "resolved_via": via,
                "per_product_omni_rtx_attributes": per_product,
                "carb_preset_applied": current.notes.get("preset_applied"),
                "carb_extra_preset": current.notes.get("extra_preset"),
                "note": "compares the per-RenderProduct omni:rtx:* attributes Isaac Sim 6.x "
                "actually reads against the global/deprecated carb keys `standard` sets -- "
                "carb accepting a key only proves carb stored it, not that this camera's "
                "render path used it (docs/answer.md, docs/PLAN.md Phase 3c experiment A)",
            }
            if not per_product:
                raise CheckFailed(
                    f"resolved a RenderProduct prim at {product_path!r} via {via!r} but it "
                    "carries no omni:rtx:* attributes -- either carb settings are the only "
                    "mechanism actually reachable on this build, or this is the wrong prim",
                    facts=facts,
                )
            return facts

        report.run("render_product_attribute_audit", check_render_product_audit)

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
                depth=args.render_depth,
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
                ),
                depth=args.render_depth,
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
                ),
                depth=args.render_depth,
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
                ),
                depth=args.render_depth,
            )

        report.run("light_warmth", check_light_warmth)

        def check_light_intensity_clipping_range() -> dict[str, Any]:
            current = need_rig()
            capture = make_attribute_capture(
                current, lambda v: write_light_intensity(current, v), depth=args.render_depth
            )
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
                    f"found in {INTENSITY_CANDIDATES_FOR_CLIPPING}, widen it",
                    facts=facts,
                )
            return facts

        report.run("light_intensity_measured_clipping_range", check_light_intensity_clipping_range)

        # -- light.azimuth / light.elevation --------------------------------------
        def check_light_direction() -> dict[str, Any]:
            current = need_rig()

            def write_direction(azimuth_elevation: tuple[float, float]) -> None:
                if getattr(args, "respawn_light_for_direction", False):
                    respawn_light_and_write_direction(
                        current,
                        azimuth_rad=azimuth_elevation[0],
                        elevation_rad=azimuth_elevation[1],
                    )
                else:
                    write_light_direction(
                        current,
                        azimuth_rad=azimuth_elevation[0],
                        elevation_rad=azimuth_elevation[1],
                    )

            return run_knob_check(
                current,
                KnobCheck(
                    role="light.azimuth_elevation",
                    write=write_direction,
                    read=lambda: read_light_direction(current),
                    readback_error=lambda wrote, read: max(
                        abs(a - b) for a, b in zip(azel_to_direction(*wrote), read, strict=True)
                    ),
                    readback_atol=LIGHT_DIRECTION_READBACK_ATOL,
                    base_value=BASE_LIGHT_AZIMUTH_ELEVATION,
                    perturbed_value=PERTURBED_LIGHT_AZIMUTH_ELEVATION,
                ),
                depth=args.render_depth,
            )

        report.run("light_direction", check_light_direction)

        # -- external research follow-up (docs/research_task_light_direction_dead_knob.md) --
        # E3: does rotating the light produce a real effect only when paired with a
        # trivial parameter (intensity) write? Tests the "Hydra separates DirtyTransform
        # from DirtyParams, and the render config we're using drops the former" theory.
        # Diagnostic only -- reports numbers, never fails.
        def check_light_direction_needs_dirty_param() -> dict[str, Any]:
            current = need_rig()
            capture = make_capture_static(current, depth=args.render_depth)
            nudge_factor = 1.000001  # negligible on its own; just forces a param write

            def set_state(azel: tuple[float, float], nudged: bool) -> None:
                write_light_direction(current, azimuth_rad=azel[0], elevation_rad=azel[1])
                intensity = BASE_LIGHT_INTENSITY * nudge_factor if nudged else BASE_LIGHT_INTENSITY
                write_light_intensity(current, intensity)

            set_state(BASE_LIGHT_AZIMUTH_ELEVATION, False)
            frame_base = capture(None).clone()
            set_state(BASE_LIGHT_AZIMUTH_ELEVATION, True)
            frame_base_nudged = capture(None).clone()
            set_state(PERTURBED_LIGHT_AZIMUTH_ELEVATION, False)
            frame_perturbed = capture(None).clone()
            set_state(PERTURBED_LIGHT_AZIMUTH_ELEVATION, True)
            frame_perturbed_nudged = capture(None).clone()
            set_state(BASE_LIGHT_AZIMUTH_ELEVATION, False)  # restore

            mad_rotation_alone = mean_abs_diff(frame_base, frame_perturbed)
            mad_nudge_alone = mean_abs_diff(frame_base, frame_base_nudged)
            mad_rotation_with_nudge = mean_abs_diff(frame_base_nudged, frame_perturbed_nudged)
            return {
                "mad_rotation_alone": mad_rotation_alone,
                "mad_nudge_alone": mad_nudge_alone,
                "mad_rotation_plus_nudge_vs_nudge_baseline": mad_rotation_with_nudge,
                "coupled_effect_beyond_nudge": mad_rotation_with_nudge - mad_nudge_alone,
            }

        report.run(
            "light_direction_needs_dirty_param_write", check_light_direction_needs_dirty_param
        )

        # E4: rotate a non-light mesh (the ground plane, tilted about X so it's
        # visually distinguishable -- a rotation about its own normal, Z, would be
        # a no-op for a flat plane regardless of the renderer) under the same
        # config, to separate "rotation writes are broken in general" from
        # "this is specific to lights". Diagnostic only.
        def check_mesh_rotation_control() -> dict[str, Any]:
            from pxr import Gf, UsdGeom

            current = need_rig()
            stage = current.cube_prim.GetStage()
            ground_prim = stage.GetPrimAtPath(GROUND_PATH)
            capture = make_capture_static(current, depth=args.render_depth)

            def write_ground_tilt(degrees_about_x: float) -> None:
                op = _find_or_add_xform_op(ground_prim, UsdGeom.XformOp.TypeRotateXYZ)
                op.Set(Gf.Vec3f(degrees_about_x, 0.0, 0.0))

            write_ground_tilt(0.0)
            frame_base = capture(None).clone()
            write_ground_tilt(10.0)
            frame_tilted = capture(None).clone()
            write_ground_tilt(0.0)  # restore

            return {
                "mad_ground_tilt": mean_abs_diff(frame_base, frame_tilted),
                "note": "rotating the ground plane (a mesh, not a light) under the same "
                "render config as light_direction -- distinguishes 'rotation writes are "
                "broken in general' from 'this is specific to lights'",
            }

        report.run("mesh_rotation_control", check_mesh_rotation_control)

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
                ),
                depth=args.render_depth,
            )

        report.run("camera_jitter_per_capture", check_camera_jitter)

        # -- table.roughness / table.albedo -------------------------------------
        # README §5.2.3: never spiked before this addition (docs/PLAN.md Phase 4
        # follow-up) -- no earlier run of this file built a table prim at all.
        def check_table_roughness() -> dict[str, Any]:
            current = need_rig()
            return run_knob_check(
                current,
                KnobCheck(
                    role="table.roughness",
                    write=lambda v: write_table_roughness(current, v),
                    read=lambda: read_table_roughness(current),
                    readback_error=lambda wrote, read: abs(read - wrote),
                    readback_atol=TABLE_ROUGHNESS_READBACK_ATOL,
                    base_value=BASE_TABLE_ROUGHNESS,
                    perturbed_value=PERTURBED_TABLE_ROUGHNESS,
                ),
                depth=args.render_depth,
            )

        report.run("table_roughness", check_table_roughness)

        def check_table_albedo() -> dict[str, Any]:
            current = need_rig()
            return run_knob_check(
                current,
                KnobCheck(
                    role="table.albedo",
                    write=lambda v: write_table_albedo(current, v),
                    read=lambda: read_table_albedo(current),
                    readback_error=lambda wrote, read: abs(read - wrote),
                    readback_atol=TABLE_ALBEDO_READBACK_ATOL,
                    base_value=BASE_TABLE_ALBEDO,
                    perturbed_value=PERTURBED_TABLE_ALBEDO,
                ),
                depth=args.render_depth,
            )

        report.run("table_albedo", check_table_albedo)

        # -- exposure ----------------------------------------------------------
        def check_exposure_lever() -> dict[str, Any]:
            current = need_rig()
            static_capture = make_capture_static(current, depth=args.render_depth)

            def apply_and_capture(values: Mapping[str, Any]) -> tuple[dict[str, Any], Tensor]:
                readback = apply_carb_settings(values)
                return readback, static_capture(None).clone()

            _, baseline = apply_and_capture({})
            exposure_readback, candidate = apply_and_capture(EXPOSURE_CANDIDATES)
            _, repeat = apply_and_capture(EXPOSURE_CANDIDATES)
            apply_and_capture({})  # restore; its own readback isn't needed again

            mad = mean_abs_diff(baseline, candidate)
            accepted = {k: v for k, v in exposure_readback.items() if v.get("accepted")}
            facts = {
                "candidates": EXPOSURE_CANDIDATES,
                "readback": exposure_readback,
                "accepted_keys": list(accepted),
                "mad_vs_baseline": mad,
                "repeat_bitwise_equal": bitwise_equal(candidate, repeat),
            }
            if not accepted:
                raise CheckFailed(
                    "no candidate exposure setting was accepted on this build -- readback "
                    f"{exposure_readback}; the handle is dropped, not faked (README §5.2.3)",
                    facts=facts,
                )
            if mad <= 0.0:
                raise CheckFailed(
                    f"exposure settings {list(accepted)} were accepted by carb but changed "
                    f"nothing in the rendered frame (mad={mad:.4g}) -- a dead lever, same as no "
                    "lever",
                    facts=facts,
                )
            return facts

        report.run("exposure_lever", check_exposure_lever)

        # -- write-order independence -------------------------------------------
        def check_write_order_independence() -> dict[str, Any]:
            current = need_rig()
            capture = make_capture_static(current, depth=args.render_depth)

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
                "preset_applied": current.notes.get("preset_applied"),
            }
            if not facts["bitwise_equal"]:
                raise CheckFailed(
                    f"size-then-hue vs hue-then-size differ (mad={mad:.4g}) -- USD "
                    "attribute writes are order-dependent here; the writer needs a "
                    "fixed canonical order",
                    facts=facts,
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
            if current.table_shader is not None:
                write_table_roughness(current, PERTURBED_TABLE_ROUGHNESS)
                write_table_albedo(current, PERTURBED_TABLE_ALBEDO)

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
            reset_table_to_default(current)

            facts = {
                "cube_translation_before": before,
                "cube_translation_after": after,
                "max_abs_delta_m": delta,
            }
            if delta > 1e-6:
                raise CheckFailed(
                    f"writing style knobs moved the cube's own transform by {delta:.3g} m -- "
                    "a camera re-aim, light write, or exposure setting must never perturb "
                    "task latents",
                    facts=facts,
                )
            return facts

        report.run("cross_talk_style_leaves_base_unchanged", check_cross_talk)

        # -- motion blur under teleport-no-step -----------------------------------
        def check_motion_blur() -> dict[str, Any]:
            current = need_rig()
            blur_settings = apply_carb_settings(MOTION_BLUR_CANDIDATES)
            capture = make_capture_static(current, depth=args.render_depth)
            first = capture(None).clone()
            second = capture(None).clone()
            mad = mean_abs_diff(first, second)
            apply_carb_settings({k: False for k in MOTION_BLUR_CANDIDATES if "enabled" in k})
            facts = {
                "mad": mad,
                "blur_settings": blur_settings,
                "preset_applied": current.notes.get("preset_applied"),
            }
            raise CheckFailed(
                f"no motion-blur artefact between two captures of the same teleported "
                f"state (mad={mad:.4g}; blur settings {blur_settings}, "
                f"preset {current.notes.get('preset_applied')}) "
                "-- expected, not a bug: motion blur needs velocity across frames and this "
                "pipeline has none by design (README §5.5). Recorded as a completed check, "
                "same category as Spike 1's aliasing FAIL.",
                facts=facts,
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

        # -- optional: the actual a1/b1/b2/a2 frames, for a few representative knobs --
        if args.save_frames:

            def check_save_frames() -> dict[str, Any]:
                current = need_rig()
                frames_dir = out_dir / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                knobs: list[tuple[str, Callable[[Any], None], Any, Any]] = [
                    (
                        "cube.size",
                        lambda v: write_cube_scale(current, v),
                        BASE_CUBE_EDGE_M,
                        PERTURBED_CUBE_EDGE_M,
                    ),
                    (
                        "cube.hue",
                        lambda v: write_cube_hue(current, v),
                        BASE_CUBE_HUE,
                        PERTURBED_CUBE_HUE,
                    ),
                    (
                        "light.intensity",
                        lambda v: write_light_intensity(current, v),
                        BASE_LIGHT_INTENSITY,
                        PERTURBED_LIGHT_INTENSITY,
                    ),
                    (
                        "light.azimuth_elevation",
                        lambda azel: write_light_direction(
                            current, azimuth_rad=azel[0], elevation_rad=azel[1]
                        ),
                        BASE_LIGHT_AZIMUTH_ELEVATION,
                        PERTURBED_LIGHT_AZIMUTH_ELEVATION,
                    ),
                    (
                        "cam.jitter",
                        lambda xy: aim_camera(current, jitter_xy=xy),
                        BASE_CAMERA_JITTER,
                        PERTURBED_CAMERA_JITTER,
                    ),
                ]
                saved: dict[str, Any] = {}
                for role, writer, base_value, perturbed_value in knobs:
                    capture = make_attribute_capture(current, writer, depth=args.render_depth)
                    a1 = capture(base_value).clone().cpu()
                    b1 = capture(perturbed_value).clone().cpu()
                    b2 = capture(perturbed_value).clone().cpu()
                    a2 = capture(base_value).clone().cpu()
                    frames = {"a1": a1, "b1": b1, "b2": b2, "a2": a2}
                    slug = role.replace(".", "_")
                    for name, frame in frames.items():
                        torch.save(
                            {
                                "rgb": frame,
                                "role": role,
                                "base_value": base_value,
                                "perturbed_value": perturbed_value,
                            },
                            frames_dir / f"{slug}_{name}.pt",
                        )
                    saved[role] = {
                        "order_independent_a1_vs_a2": diff_summary(a1, a2),
                        "back_to_back_b1_vs_b2": diff_summary(b1, b2),
                        # Absolute stats, not just diffs (external research
                        # experiment E1) -- a diff of exactly 0.0 between two
                        # all-black or constant-colour frames looks identical
                        # to a diff of 0.0 between two correctly-lit but
                        # identical frames; this tells them apart.
                        "frame_stats_a1": frame_stats(a1),
                        "frame_stats_b1": frame_stats(b1),
                    }
                return {"frames_dir": str(frames_dir), "saved": saved}

            report.run("save_sample_frames", check_save_frames)

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
