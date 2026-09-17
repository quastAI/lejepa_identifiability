"""One standalone script that meets the whole Isaac API surface in a single boot.

Every Isaac signature in README §4.4 came from reading documentation, never from
running anything. This script is where that stops being true: it answers §7.2's
four spikes against the real runtime, and its output is what §3.1/§3.2/§7.3 get
rewritten from.

Run it on the pod, not here::

    ./isaaclab.sh -p spikes/spike_api.py --out /idtb/data/spike

Three structural rules, all of them load-bearing (docs/PLAN.md Phase 3):

1. **Never fail fast.** ``SimulationApp`` is one-shot per process and boot is
   slow, so an assert-and-die script yields exactly one failure per boot. Every
   check is isolated and all of them run; the PASS/FAIL table and ``facts.json``
   are the deliverable. A FAIL that is *understood* is a completed check.
2. **num_envs=2, widely spaced.** [IsaacSim #251](https://github.com/isaac-sim/IsaacSim/issues/251)
   is "frozen *at the env origin*". With one env the origin is ``(0,0,0)`` and a
   plausible cube target is centimetres away, so "frozen" and "correct" both sit
   inside any sloppy tolerance. Two widely-spaced envs make it a metres-scale,
   unmissable error.
3. **Determinism and convergence are reusable functions of a ``capture``
   callable**, not inline asserts -- scene v1 re-runs them unchanged, which is
   the whole reason §7.4's closing note does not cost a rewrite.

**Sensitivity ranks above determinism**, deliberately. Read-back, non-degenerate,
same-state-equal and A/B/B/A are *all* satisfied by a pipeline that renders the
same stale frame every time -- a Fabric flush that never reaches the render
graph, or a cached annotator buffer. That is the most deterministic pipeline
imaginable and it is 100% garbage, discovered weeks later when R² comes out
zero. So the first question asked of the renderer is whether it responds to the
state at all.

Everything above the "Isaac layer" banner is pure and imports no Isaac, so
``tests/test_spike_api.py`` exercises it on the dev machine -- including the
negative controls that prove these detectors can actually fire.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

Tensor = torch.Tensor
Capture = Callable[[Any], Tensor]

# The image's HUB__ARGS__DETECT_ONLY=true forbids Hub from starting, while
# omni.client's own default ("shared") asks it to anyway -- the two contradict
# each other and the result is ~35 retry warnings over ~10s at every boot
# (IsaacLab #6971, merged; still open for this exact image as #7732). Hub is a
# local USD caching layer, not required for https:// resolution at all (Isaac
# Lab disables it the same way for kitless runs, #6985), so silencing it costs
# nothing. `setdefault`: an operator's own env.sh export always wins.
os.environ.setdefault("OMNICLIENT_HUB_MODE", "disabled")

# ---------------------------------------------------------------------------
# Pure layer: no Isaac, no globals. Unit-tested locally.
# ---------------------------------------------------------------------------


class CheckFailed(Exception):
    """A check ran and the answer was no. Not an error -- a result."""


class CheckSkipped(Exception):
    """A check could not run, usually because something it depends on failed."""


def _f32(t: Tensor) -> Tensor:
    """Widen before subtracting.

    Frames come back ``uint8``; ``a - b`` on ``uint8`` wraps, so a maximal
    difference of 255 would read as 1. Every comparison below goes through here.
    """
    return t.detach().to(dtype=torch.float32)


def mean_abs_diff(a: Tensor, b: Tensor) -> float:
    return float((_f32(a) - _f32(b)).abs().mean().item())


def max_abs_diff(a: Tensor, b: Tensor) -> float:
    return float((_f32(a) - _f32(b)).abs().max().item())


def bitwise_equal(a: Tensor, b: Tensor) -> bool:
    """§7.1's acceptance is bitwise: a temporal denoiser leak sits at ~0.3/255.

    That passes any sane threshold while being exactly the structure a
    contrastive encoder is trained to find, so a threshold does not weaken this
    gate, it disables it. Reported alongside the magnitudes, never instead.
    """
    a, b = a.detach(), b.detach()
    return a.shape == b.shape and a.dtype == b.dtype and bool(torch.equal(a.cpu(), b.cpu()))


def frame_stats(frame: Tensor) -> dict[str, float]:
    """Enough to tell a real render from an all-black or constant one."""
    flat = _f32(frame).flatten()
    return {
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item()),
        "unique": float(torch.unique(flat).numel()),
    }


def frame_hash(frame: Tensor) -> str:
    """Stable content hash, for the cross-process check."""
    raw = frame.detach().cpu().contiguous()
    return hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest()


def determinism_report(
    capture: Capture,
    state_a: Any,
    state_b: Any,
    *,
    tol: float = 0.0,
) -> dict[str, Any]:
    """Spike 1: same-state equality, A/B/B/A order-independence, buffer aliasing.

    Takes a ``capture`` callable rather than a scene so scene v1 re-runs it
    unchanged. ``capture`` must return the renderer's own tensor, **not** a copy
    -- the aliasing probe depends on holding a live reference.

    The A/B/B/A order is §7.1's: ``mad(A1, A2)`` is order-independence (A
    rendered first and last) and ``mad(B1, B2)`` is the back-to-back render where
    temporal accumulation leaks if it leaks anywhere.

    ``content_reproducible`` and ``buffers_aliased`` are orthogonal and reported
    separately for that reason: the former is a property of the render mode
    (varies by preset), the latter is a property of the sensor API (measured
    True on every mode tried, an unavoidable "always clone" caller obligation,
    not a rendering-quality question). ``deterministic`` is the strict AND of
    both, for a caller that does not clone -- a real pipeline must, so
    per-preset comparisons should read ``content_reproducible``.
    """
    a1_live = capture(state_a)
    a1 = a1_live.clone()
    b1 = capture(state_b).clone()

    # If capture hands back a view onto a renderer-owned buffer, the B render
    # just overwrote a1_live -- and "same state renders equal" would be measuring
    # memory reuse rather than determinism. A1 and A2 would compare equal
    # *because they are the same memory*.
    aliased = not bitwise_equal(a1_live, a1)

    b2 = capture(state_b).clone()
    a2 = capture(state_a).clone()

    same_a, same_b = mean_abs_diff(a1, a2), mean_abs_diff(b1, b2)
    distinct = mean_abs_diff(a1, b1)
    # a1/a2/b1/b2 are all independent clones, so this is trustworthy regardless
    # of buffer reuse -- it answers "would a caller that clones correctly see
    # the same content", separately from "does a caller that doesn't clone get
    # fooled" (buffers_aliased). Measured on the pod: Isaac's camera output is
    # aliased on *every* render mode tested, which made `deterministic` below
    # permanently False for every preset -- true and important on its own, but
    # it was also silently hiding which presets actually render reproducibly.
    content_reproducible = same_a <= tol and same_b <= tol and distinct > max(tol, 0.0)
    return {
        "order_independent_mad": same_a,
        "order_independent_max": max_abs_diff(a1, a2),
        "order_independent_bitwise": bitwise_equal(a1, a2),
        "back_to_back_mad": same_b,
        "back_to_back_max": max_abs_diff(b1, b2),
        "back_to_back_bitwise": bitwise_equal(b1, b2),
        "states_distinguishable_mad": distinct,
        # Without this the whole report is vacuous: two identical frames are
        # perfectly "deterministic" no matter what the renderer is doing.
        "states_distinguishable": distinct > max(tol, 0.0),
        "buffers_aliased": aliased,
        "content_reproducible": content_reproducible,
        # Strict AND: true determinism for a caller that does *not* clone.
        "deterministic": content_reproducible and not aliased,
        "tol": tol,
    }


def convergence_report(
    capture_at_depth: Callable[[int], Tensor],
    depths: Sequence[int],
    *,
    tol: float,
) -> dict[str, Any]:
    """Spike 3: how many ``sim.render()`` calls until the image stops changing.

    This is the per-sample cost multiplier and therefore the entire GPU-hour
    budget, so it is recorded as *a procedure plus the scene it was measured on*
    -- never as a bare constant N, which would not survive scene v1 (§7.4).
    """
    depths = sorted(set(int(d) for d in depths))
    if len(depths) < 2:
        raise ValueError("need at least two depths to see a change")

    frames = {d: capture_at_depth(d).clone() for d in depths}
    # A SECOND, independent capture at the deepest depth -- not the dict entry
    # reused. Comparing the deepest depth's curve row to itself is tautological
    # (mad=0 always, proving nothing), and it is exactly what made a flat,
    # non-improving noise floor from depth 1 to 32 look like "converged at 64"
    # on the real pod run. This way even the deepest row can show it hasn't
    # actually settled.
    reference = capture_at_depth(depths[-1]).clone()

    curve = []
    previous = None
    for depth in depths:
        entry = {
            "depth": depth,
            "mad_vs_deepest": mean_abs_diff(frames[depth], reference),
            "max_vs_deepest": max_abs_diff(frames[depth], reference),
            "bitwise_vs_deepest": bitwise_equal(frames[depth], reference),
        }
        if previous is not None:
            entry["mad_vs_previous"] = mean_abs_diff(frames[depth], frames[previous])
        curve.append(entry)
        previous = depth

    converged = [e["depth"] for e in curve if e["mad_vs_deepest"] <= tol]
    return {
        "curve": curve,
        "smallest_converged_depth": converged[0] if converged else None,
        "deepest_depth": depths[-1],
        "tol": tol,
        # The deepest sample is its own reference, so "everything converged" can
        # mean the image never changed at all -- a stale pipeline, not a cheap one.
        "changed_across_depths": curve[0]["mad_vs_deepest"] > tol,
    }


def sensitivity_report(
    capture: Capture,
    base_state: Any,
    perturbations: Mapping[str, Any],
    *,
    noise_floor: float,
    min_ratio: float = 10.0,
) -> dict[str, Any]:
    """Does the render respond to the state at all -- and by how much vs. noise.

    Ranked above determinism (see the module docstring). Each perturbation moves
    exactly one group of handles, so a factor that fails here is one the pipeline
    is blind to, which would show up much later as an unexplainable per-dimension
    R² of zero.
    """
    base = capture(base_state).clone()
    floor = max(noise_floor, 0.0)
    responses: dict[str, Any] = {}
    for name, state in perturbations.items():
        frame = capture(state).clone()
        mad = mean_abs_diff(base, frame)
        responses[name] = {
            "mad_vs_base": mad,
            "max_vs_base": max_abs_diff(base, frame),
            # Ratio against the same-state noise floor: an absolute delta means
            # nothing without knowing what the renderer does when nothing moves.
            "ratio_to_noise_floor": (mad / floor) if floor > 0 else float("inf"),
            "responsive": mad > floor * min_ratio if floor > 0 else mad > 0.0,
        }
    return {
        "noise_floor_mad": floor,
        "min_ratio": min_ratio,
        "responses": responses,
        "all_responsive": all(r["responsive"] for r in responses.values()),
    }


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | SKIP
    seconds: float = 0.0
    note: str = ""
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    """Runs checks in order, swallowing every exception so the rest still run."""

    results: list[CheckResult] = field(default_factory=list)
    verbose: bool = True

    def run(self, name: str, fn: Callable[[], dict[str, Any] | None]) -> CheckResult:
        started = time.perf_counter()
        try:
            facts = fn() or {}
            result = CheckResult(name, "PASS", 0.0, "", facts)
        except CheckSkipped as exc:
            result = CheckResult(name, "SKIP", 0.0, str(exc))
        except CheckFailed as exc:
            result = CheckResult(name, "FAIL", 0.0, str(exc))
        except Exception:
            # An unexpected exception is a FAIL with evidence, never a crash:
            # one boot, one full report.
            result = CheckResult(name, "FAIL", 0.0, traceback.format_exc(limit=6).strip())
        result.seconds = time.perf_counter() - started
        self.results.append(result)
        if self.verbose:
            head = result.note.splitlines()[0] if result.note else ""
            print(f"[{result.status:4}] {name}  ({result.seconds:.1f}s) {head}", flush=True)
        return result

    def facts_of(self, name: str) -> dict[str, Any]:
        for result in self.results:
            if result.name == name:
                return result.facts
        return {}

    def status_of(self, name: str) -> str | None:
        for result in self.results:
            if result.name == name:
                return result.status
        return None

    def counts(self) -> dict[str, int]:
        counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    def table(self) -> str:
        width = max((len(r.name) for r in self.results), default=4)
        lines = [f"{'STATUS':6}  {'CHECK'.ljust(width)}  {'SECONDS':>7}  NOTE"]
        lines.append("-" * (len(lines[0]) + 20))
        for result in self.results:
            head = result.note.splitlines()[0] if result.note else ""
            lines.append(
                f"{result.status:6}  {result.name.ljust(width)}  {result.seconds:7.1f}  {head}"
            )
        counts = self.counts()
        lines.append("")
        lines.append(f"{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [
                {
                    "name": r.name,
                    "status": r.status,
                    "seconds": round(r.seconds, 3),
                    "note": r.note,
                    "facts": r.facts,
                }
                for r in self.results
            ],
            "summary": self.counts(),
        }


def resolve(paths: Sequence[str]) -> tuple[Any, str]:
    """Import the first dotted path that answers, and report which one did.

    Every symbol here was read from documentation and 3.0 moved several of them
    (README §4.4). Trying the known spellings turns a wrong guess into a recorded
    fact instead of a crash that costs a container restart to diagnose.
    """
    errors = []
    for path in paths:
        module_path, _, attr = path.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attr), path
        except Exception as exc:  # ImportError, AttributeError, and Kit's own
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    raise CheckFailed("no candidate path resolved:\n  " + "\n  ".join(errors))


# The shipped FRANKA_PANDA_CFG's usd_path 404s on the real asset tree -- the
# object moved under a Legacy/ subfolder upstream and the config was never
# updated to match. Verified by direct HEAD request against the asset tree, not
# assumed -- no tracked issue found for this specific divergence. Franka is the
# only shipped robot config with this split (checked Unitree, ANYbotics, UR,
# Kuka/Allegro); nothing else needs this patch.
FRANKA_USD_BROKEN_SUFFIX = "Robots/FrankaEmika/panda_instanceable.usd"
FRANKA_USD_LEGACY_SUFFIX = "Robots/FrankaEmika/Legacy/panda_instanceable.usd"


def correct_franka_usd_path(current: str | None) -> str | None:
    """Return the fixed path, or ``None`` if ``current`` doesn't match the known bug.

    Pure string patch, no Isaac involved -- tested here rather than trusted only
    on the pod. A non-match is not an error: it means either the config has
    already been fixed upstream, or points somewhere this patch shouldn't touch,
    and the caller records that rather than silently forcing a rewrite.
    """
    if current and current.endswith(FRANKA_USD_BROKEN_SUFFIX):
        return current[: -len(FRANKA_USD_BROKEN_SUFFIX)] + FRANKA_USD_LEGACY_SUFFIX
    return None


# ---------------------------------------------------------------------------
# Isaac layer: every import lives inside a function, after SimulationApp exists.
# ---------------------------------------------------------------------------

# Candidate carb settings per render mode (§7.3 presets are *defined* by what
# this run measures). Written blind: each key is read back after being set, so a
# path that does not exist in this build shows up in facts.json as a mismatch
# rather than as a silently ignored setting.
PRESET_CANDIDATES: dict[str, dict[str, Any]] = {
    "as_booted": {},
    "raytraced_lighting": {"/rtx/rendermode": "RaytracedLighting"},
    "realtime_pathtracing": {"/rtx/rendermode": "RealTimePathTracing"},
    "pathtracing_denoiser_off": {
        "/rtx/rendermode": "PathTracing",
        "/rtx/pathtracing/spp": 1,
        "/rtx/pathtracing/totalSpp": 64,
        "/rtx/pathtracing/optixDenoiser/enabled": 0,
    },
}

ARM_ROLE_JOINTS = ["panda_joint1", "panda_joint2", "panda_joint4", "panda_joint6"]
CUBE_NUDGE_M = 0.08  # spike-local, not a latent radius: this scene has no table


@dataclass
class Rig:
    """Everything the checks need from one built scene."""

    sim: Any
    scene: Any
    device: str
    arm_joint_ids: list[int]
    arm_joint_names: list[str]
    base_joint_pos: Tensor
    perturbed_joint_pos: Tensor
    cube_base_local: Tensor
    cube_moved_local: Tensor
    has_semantics: bool
    has_tiled: bool
    notes: dict[str, Any]


def build_rig(args: argparse.Namespace) -> Rig:
    """Build the smallest scene that can answer §7.2: Franka + cube + cameras."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, TiledCameraCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils import configclass

    notes: dict[str, Any] = {}
    franka_cfg, franka_path = resolve(
        [
            "isaaclab_assets.robots.franka.FRANKA_PANDA_CFG",
            "isaaclab_assets.FRANKA_PANDA_CFG",
            "isaaclab_assets.robots.FRANKA_PANDA_CFG",
            "omni.isaac.lab_assets.FRANKA_PANDA_CFG",
        ]
    )
    notes["franka_cfg_path"] = franka_path

    try:
        current_usd = franka_cfg.spawn.usd_path
        legacy_usd = correct_franka_usd_path(current_usd)
        if legacy_usd is not None:
            franka_cfg = franka_cfg.replace(spawn=franka_cfg.spawn.replace(usd_path=legacy_usd))
            notes["franka_usd_path"] = f"corrected (Legacy/): {current_usd!r} -> {legacy_usd!r}"
        else:
            notes["franka_usd_path"] = f"left as-is, unexpected suffix: {current_usd!r}"
    except Exception as exc:
        notes["franka_usd_path"] = f"could not correct: {type(exc).__name__}: {exc}"

    height, width = args.resolution

    def make_scene_cfg(*, semantics: bool, tiled: bool, rich_camera: bool, suffix: str):
        # Every prim path carries `suffix`, unique per ladder rung. A failed
        # InteractiveScene() call can leave prims it already created sitting on
        # the stage -- USD construction has no transactional rollback -- and the
        # next rung would otherwise die on "prim already exists" instead of its
        # own error. Distinct paths sidestep that without touching the stage or
        # SimulationContext between attempts, which is not a reset this script
        # is confident is safe mid-run.
        cube_spawn_kwargs: dict[str, Any] = {
            "size": (0.06, 0.06, 0.06),
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=0.1),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        }
        if semantics:
            cube_spawn_kwargs["semantic_tags"] = [("class", "cube")]

        data_types = ["rgb"]
        camera_kwargs: dict[str, Any] = {}
        if rich_camera:
            data_types = ["rgb", "distance_to_image_plane"]
            if semantics:
                data_types.append("semantic_segmentation")
                # Ids, not colours: "more than one id present" is the check, and
                # a colourised RGBA pass cannot answer it.
                camera_kwargs["colorize_semantic_segmentation"] = False

        @configclass
        class SpikeSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path=f"/World/ground_{suffix}", spawn=sim_utils.GroundPlaneCfg()
            )
            dome = AssetBaseCfg(
                prim_path=f"/World/DomeLight_{suffix}",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
            )
            robot = franka_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot_" + suffix)
            cube = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube_" + suffix,
                spawn=sim_utils.CuboidCfg(**cube_spawn_kwargs),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.03)),
            )
            camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Camera_" + suffix,
                update_period=0.0,
                height=height,
                width=width,
                data_types=data_types,
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 40.0)),
                # High and oblique per §5.3/§7.4: makes arm-over-cube occlusion
                # far rarer than an eye-level view, for free.
                offset=CameraCfg.OffsetCfg(pos=(1.4, 1.4, 1.2), convention="world"),
                **camera_kwargs,
            )

        if not tiled:
            return SpikeSceneCfg(num_envs=args.num_envs, env_spacing=args.env_spacing)

        @configclass
        class SpikeSceneWithTiledCfg(SpikeSceneCfg):
            tiled_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/TiledCamera_" + suffix,
                update_period=0.0,
                height=height,
                width=width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 40.0)),
                offset=TiledCameraCfg.OffsetCfg(pos=(1.4, 1.4, 1.2), convention="world"),
            )

        return SpikeSceneWithTiledCfg(num_envs=args.num_envs, env_spacing=args.env_spacing)

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

    # Build down a ladder of ambition: a single unknown kwarg must not cost the
    # whole run, and which rung answered is itself a §4.4 correction.
    scene = None
    attempts = []
    rungs = ((True, True, True), (True, False, True), (False, False, False))
    for rung_index, (semantics, tiled, rich) in enumerate(rungs):
        try:
            scene = InteractiveScene(
                make_scene_cfg(
                    semantics=semantics, tiled=tiled, rich_camera=rich, suffix=f"r{rung_index}"
                )
            )
            notes["scene_variant"] = {"semantics": semantics, "tiled": tiled, "rich_camera": rich}
            break
        except Exception as exc:
            attempts.append(
                f"semantics={semantics} tiled={tiled} rich={rich}: {type(exc).__name__}: {exc}"
            )
    if scene is None:
        raise CheckFailed("scene did not build at any level:\n  " + "\n  ".join(attempts))
    notes["scene_build_attempts"] = attempts

    sim.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    try:
        arm_ids, arm_names = robot.find_joints(ARM_ROLE_JOINTS)
    except Exception as exc:
        # §5.2 keys handles by role, never by Franka joint name, precisely so a
        # naming change costs a recorded note instead of the spike.
        arm_ids = list(range(min(4, robot.num_joints)))
        arm_names = [robot.joint_names[i] for i in arm_ids]
        notes["find_joints"] = f"{type(exc).__name__}: {exc} -- fell back to {arm_names}"

    base_joint_pos = robot.data.default_joint_pos.clone()
    limits = None
    for attr in ("soft_joint_pos_limits", "joint_pos_limits", "joint_limits"):
        limits = getattr(robot.data, attr, None)
        if limits is not None:
            notes["joint_limits_attr"] = attr
            break

    # A fraction of the *measured* half-range, never an absolute number baked
    # into code (§5.2). Falls back only if no limit tensor is exposed at all.
    perturbed_joint_pos = base_joint_pos.clone()
    ids = torch.as_tensor(list(arm_ids), device=base_joint_pos.device, dtype=torch.long)
    if limits is not None:
        lo, hi = limits[..., 0][:, ids], limits[..., 1][:, ids]
        half_range = 0.5 * (hi - lo)
        target = base_joint_pos[:, ids] + 0.25 * half_range
        perturbed_joint_pos[:, ids] = torch.minimum(target, hi - 1e-3)
        notes["arm_perturbation"] = "+0.25 x measured half-range, clamped inside limits"
    else:
        perturbed_joint_pos[:, ids] = base_joint_pos[:, ids] + 0.3
        notes["arm_perturbation"] = "+0.3 rad (no limit tensor found)"

    cube_base_local = torch.tensor(
        [0.45, 0.0, 0.03], device=sim.device, dtype=torch.float32
    ).repeat(args.num_envs, 1)
    cube_moved_local = cube_base_local.clone()
    cube_moved_local[:, 0] += CUBE_NUDGE_M
    cube_moved_local[:, 1] += CUBE_NUDGE_M

    rig = Rig(
        sim=sim,
        scene=scene,
        device=str(sim.device),
        arm_joint_ids=[int(i) for i in arm_ids],
        arm_joint_names=list(arm_names),
        base_joint_pos=base_joint_pos,
        perturbed_joint_pos=perturbed_joint_pos,
        cube_base_local=cube_base_local,
        cube_moved_local=cube_moved_local,
        has_semantics=bool(notes["scene_variant"]["semantics"]),
        has_tiled=bool(notes["scene_variant"]["tiled"]),
        notes=notes,
    )
    _aim_cameras(rig)
    return rig


def _aim_cameras(rig: Rig) -> None:
    """Point the cameras at the workspace.

    ``set_world_poses_from_view`` rather than a hand-built quaternion: getting a
    look-at rotation wrong is silent -- it renders an empty corner of the room
    and every image check below still "passes".
    """
    origins = rig.scene.env_origins
    eyes = origins + torch.tensor([1.4, 1.4, 1.2], device=origins.device)
    targets = origins + torch.tensor([0.35, 0.0, 0.1], device=origins.device)
    for key in ("camera", "tiled_camera"):
        try:
            sensor = rig.scene[key]
        except Exception:
            continue
        try:
            sensor.set_world_poses_from_view(eyes, targets)
            rig.notes[f"{key}_aim"] = "set_world_poses_from_view"
        except Exception as exc:
            rig.notes[f"{key}_aim"] = f"offset only: {type(exc).__name__}: {exc}"


def state_of(*, joint_pos: Tensor, cube_local: Tensor) -> dict[str, Tensor]:
    """A spike state is just the two handle groups §5.2 cares about at stage 1."""
    return {"joint_pos": joint_pos, "cube_local": cube_local}


def write_state(rig: Rig, state: Mapping[str, Tensor]) -> None:
    """The §4.5 write half: teleport, no rollout, velocities zeroed."""
    robot = rig.scene["robot"]
    cube = rig.scene["cube"]
    joint_pos = state["joint_pos"]
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    root = cube.data.default_root_state.clone()
    root[:, :3] = rig.scene.env_origins + state["cube_local"]
    root[:, 7:] = 0.0  # linear and angular velocity -- §4.4
    cube.write_root_state_to_sim(root)


def read_state(rig: Rig) -> dict[str, Tensor]:
    robot = rig.scene["robot"]
    cube = rig.scene["cube"]
    return {
        "joint_pos": robot.data.joint_pos.clone(),
        "cube_pos_w": cube.data.root_pos_w.clone(),
        "cube_local": cube.data.root_pos_w - rig.scene.env_origins,
    }


def make_capture(
    rig: Rig, *, depth: int, sensor: str = "camera", data_type: str = "rgb"
) -> Capture:
    """The §4.5 capture sequence, as a closure over one (scene, preset).

    Returns the sensor's own tensor rather than a copy, which is what lets
    :func:`determinism_report` detect an aliased buffer. Callers that keep a
    frame must clone it.
    """

    def capture(state: Mapping[str, Tensor]) -> Tensor:
        write_state(rig, state)
        rig.scene.write_data_to_sim()
        rig.sim.forward()  # flush USD/Fabric, no time advance
        camera = rig.scene[sensor]
        camera.update(dt=0.0, force_recompute=True)
        for _ in range(depth):
            rig.sim.render()
        camera.update(dt=0.0, force_recompute=True)
        return camera.data.output[data_type]

    return capture


def apply_preset(name: str) -> dict[str, Any]:
    """Set a preset's carb values and read every one of them back.

    Switching mode at runtime is not provably identical to booting in it -- carb
    settings are the only lever available inside one ``SimulationApp``, and that
    caveat is recorded rather than assumed away.
    """
    import carb

    settings = carb.settings.get_settings()
    applied = {}
    for key, value in PRESET_CANDIDATES[name].items():
        try:
            settings.set(key, value)
            read_back = settings.get(key)
        except Exception as exc:
            applied[key] = {"wanted": value, "error": f"{type(exc).__name__}: {exc}"}
            continue
        applied[key] = {
            "wanted": value,
            "read_back": read_back,
            # carb happily creates unknown keys, so equality here is the only
            # evidence that the setting is one this build actually has.
            "accepted": read_back == value,
        }
    return applied


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("/idtb/data/spike"))
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument(
        "--env-spacing",
        type=float,
        default=8.0,
        help="metres; wide on purpose so an IsaacSim#251 freeze is unmissable",
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=(128, 128), metavar=("H", "W"))
    parser.add_argument(
        "--render-depth", type=int, default=8, help="sim.render() calls per capture"
    )
    parser.add_argument("--max-depth", type=int, default=64, help="deepest depth for Spike 3")
    parser.add_argument("--tol", type=float, default=0.0, help="0.0 = bitwise, per §7.1")
    parser.add_argument(
        "--readback-tol",
        type=float,
        default=1e-4,
        help="rad / m; solver tolerance, not a render tolerance (§6.3)",
    )
    parser.add_argument("--presets", nargs="*", default=list(PRESET_CANDIDATES))
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="save a few captured frames + their generating state to <out>/frames "
        "(docs/PLAN.md Phase 3 leftover -- so the mock is built to real conventions, "
        "not assumed ones)",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Sensor rendering in a standalone script needs this (§4.4); harmless if the
    # flag has already been folded into the 3.0 defaults.
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    report = Report()
    out_dir: Path = args.out
    rig: Rig | None = None
    started = time.time()
    # Early, not in the finally: the cross-process check writes its fingerprint
    # here mid-run, and a missing directory would turn that into a FAIL that
    # says nothing about determinism.
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # reported, never fatal
        print(f"[warn] cannot create {out_dir}: {type(exc).__name__}: {exc}", flush=True)

    def need_rig() -> Rig:
        if rig is None:
            raise CheckSkipped("scene did not build")
        return rig

    try:
        # -- boot ---------------------------------------------------------
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

        # -- scene --------------------------------------------------------
        def check_scene() -> dict[str, Any]:
            nonlocal rig
            rig = build_rig(args)
            robot = rig.scene["robot"]
            cube = rig.scene["cube"]
            camera = rig.scene["camera"]
            facts: dict[str, Any] = {
                "num_envs": int(rig.scene.num_envs),
                "env_spacing_m": args.env_spacing,
                "env_origins": rig.scene.env_origins.tolist(),
                "joint_names": list(robot.joint_names),
                "arm_role_joints": dict(zip(ARM_ROLE_JOINTS, rig.arm_joint_names, strict=False)),
                "arm_joint_ids": rig.arm_joint_ids,
                "default_joint_pos": robot.data.default_joint_pos[0].tolist(),
                "cube_default_root_state": cube.data.default_root_state[0].tolist(),
                **rig.notes,
            }
            limits_attr = rig.notes.get("joint_limits_attr")
            if limits_attr:
                # Promotes §5.2 from provisional: the radii become fractions of
                # these, rather than of a guess.
                facts["joint_limits"] = getattr(robot.data, limits_attr)[0].tolist()
            try:
                facts["intrinsic_matrices"] = camera.data.intrinsic_matrices[0].tolist()
            except Exception as exc:
                facts["intrinsic_matrices"] = f"unavailable: {type(exc).__name__}: {exc}"
            return facts

        report.run("scene_builds_and_measures", check_scene)

        # -- write / read-back, zero physics steps ------------------------
        def check_read_back() -> dict[str, Any]:
            current = need_rig()
            state = state_of(joint_pos=current.base_joint_pos, cube_local=current.cube_base_local)
            write_state(current, state)
            current.scene.write_data_to_sim()
            current.sim.forward()
            current.scene.update(current.sim.get_physics_dt())
            observed = read_state(current)

            joint_err = float((observed["joint_pos"] - current.base_joint_pos).abs().max().item())
            cube_err = float((observed["cube_local"] - current.cube_base_local).abs().max().item())
            # IsaacSim #251 is "frozen *at the env origin*". With env_spacing
            # metres wide, a per-env world position still sitting on its origin
            # is a metres-scale error no tolerance can absorb.
            dist_from_origin = (
                (observed["cube_pos_w"] - current.scene.env_origins).norm(dim=-1).tolist()
            )
            facts = {
                "sim_steps_taken": 0,
                "max_joint_error_rad": joint_err,
                "max_cube_error_m": cube_err,
                "cube_distance_from_env_origin_m": dist_from_origin,
                "cube_pos_w": observed["cube_pos_w"].tolist(),
                "tolerance": args.readback_tol,
            }
            if joint_err > args.readback_tol or cube_err > args.readback_tol:
                raise CheckFailed(
                    f"read-back off by {joint_err:.3g} rad / {cube_err:.3g} m "
                    f"(tol {args.readback_tol:g}) with zero sim.step() -- Spike 2 says "
                    "the capture sequence needs more than forward()"
                )
            if min(dist_from_origin) < 0.5 * CUBE_NUDGE_M:
                raise CheckFailed("cube sits on the env origin -- IsaacSim #251 pattern")
            return facts

        report.run("write_read_back_zero_step", check_read_back)

        # -- does one physics step move it (informs §5.5) ------------------
        def check_step_drift() -> dict[str, Any]:
            current = need_rig()
            before = read_state(current)
            current.sim.step(render=False)
            current.scene.update(current.sim.get_physics_dt())
            after = read_state(current)
            return {
                "joint_drift_rad": float(
                    (after["joint_pos"] - before["joint_pos"]).abs().max().item()
                ),
                "cube_drift_m": float(
                    (after["cube_pos_w"] - before["cube_pos_w"]).abs().max().item()
                ),
                "note": "informational: §5.5's policy depends on how far one step moves things",
            }

        report.run("one_step_drift", check_step_drift)

        # -- render is not degenerate -------------------------------------
        def check_render() -> dict[str, Any]:
            current = need_rig()
            capture = make_capture(current, depth=args.render_depth)
            frame = capture(
                state_of(joint_pos=current.base_joint_pos, cube_local=current.cube_base_local)
            ).clone()
            stats = frame_stats(frame)
            facts = {
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "render_depth": args.render_depth,
                **stats,
            }
            if stats["std"] == 0.0:
                raise CheckFailed(f"frame is constant at {stats['mean']:.3g} -- nothing rendered")
            return facts

        report.run("render_non_degenerate", check_render)

        # -- sensitivity, ranked above determinism ------------------------
        def check_sensitivity() -> dict[str, Any]:
            current = need_rig()
            # Under the one preset already measured reproducible (render_mode_sweep
            # below), not whatever the sim booted into. README §7.1: naming a
            # realtime mode does not buy determinism, it has to be constructed --
            # testing sensitivity against the noisy boot default was measuring the
            # wrong thing. §7.3's actual preset choice stays open; this is this
            # spike's own best evidence so far, applied here rather than assumed.
            preset_settings = apply_preset("pathtracing_denoiser_off")
            capture = make_capture(current, depth=args.render_depth)
            base = state_of(joint_pos=current.base_joint_pos, cube_local=current.cube_base_local)
            # Warm-up, discarded: this is the first capture since the preset switch
            # above, and a `--num-envs` sweep measured the noise floor blow up to
            # ~75 mad (vs. 0.0 normally) on 2 of 4 repeats at the exact same
            # settings -- flaky, not deterministic (a repeat of the same failing
            # run passed cleanly), consistent with a full render-mode switch
            # needing a discarded settle render the same way an individual
            # attribute write did in spikes/spike_dynamic_attrs.py.
            capture(base)
            # The floor is what the renderer does when *nothing* changes; an
            # absolute delta is meaningless without it.
            first = capture(base).clone()
            second = capture(base).clone()
            floor = mean_abs_diff(first, second)

            facts = sensitivity_report(
                capture,
                base,
                {
                    "arm_only": state_of(
                        joint_pos=current.perturbed_joint_pos,
                        cube_local=current.cube_base_local,
                    ),
                    "cube_only": state_of(
                        joint_pos=current.base_joint_pos,
                        cube_local=current.cube_moved_local,
                    ),
                },
                noise_floor=floor,
            )
            facts["preset_applied"] = {
                "name": "pathtracing_denoiser_off",
                "settings": preset_settings,
            }
            if not facts["all_responsive"]:
                blind = {k: v for k, v in facts["responses"].items() if not v["responsive"]}
                raise CheckFailed(
                    f"render does not respond to {list(blind)} above the noise floor "
                    f"(noise_floor_mad={floor:.4g}, min_ratio={facts['min_ratio']:g}); "
                    f"details {blind} -- a stale frame satisfies every other check in this table"
                )
            return facts

        report.run("sensitivity_arm_and_cube", check_sensitivity)

        # -- determinism + aliasing (Spike 1) -----------------------------
        def check_determinism() -> dict[str, Any]:
            current = need_rig()
            # Same preset as check_sensitivity, same reasoning: this is testing
            # whether determinism is achievable at all, which the boot default
            # already answered "no" to (README §7.1) -- not whether this specific
            # setting choice needs revisiting later against scene v1.
            preset_settings = apply_preset("pathtracing_denoiser_off")
            capture = make_capture(current, depth=args.render_depth)
            facts = determinism_report(
                capture,
                state_of(joint_pos=current.base_joint_pos, cube_local=current.cube_base_local),
                state_of(
                    joint_pos=current.perturbed_joint_pos,
                    cube_local=current.cube_moved_local,
                ),
                tol=args.tol,
            )
            facts["preset_applied"] = {
                "name": "pathtracing_denoiser_off",
                "settings": preset_settings,
            }
            # Reported as two separate problems, not one blob: content noise is a
            # preset/settings question (fixable by choosing differently), buffer
            # aliasing is a structural sensor-API fact confirmed on every mode
            # tried (fixable only by always cloning in caller code, permanently).
            problems = []
            if not facts["content_reproducible"]:
                problems.append(
                    f"content not reproducible: order-independence "
                    f"{facts['order_independent_mad']:.4g}, back-to-back "
                    f"{facts['back_to_back_mad']:.4g} (tol {args.tol:g})"
                )
            if facts["buffers_aliased"]:
                problems.append(
                    "buffers aliased: sensor returns a view onto a reused buffer -- "
                    "every caller must .clone() immediately, always, on this Isaac version"
                )
            if problems:
                raise CheckFailed("; ".join(problems))
            return facts

        report.run("determinism_and_aliasing", check_determinism)

        # -- convergence depth (Spike 3) ----------------------------------
        def check_convergence() -> dict[str, Any]:
            current = need_rig()
            state = state_of(joint_pos=current.base_joint_pos, cube_local=current.cube_base_local)
            depths = [d for d in (1, 2, 4, 8, 16, 32, args.max_depth) if d <= args.max_depth]

            def at_depth(depth: int) -> Tensor:
                return make_capture(current, depth=depth)(state)

            facts = convergence_report(at_depth, depths, tol=args.tol)
            facts["procedure"] = (
                "mean |delta| vs the deepest render, Franka + cuboid + dome light, "
                f"{args.resolution[0]}x{args.resolution[1]}; N is this procedure, "
                "not the number -- re-measure on scene v1 (§7.4)"
            )
            return facts

        report.run("convergence_depth", check_convergence)

        # -- per-preset sweep (Spike 1 verdict) ---------------------------
        def check_presets() -> dict[str, Any]:
            current = need_rig()
            base = state_of(joint_pos=current.base_joint_pos, cube_local=current.cube_base_local)
            other = state_of(
                joint_pos=current.perturbed_joint_pos,
                cube_local=current.cube_moved_local,
            )
            results: dict[str, Any] = {}
            for name in args.presets:
                if name not in PRESET_CANDIDATES:
                    results[name] = {"error": "unknown preset"}
                    continue
                entry: dict[str, Any] = {"settings": apply_preset(name)}
                try:
                    capture = make_capture(current, depth=args.render_depth)
                    entry["determinism"] = determinism_report(capture, base, other, tol=args.tol)
                    entry["frame"] = frame_stats(capture(base))
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                results[name] = entry
            # content_reproducible, not the stricter deterministic: buffers_aliased
            # measured True on every mode tried here (a sensor-API fact, not a
            # rendering-quality one -- see determinism_report's docstring), which
            # would otherwise make this list empty regardless of which preset is
            # actually good.
            deterministic = [
                name
                for name, entry in results.items()
                if entry.get("determinism", {}).get("content_reproducible")
            ]
            return {
                "presets": results,
                "deterministic_presets": deterministic,
                "caveat": "modes switched at runtime via carb, not booted into",
            }

        report.run("render_mode_sweep", check_presets)

        # -- TiledCamera vs Camera (Spike 4) ------------------------------
        def check_tiled() -> dict[str, Any]:
            current = need_rig()
            if not current.has_tiled:
                raise CheckSkipped("scene built without a TiledCamera")
            # Distinct state per env, or two identical tiles prove nothing.
            joint_pos = current.base_joint_pos.clone()
            cube_local = current.cube_base_local.clone()
            if current.scene.num_envs > 1:
                joint_pos[1] = current.perturbed_joint_pos[1]
                cube_local[1] = current.cube_moved_local[1]
            state = state_of(joint_pos=joint_pos, cube_local=cube_local)

            facts: dict[str, Any] = {}
            for sensor in ("camera", "tiled_camera"):
                capture = make_capture(current, depth=args.render_depth, sensor=sensor)
                start = time.perf_counter()
                frame = capture(state).clone()
                elapsed = time.perf_counter() - start
                facts[sensor] = {
                    "shape": list(frame.shape),
                    "seconds_per_capture": elapsed,
                    "ms_per_sample": 1000.0 * elapsed / max(1, current.scene.num_envs),
                    **frame_stats(frame),
                }
                if current.scene.num_envs > 1:
                    # #367 is exactly this: correct total resolution, per-camera
                    # tiles gone. Two envs holding different states must differ.
                    facts[sensor]["tiles_distinct_mad"] = mean_abs_diff(frame[0], frame[1])
            tiled = facts.get("tiled_camera", {})
            if current.scene.num_envs > 1 and tiled.get("tiles_distinct_mad", 1.0) <= args.tol:
                raise CheckFailed(
                    "TiledCamera returns identical tiles for different states -- "
                    "the IsaacSim #367 pattern, which 6.0 is supposed to have fixed"
                )
            return facts

        report.run("tiled_vs_camera", check_tiled)

        # -- throughput ----------------------------------------------------
        def check_throughput() -> dict[str, Any]:
            current = need_rig()
            capture = make_capture(current, depth=args.render_depth)
            state = state_of(joint_pos=current.base_joint_pos, cube_local=current.cube_base_local)
            capture(state)  # warm up; the first capture pays for shader compiles
            repeats = 5
            start = time.perf_counter()
            for _ in range(repeats):
                capture(state)
            elapsed = time.perf_counter() - start
            per_capture = elapsed / repeats
            samples = max(1, current.scene.num_envs)
            return {
                "resolution": list(args.resolution),
                "num_envs": samples,
                "render_depth": args.render_depth,
                "ms_per_capture": 1000.0 * per_capture,
                "ms_per_sample": 1000.0 * per_capture / samples,
                "projected_hours_per_100k_pairs": (2 * 100_000 * per_capture / samples) / 3600.0,
                "note": "the B sweep is across runs: re-run with --num-envs {1,2,8,32}",
            }

        report.run("throughput", check_throughput)

        # -- segmentation annotator ---------------------------------------
        def check_segmentation() -> dict[str, Any]:
            current = need_rig()
            if not current.has_semantics:
                raise CheckSkipped("scene built without semantic tags")
            capture = make_capture(
                current, depth=args.render_depth, data_type="semantic_segmentation"
            )
            state = state_of(joint_pos=current.base_joint_pos, cube_local=current.cube_base_local)
            try:
                frame = capture(state).clone()
            except KeyError as exc:
                raise CheckFailed(f"annotator missing from camera output: {exc}") from exc
            ids = torch.unique(_f32(frame))
            facts = {
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "unique_ids": ids.numel(),
                "ids": ids.flatten()[:16].tolist(),
            }
            try:
                info = current.scene["camera"].data.info
                facts["id_to_labels"] = str(info[0].get("semantic_segmentation"))[:800]
            except Exception as exc:
                facts["id_to_labels"] = f"unavailable: {type(exc).__name__}: {exc}"
            if ids.numel() < 2:
                # Without this the occlusion-conditioned metric (§5.3) silently
                # becomes a constant column.
                raise CheckFailed("segmentation has a single id -- nothing is labelled")
            return facts

        report.run("segmentation_annotator", check_segmentation)

        # -- cross-process determinism ------------------------------------
        def check_cross_process() -> dict[str, Any]:
            current = need_rig()
            capture = make_capture(current, depth=args.render_depth)
            state = state_of(joint_pos=current.base_joint_pos, cube_local=current.cube_base_local)
            digest = frame_hash(capture(state))
            fingerprint = {
                "sha256": digest,
                "num_envs": int(current.scene.num_envs),
                "resolution": list(args.resolution),
                "render_depth": args.render_depth,
                "device": current.device,
            }
            path = out_dir / "canonical_frame.json"
            previous = json.loads(path.read_text()) if path.exists() else None
            comparable = (
                {k: previous.get(k) for k in fingerprint if k != "sha256"}
                if previous is not None
                else None
            )
            settings_match = comparable == {k: v for k, v in fingerprint.items() if k != "sha256"}
            if previous is None or not settings_match:
                # A settings change (e.g. a different --render-depth) has to reset
                # the baseline here, not skip forever: leaving the stale fingerprint
                # in place means every future run at the *new* settings compares
                # against the *old* ones and never stops SKIPping.
                path.write_text(json.dumps(fingerprint, indent=2))
                reason = (
                    "no prior run to compare against"
                    if previous is None
                    else (f"prior run used different settings ({previous}); baseline reset")
                )
                raise CheckSkipped(f"{reason} -- run this script again to complete the check")
            if previous["sha256"] != digest:
                raise CheckFailed(
                    "same state hashes differently in a second process -- "
                    f"{previous['sha256'][:16]} vs {digest[:16]}; spot/checkpoint "
                    "resume would not be reproducible"
                )
            return {"sha256": digest, "matched_previous_process": True}

        report.run("cross_process_determinism", check_cross_process)

        # -- optional: a few real frames + generating state, for the mock ----
        if args.save_frames:

            def check_save_frames() -> dict[str, Any]:
                current = need_rig()
                preset_settings = apply_preset("pathtracing_denoiser_off")
                capture = make_capture(current, depth=args.render_depth)
                states = {
                    "base": state_of(
                        joint_pos=current.base_joint_pos, cube_local=current.cube_base_local
                    ),
                    "arm_only": state_of(
                        joint_pos=current.perturbed_joint_pos, cube_local=current.cube_base_local
                    ),
                    "cube_only": state_of(
                        joint_pos=current.base_joint_pos, cube_local=current.cube_moved_local
                    ),
                    "arm_and_cube": state_of(
                        joint_pos=current.perturbed_joint_pos, cube_local=current.cube_moved_local
                    ),
                }
                frames_dir = out_dir / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                saved: dict[str, Any] = {}
                for name, state in states.items():
                    frame = capture(state).clone().cpu()
                    path = frames_dir / f"{name}.pt"
                    torch.save(
                        {
                            "rgb": frame,
                            "joint_pos": state["joint_pos"].cpu(),
                            "cube_local": state["cube_local"].cpu(),
                            "preset": "pathtracing_denoiser_off",
                            "render_depth": args.render_depth,
                        },
                        path,
                    )
                    saved[name] = {
                        "path": str(path),
                        "shape": list(frame.shape),
                        "dtype": str(frame.dtype),
                    }
                return {
                    "frames_dir": str(frames_dir),
                    "saved": saved,
                    "preset_applied": {
                        "name": "pathtracing_denoiser_off",
                        "settings": preset_settings,
                    },
                    "note": "load with torch.load(path) locally -- no Isaac needed, "
                    "torch is a plain dependency",
                }

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
