"""Builds the scene and implements `SceneBackend` against it (README §4.3).

Combining the Franka + cube + camera rig with every attribute write in one
scene, all at once, found two real defects on the pod beyond what Spikes
1/2/5 verified in isolation (docs/PLAN.md Phase 4's third/fourth pod runs:
a `FabricFrameView` prim-path gap for the table, and a `SimulationApp`
session-scope/teardown issue) plus one deeper one `writer.py`'s docstring
covers (the fifth pod run's render-staleness finding).

Only the roles Spikes 1–5 actually verified are wired up: `light.azimuth`,
`light.elevation` and `table.roughness` are confirmed **blocked** (README
§7.5, §11) and are refused at :meth:`IsaacSceneBackend.bind`, not faked.

Scope: `B=1` (single env) only — see `writer.py`'s module docstring for why.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from idtb.latents import LatentSpec
from idtb.sim.backend import HandleInfo, UnsupportedRoleError
from idtb.sim.writer import (
    ROLE_WRITE_PATHS,
    handle_infos,
    read_cube_position,
    read_latent_state,
    write_latent_state,
)

Tensor = torch.Tensor

#: `spikes/spike_api.py`'s verified fix: the shipped config's `usd_path` 404s;
#: the asset moved under a `Legacy/` subfolder upstream (README §7.2).
_FRANKA_USD_BROKEN_SUFFIX = "Robots/FrankaEmika/panda_instanceable.usd"
_FRANKA_USD_LEGACY_SUFFIX = "Robots/FrankaEmika/Legacy/panda_instanceable.usd"

_ARM_JOINT_NAMES = ["panda_joint1", "panda_joint2", "panda_joint4", "panda_joint6"]
_FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]

_GROUND_PATH = "/World/ground"
_LIGHT_PATH = "/World/Light"
_ROBOT_PATH = "{ENV_REGEX_NS}/Robot"
_CUBE_PATH = "{ENV_REGEX_NS}/Cube"
_TABLE_PATH = "{ENV_REGEX_NS}/Table"
_CAMERA_PATH = "{ENV_REGEX_NS}/Camera"

_DEFAULT_CUBE_EDGE_M = 0.06
_DEFAULT_CUBE_HUE = 0.05
_DEFAULT_LIGHT_INTENSITY = 900.0
_DEFAULT_LIGHT_WARMTH_K = 5500.0
_DEFAULT_TABLE_ALBEDO = 0.5
_DEFAULT_EXPOSURE = 1.0

#: High, oblique, near-top-down: makes arm-over-cube occlusion far rarer
#: than an eye-level view (README §5.3, §7.4), and is the exact placement
#: both spikes' cameras used.
_CAMERA_EYE = (1.0, 1.0, 0.9)
_CAMERA_TARGET = (0.45, 0.0, 0.1)
_CUBE_XY = (0.45, 0.0)
_TABLE_TRANSLATION = (_CUBE_XY[0] - 0.25, _CUBE_XY[1] + 0.25, 0.16)
_TABLE_SIZE_M = (0.3, 0.3, 0.01)


def _correct_franka_usd_path(current: str | None) -> str | None:
    if current and current.endswith(_FRANKA_USD_BROKEN_SUFFIX):
        return current[: -len(_FRANKA_USD_BROKEN_SUFFIX)] + _FRANKA_USD_LEGACY_SUFFIX
    return None


@dataclass
class Rig:
    """Everything :mod:`idtb.sim.writer` needs from one built scene."""

    sim: Any
    scene: Any
    camera: Any
    robot: Any
    cube_prim: Any
    cube_shader: Any
    table_prim: Any
    table_shader: Any
    light_prim: Any
    arm_joint_ids: list[int]
    finger_joint_ids: list[int]
    default_cube_edge_m: float
    default_cube_xy: tuple[float, float]
    ground_z: float
    camera_eye: tuple[float, float, float]
    camera_target: tuple[float, float, float]
    pending_jitter: tuple[float, float] = (0.0, 0.0)


def _resolve_bound_shader(prim: Any) -> Any:
    """Same technique as `spikes/spike_dynamic_attrs.py::resolve_bound_shader`
    (README §7.5): try the binding API directly, then a recursive subtree
    search, then the `Looks/` naming convention -- whichever answers first."""
    from pxr import Usd, UsdShade

    try:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            source, _, _ = material.ComputeSurfaceSource()
            if source:
                return source
    except Exception:
        pass
    for descendant in Usd.PrimRange(prim):
        try:
            material, _ = UsdShade.MaterialBindingAPI(descendant).ComputeBoundMaterial()
        except Exception:
            continue
        if not material:
            continue
        source, _, _ = material.ComputeSurfaceSource()
        if source:
            return source
    raise RuntimeError(f"no bound shader found under {prim.GetPath()}")


def build_rig(*, resolution: tuple[int, int] = (128, 128), device: str = "cuda:0") -> Rig:
    """Build the single-env scene `IsaacSceneBackend` operates on."""
    import isaaclab.sim as sim_utils
    import omni.usd
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import RenderCfg, SimulationCfg, SimulationContext
    from isaaclab.utils import configclass

    franka_cfg, _ = _resolve_franka_cfg()
    legacy_usd = _correct_franka_usd_path(franka_cfg.spawn.usd_path)
    if legacy_usd is not None:
        franka_cfg = franka_cfg.replace(spawn=franka_cfg.spawn.replace(usd_path=legacy_usd))

    height, width = resolution

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path=_GROUND_PATH, spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(
            prim_path=_LIGHT_PATH,
            spawn=sim_utils.DistantLightCfg(
                intensity=_DEFAULT_LIGHT_INTENSITY, color=(1.0, 1.0, 1.0)
            ),
        )
        robot = franka_cfg.replace(prim_path=_ROBOT_PATH)
        cube = RigidObjectCfg(
            prim_path=_CUBE_PATH,
            spawn=sim_utils.CuboidCfg(
                size=(_DEFAULT_CUBE_EDGE_M,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.13, 0.11)),
                semantic_tags=[("class", "cube")],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(*_CUBE_XY, 0.5 * _DEFAULT_CUBE_EDGE_M)
            ),
        )
        table = AssetBaseCfg(
            prim_path=_TABLE_PATH,
            spawn=sim_utils.CuboidCfg(
                size=_TABLE_SIZE_M,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(_DEFAULT_TABLE_ALBEDO,) * 3
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=_TABLE_TRANSLATION),
        )
        camera = CameraCfg(
            prim_path=_CAMERA_PATH,
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 40.0)),
            offset=CameraCfg.OffsetCfg(pos=_CAMERA_EYE, convention="world"),
        )

    sim = SimulationContext(
        SimulationCfg(
            dt=1.0 / 60.0,
            device=device,
            render=RenderCfg(antialiasing_mode="Off", enable_dl_denoiser=False),
        )
    )
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=8.0))
    sim.reset()
    scene.update(sim.get_physics_dt())

    robot = scene["robot"]
    arm_ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    finger_ids, _ = robot.find_joints(_FINGER_JOINT_NAMES)

    # Isaac Lab's standard, stable expansion for env 0 (used throughout the
    # spikes, e.g. spike_api.py's "{ENV_REGEX_NS}/Robot_" + suffix pattern).
    # Built from our own path templates directly, not from `scene["table"].cfg`
    # -- confirmed on the pod that `scene[...]` indexing returns a bare
    # `FabricFrameView` (no `.cfg` attribute) for a purely static `AssetBaseCfg`
    # entry like the table, unlike the `RigidObjectCfg`-backed cube. Using the
    # templates we already hold avoids depending on that asset-type distinction
    # at all.
    env0 = "/World/envs/env_0"
    stage = omni.usd.get_context().get_stage()
    cube_prim = stage.GetPrimAtPath(_CUBE_PATH.replace("{ENV_REGEX_NS}", env0))
    table_prim = stage.GetPrimAtPath(_TABLE_PATH.replace("{ENV_REGEX_NS}", env0))
    light_prim = stage.GetPrimAtPath(_LIGHT_PATH)

    cube_shader = _resolve_bound_shader(cube_prim)
    table_shader = _resolve_bound_shader(table_prim)

    from idtb.sim.render import apply_standard_preset

    apply_standard_preset()

    rig = Rig(
        sim=sim,
        scene=scene,
        camera=scene["camera"],
        robot=robot,
        cube_prim=cube_prim,
        cube_shader=cube_shader,
        table_prim=table_prim,
        table_shader=table_shader,
        light_prim=light_prim,
        arm_joint_ids=[int(i) for i in arm_ids],
        finger_joint_ids=[int(i) for i in finger_ids],
        default_cube_edge_m=_DEFAULT_CUBE_EDGE_M,
        default_cube_xy=_CUBE_XY,
        ground_z=0.0,
        camera_eye=_CAMERA_EYE,
        camera_target=_CAMERA_TARGET,
    )
    return rig


def _resolve_franka_cfg() -> tuple[Any, str]:
    candidates = [
        "isaaclab_assets.robots.franka.FRANKA_PANDA_CFG",
        "isaaclab_assets.FRANKA_PANDA_CFG",
        "isaaclab_assets.robots.FRANKA_PANDA_CFG",
        "omni.isaac.lab_assets.FRANKA_PANDA_CFG",
    ]
    import importlib

    errors = []
    for path in candidates:
        module_path, _, attr = path.rpartition(".")
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attr), path
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no candidate FRANKA_PANDA_CFG path resolved:\n  " + "\n  ".join(errors))


class IsaacSceneBackend:
    """`SceneBackend` (README §4.3) against a live Isaac Lab scene.

    Must be constructed after `idtb.sim.app.launch()` has booted
    `SimulationApp` in this process (README §4.2) -- this class's own
    imports are all lazy, but that boot is the caller's responsibility.

    **At most once per `SimulationApp` process.** Confirmed on the pod: a
    second instance builds a second `InteractiveScene`/`SimulationContext`
    against the same live stage, which either collides with the prims the
    first one already spawned or violates `SimulationContext` being a
    process-singleton the same way `SimulationApp` is -- the process died
    with no catchable Python exception. Callers that need many samples from
    one bound spec should construct this once and call `bind()`/`write_state()`
    repeatedly, not construct a new instance per sample or per test.
    """

    def __init__(self, *, resolution: tuple[int, int] = (128, 128), device: str = "cuda:0") -> None:
        self._spec: LatentSpec | None = None
        self._rig = build_rig(resolution=resolution, device=device)

    def bind(self, spec: LatentSpec) -> None:
        unsupported = [role for role in spec.roles if role not in ROLE_WRITE_PATHS]
        if unsupported:
            raise UnsupportedRoleError(
                f"IsaacSceneBackend cannot write role(s) {unsupported!r} -- "
                f"blocked per README §7.5/§11, or never spiked. Known roles: "
                f"{sorted(ROLE_WRITE_PATHS)}"
            )
        self._spec = spec

    def handles(self) -> Mapping[str, HandleInfo]:
        return handle_infos(self._require_bound())

    def write_state(self, phi: Tensor) -> None:
        write_latent_state(self._rig, self._require_bound(), phi)

    def read_state(self) -> Tensor:
        return read_latent_state(self._rig, self._require_bound())

    def render(self, sample_idx: int) -> dict[str, dict[str, Tensor]]:
        """Confirmed flaky on the pod without the discarded settle render
        below: a sensitivity check that chains many captures in one test
        would intermittently find one attribute dead -- a *different* one
        each rerun, on identical code -- while a lone one-shot capture
        (e.g. determinism's A/B pair) reliably didn't. Matches
        `spikes/spike_api.py`'s own documented history of exactly this
        ("a full render-mode switch needing a discarded settle render...
        flaky, not deterministic"). Doubles every render's cost; the right
        place to pay it is here, once, rather than in every caller."""
        self._require_bound()
        for _ in range(5):  # DIAGNOSTIC: was 2 total renders (1 discarded) -- testing 5
            self._rig.camera.update(dt=0.0, force_recompute=True)
            self._rig.sim.render()
        self._rig.camera.update(dt=0.0, force_recompute=True)
        rgb = self._rig.camera.data.output["rgb"].clone()
        seg = self._rig.camera.data.output["semantic_segmentation"].clone()
        self._last_seg = seg
        return {"cam0": {"rgb": rgb, "seg": seg.to(torch.int64)}}

    def diagnostics(self) -> dict[str, Tensor]:
        """Coarse proxies, not direct PhysX/renderer queries (README §5.3, §5.5).

        `visibility` is binary -- whether *any* cube-tagged pixel is visible
        this frame -- not the fractional-area version README §5.3 eventually
        wants; that needs a reference unoccluded-area calibration this class
        does not yet do. `collision` thresholds the distance between the
        cube and the last robot body (the end-effector, by articulation
        order) against the cube's own half-edge, the same order-of-magnitude
        approximation `read_cube_bbox_extent` already uses elsewhere.

        Cube position comes from `read_cube_position()` (USD, not the
        tensor-API root-pose buffer) -- `write_latent_state()` no longer
        writes the cube's pose through the tensor API at all (see its
        docstring), so `cube.data.root_pos_w` would be permanently stale.
        """
        if not hasattr(self, "_last_seg"):
            raise RuntimeError("diagnostics() before any render()")
        visible = float((self._last_seg > 0).any().item())
        cube_x, cube_y = read_cube_position(self._rig)
        cube_z = self._rig.ground_z + 0.5 * self._rig.default_cube_edge_m
        origin = self._rig.scene.env_origins[0]
        cube_pos = torch.tensor([cube_x, cube_y, cube_z], device=origin.device) + origin
        ee_pos = self._rig.robot.data.body_pos_w[0, -1]
        distance = float(torch.linalg.norm(cube_pos - ee_pos).item())
        collided = distance < 0.5 * self._rig.default_cube_edge_m
        return {
            "visibility": torch.tensor([visible]),
            "collision": torch.tensor([float(collided)]),
        }

    def close(self) -> None:
        # SimulationApp.close() is the caller's responsibility (README §4.4)
        # -- it's process-global, booted by idtb.sim.app.launch(), not by
        # this backend, so there is nothing scoped to this instance to close.
        pass

    def _require_bound(self) -> LatentSpec:
        if self._spec is None:
            raise RuntimeError("bind() must be called before use")
        return self._spec
