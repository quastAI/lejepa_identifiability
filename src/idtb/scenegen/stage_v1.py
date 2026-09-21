"""Authors `scenes/stage_v1_tabletop.usd` (docs/PLAN.md Phase 2): a real
tabletop, PBR materials, an HDRI dome + two area lights, and a head-mounted
stereo camera pair -- replacing the spike scene's disconnected
material-test "table" prop (never actually load-bearing: the cube always
rested on the ground plane, at `ground_z`, regardless of the table's
position) and single camera.

Scope, deliberately: this file authors the *static* scene -- table, cube,
materials, lights, cameras. It does not spawn the Franka (see the package
docstring) and it does not touch the latent-to-physical wiring (cube resting
height, squash radii) -- both are Phase 3 rework's and Phase 2b's jobs
respectively, once this stage's real table extent is a measured thing rather
than a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from idtb.scenegen.materials import CUBE_MATERIAL_PATH, TABLE_MATERIAL_PATH, define_materials
from idtb.scenegen.primitives import bind_material, set_box_prim

#: A real worktable height and a top surface big enough for the Franka's
#: reach to matter, replacing the spike scene's 0.3x0.3 material-test pad
#: that the cube never actually sat on.
TABLE_TOP_Z = 0.40
TABLE_SIZE_M = (0.8, 0.6, 0.04)
CUBE_XY = (0.5, 0.0)
DEFAULT_CUBE_EDGE_M = 0.06

TABLE_PRIM_PATH = "/World/Table"
TABLE_PEDESTAL_PRIM_PATH = "/World/TablePedestal"
CUBE_PRIM_PATH = "/World/Cube"
DOME_LIGHT_PRIM_PATH = "/World/Lights/Dome"
#: The `light.intensity`/`light.warmth` target going forward (README §5.2.3)
#: -- the dome is ambient/global and, per Phase 2b, needs its own re-gate
#: rather than doubling as the per-sample-driven handle.
KEY_LIGHT_PRIM_PATH = "/World/Lights/KeyLight"
FILL_LIGHT_PRIM_PATH = "/World/Lights/FillLight"
CAMERA_PARENT_PATH = "/World/Cameras"

_CUBE_TARGET = (CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z + 0.1)


@dataclass(frozen=True)
class CameraSpec:
    """One authored camera prim: name, eye, and look-at target (world frame).

    The runtime re-aims cameras every capture via `set_world_poses_from_view`
    (`writer.py::_aim_camera`), so this authored transform only matters for
    Phase 2's own debug-preset visual smoke test, before that runtime
    machinery is wired up in Phase 3 rework -- but it should still look
    sensible on its own.
    """

    name: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]


#: docs/PLAN.md Phase 2c, replacing the original 3-monocular-camera rig:
#: a single head-mounted stereo pair, matching how real humanoids actually
#: see (Figure 01/02, 1X NEO, Unitree H1/G1 all use head-mounted stereo as
#: their primary/sole vision sensor) -- not a wrist/eye-in-hand camera,
#: which would move with the arm's own `full`-latent joints and confound
#: monocular scale cues with `cube.size` exactly the way this rig is meant
#: to avoid, and not the previous rig's arbitrary oblique "surveillance
#: camera" placement.
#:
#: The head is positioned as a humanoid actually leaning over the table
#: would be -- looking down at roughly 50 degrees below horizontal from
#: about 0.8 m away, not hovering perfectly overhead -- and
#: `_STEREO_BASELINE_M` (70 mm) matches a real wrist/head-mountable stereo
#: module (Intel RealSense D435-class; a ZED-class ~120 mm module is wider
#: than a humanoid head can usually carry). `_stereo_eye_pair` offsets two
#: eyes from that one head position along the head's own horizontal "right"
#: vector, the same way two real eyes/lenses sit side by side.
#:
#: The previous rig's `Camera2`/`Camera3` existed to patch two problems
#: specific to `Camera1`'s arbitrary angle: a weak `cube.y` signal from a
#: single shallow azimuth, and arm-over-cube occlusion (README §5.3,
#: docs/PLAN.md carried-forward item 3 and Context item 5). Both are
#: deliberately left unverified against this new single viewpoint for now:
#: occlusion cannot be measured at all in this static scene regardless --
#: the Franka is not spawned here (see module docstring) -- and the
#: `cube.y` re-check is real but not blocking, deferred to whenever this
#: rig is next pod-tested against the real backend (Phase 3 rework).
_STEREO_BASELINE_M = 0.07

_HEAD_TARGET = _CUBE_TARGET
_HEAD_EYE = (CUBE_XY[0] + 0.35, CUBE_XY[1] + 0.35, _CUBE_TARGET[2] + 0.6)


def _stereo_eye_pair(
    eye: tuple[float, float, float], target: tuple[float, float, float], baseline: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Two eyes either side of `eye`, offset by `baseline` along the head's
    own horizontal "right" vector (perpendicular to the view direction and
    to world-up) -- the same geometry two real side-by-side lenses/eyes
    sit in. Degenerate only if `eye`/`target` are stacked exactly on the
    world-up axis, which this rig's ~50-degree look-down angle is nowhere
    near; unlike `_look_at_transform`, not worth a fallback for an angle
    this rig will never author.

    Plain-tuple arithmetic, not `pxr.Gf` -- this is called at module scope
    to build `CAMERAS` (see below), and a `pxr` import here would run at
    *import time*, exactly the module-level Isaac import §4.2 bans (caught
    by `tests/test_import_guard.py`, not just a style preference). A 3D
    cross product needs no vector library.
    """
    import math

    ex, ey, ez = eye
    tx, ty, tz = target
    vx, vy, vz = tx - ex, ty - ey, tz - ez
    vlen = math.sqrt(vx * vx + vy * vy + vz * vz)
    vx, vy, vz = vx / vlen, vy / vlen, vz / vlen
    # cross(view_dir, world_up=(0,0,1))
    rx, ry, rz = vy, -vx, 0.0
    rlen = math.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / rlen, ry / rlen, rz / rlen
    half = baseline / 2.0
    left = (ex - rx * half, ey - ry * half, ez - rz * half)
    right_eye = (ex + rx * half, ey + ry * half, ez + rz * half)
    return left, right_eye


_LEFT_EYE, _RIGHT_EYE = _stereo_eye_pair(_HEAD_EYE, _HEAD_TARGET, _STEREO_BASELINE_M)

CAMERAS: tuple[CameraSpec, ...] = (
    CameraSpec("CameraL", eye=_LEFT_EYE, target=_HEAD_TARGET),
    CameraSpec("CameraR", eye=_RIGHT_EYE, target=_HEAD_TARGET),
)


def _look_at_transform(eye: tuple[float, float, float], target: tuple[float, float, float]) -> Any:
    """Camera-prim local-to-world transform looking from `eye` at `target`.

    `Gf.Matrix4d.SetLookAt` builds the *view* matrix (world -> camera
    space); a prim's authored transform is the inverse of that.

    World-up `(0,0,1)` is degenerate for a near-vertical view direction --
    confirmed empirically, not just in theory: the original rig's
    near-top-down camera authored a literal-FLT_MAX transform the first
    time this ran, because its top-down eye/target pair makes the view
    direction parallel to that up vector, collapsing `SetLookAt`'s internal
    cross product to zero before the inversion blows it up. Falls back to
    world `(0,1,0)` whenever the view direction is nearly parallel to
    `(0,0,1)` -- kept even though the current head-stereo rig's ~50-degree
    look-down angle is nowhere near that case, since any future camera
    added here gets the same protection for free.
    """
    from pxr import Gf

    view_dir = (Gf.Vec3d(*target) - Gf.Vec3d(*eye)).GetNormalized()
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    if abs(Gf.Dot(view_dir, up)) > 0.99:
        up = Gf.Vec3d(0.0, 1.0, 0.0)
    view = Gf.Matrix4d(1.0).SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up)
    return view.GetInverse()


def _define_camera(stage: Any, spec: CameraSpec) -> Any:
    from pxr import Gf, UsdGeom

    path = f"{CAMERA_PARENT_PATH}/{spec.name}"
    camera = UsdGeom.Camera.Define(stage, path)
    UsdGeom.Xformable(camera).AddTransformOp().Set(_look_at_transform(spec.eye, spec.target))
    # `UsdGeom.Camera`'s schema fallback is `(1, 1000000)` -- a 1 m near
    # plane, confirmed locally (no camera authored one before this). That's
    # larger than this scene's own scale: docs/PLAN.md Phase 2c's pod
    # session found it clipping the original rig's near foreground, and one
    # camera's eye-to-target distance sat almost exactly on that 1 m
    # boundary. 1 cm near / 100 m far comfortably covers a
    # tabletop-and-a-room-sized scene either way, independent of exactly
    # how far any future camera sits from its target.
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    return camera


def _define_dome_light(stage: Any, *, texture_file: str | None) -> Any:
    """`UsdLux.DomeLight`: ambient HDRI fill. No texture asset is pinned by
    default (README's "no premature pins" -- Isaac's Nucleus asset root for
    a shipped HDRI is resolved dynamically, same reasoning as the Franka
    path in the package docstring); `texture_file` lets a caller wire one in
    once a real, resolved asset path exists. A textureless dome still gives
    a valid, uniform ambient term via `inputs:color`.

    This is an indoor tabletop scene with no room shell yet (README §9's
    room-shell scene v2 is deferred) -- the fallback color is a neutral,
    slightly warm tone standing in for generic indoor wall/ceiling bounce,
    not the cool blue a sky dome would use for an outdoor scene.
    """
    from pxr import Sdf, UsdLux

    light = UsdLux.DomeLight.Define(stage, DOME_LIGHT_PRIM_PATH)
    light.CreateIntensityAttr(1000.0)
    light.CreateColorAttr((0.85, 0.83, 0.80))  # neutral indoor bounce, not sky
    if texture_file is not None:
        light.CreateTextureFileAttr().Set(Sdf.AssetPath(texture_file))
    return light


def _define_rect_light(
    stage: Any,
    path: str,
    *,
    translate: tuple[float, float, float],
    intensity: float,
    width: float,
    height: float,
) -> Any:
    from pxr import Gf, UsdGeom, UsdLux

    light = UsdLux.RectLight.Define(stage, path)
    light.CreateIntensityAttr(intensity)
    light.CreateWidthAttr(width)
    light.CreateHeightAttr(height)
    light.CreateEnableColorTemperatureAttr(True)
    light.CreateColorTemperatureAttr(5500.0)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*translate))
    return light


def build_stage_v1(*, dome_texture_file: str | None = None) -> tuple[Any, Any]:
    """Returns `(stage, materials_layer)`.

    `materials_layer` is authored via `define_materials` as an *anonymous
    sublayer* of `stage`'s root layer, added before anything else so that
    `bind_material`'s `UsdShade.Material.Get(stage, ...)` resolves within
    this same build pass -- binding against a material that doesn't yet
    exist anywhere the stage can see would silently author a dangling
    relationship. :func:`write_stage_v1_usda` exports `materials_layer` to
    its own file and repoints the root layer's sublayer entry at it, so the
    two files it hands `docs/PLAN.md` reference each other exactly the way
    a caller loading `stage_v1_tabletop.usd` off disk would see.
    """
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    materials_layer = Sdf.Layer.CreateAnonymous("materials")
    stage.GetRootLayer().subLayerPaths.append(materials_layer.identifier)
    with Usd.EditContext(stage, materials_layer):
        define_materials(stage)

    set_box_prim(
        stage,
        TABLE_PRIM_PATH,
        size=TABLE_SIZE_M,
        center=(CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z),
    )
    bind_material(stage, TABLE_PRIM_PATH, TABLE_MATERIAL_PATH)

    pedestal_size = (0.1, 0.1, TABLE_TOP_Z - 0.5 * TABLE_SIZE_M[2])
    set_box_prim(
        stage,
        TABLE_PEDESTAL_PRIM_PATH,
        size=pedestal_size,
        center=(CUBE_XY[0], CUBE_XY[1], 0.5 * pedestal_size[2]),
    )
    bind_material(stage, TABLE_PEDESTAL_PRIM_PATH, TABLE_MATERIAL_PATH)

    cube_z = TABLE_TOP_Z + 0.5 * TABLE_SIZE_M[2] + 0.5 * DEFAULT_CUBE_EDGE_M
    set_box_prim(
        stage,
        CUBE_PRIM_PATH,
        size=(DEFAULT_CUBE_EDGE_M,) * 3,
        center=(CUBE_XY[0], CUBE_XY[1], cube_z),
    )
    bind_material(stage, CUBE_PRIM_PATH, CUBE_MATERIAL_PATH)

    _define_dome_light(stage, texture_file=dome_texture_file)
    key_translate = (CUBE_XY[0] - 0.3, CUBE_XY[1] - 0.3, TABLE_TOP_Z + 1.0)
    _define_rect_light(
        stage, KEY_LIGHT_PRIM_PATH, translate=key_translate, intensity=3000.0, width=0.4, height=0.4
    )
    fill_translate = (CUBE_XY[0] + 0.4, CUBE_XY[1] + 0.4, TABLE_TOP_Z + 0.8)
    _define_rect_light(
        stage,
        FILL_LIGHT_PRIM_PATH,
        translate=fill_translate,
        intensity=800.0,
        width=0.6,
        height=0.6,
    )

    for spec in CAMERAS:
        _define_camera(stage, spec)

    return stage, materials_layer


def write_stage_v1_usda(
    output_dir: Path,
    *,
    stage_filename: str = "stage_v1_tabletop.usda",
    materials_relpath: str = "materials/pbr.usda",
    dome_texture_file: str | None = None,
) -> tuple[Path, Path]:
    """Writes `<output_dir>/<stage_filename>` and `<output_dir>/<materials_relpath>`.

    Returns `(stage_path, materials_path)`.
    """
    stage, materials_layer = build_stage_v1(dome_texture_file=dome_texture_file)

    materials_path = output_dir / materials_relpath
    materials_path.parent.mkdir(parents=True, exist_ok=True)
    materials_layer.Export(str(materials_path))

    root_layer = stage.GetRootLayer()
    sublayer_paths = list(root_layer.subLayerPaths)
    idx = sublayer_paths.index(materials_layer.identifier)
    root_layer.subLayerPaths[idx] = f"./{materials_relpath}"

    stage_path = output_dir / stage_filename
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    root_layer.Export(str(stage_path))
    return stage_path, materials_path
