"""Authors `scenes/stage_v1_tabletop.usd` (docs/PLAN.md Phase 2): a real
tabletop, PBR materials, an HDRI dome + two area lights, and a 3-camera rig
with real parallax -- replacing the spike scene's disconnected material-test
"table" prop (never actually load-bearing: the cube always rested on the
ground plane, at `ground_z`, regardless of the table's position) and single
camera.

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


CAMERAS: tuple[CameraSpec, ...] = (
    # High, oblique, near-top-down -- the spike's own placement (README
    # §5.3/§7.4), kept for continuity with what's already measured.
    CameraSpec("Camera1", eye=(1.0, 1.0, TABLE_TOP_Z + 0.5), target=_CUBE_TARGET),
    # A second view along a distinctly different azimuth (roughly
    # perpendicular to Camera1 about the cube, not just offset along the
    # same axis) so world-Y motion projects with real magnitude in at least
    # one view -- directly targeting the weak `cube.y` signal measured from
    # Camera1's single angle alone (README §5.3, docs/PLAN.md carried-forward
    # item 3).
    CameraSpec("Camera2", eye=(CUBE_XY[0], 1.3, TABLE_TOP_Z + 0.3), target=_CUBE_TARGET),
    # Near-top-down: both cube.x and cube.y project with comparable
    # magnitude, and it's independently the strongest arm-over-cube
    # occlusion mitigation (README §5.3 mitigation 2).
    CameraSpec("Camera3", eye=(CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z + 1.0), target=(
        CUBE_XY[0],
        CUBE_XY[1],
        TABLE_TOP_Z,
    )),
)


def _set_box_prim(
    stage: Any,
    path: str,
    *,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
) -> Any:
    """A `UsdGeom.Cube` (default size=2, i.e. unit half-extent) scaled to an
    arbitrary axis-aligned box -- same pattern `writer.py::_write_cube_scale`
    already uses for `cube.size` (a scale op on top of an authored unit
    edge), so the two stay consistent."""
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xformable = UsdGeom.Xformable(cube)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*center))
    xformable.AddScaleOp().Set(Gf.Vec3f(*size))
    return cube


def _bind_material(stage: Any, prim_path: str, material_path: str) -> None:
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    material = UsdShade.Material.Get(stage, material_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _look_at_transform(eye: tuple[float, float, float], target: tuple[float, float, float]) -> Any:
    """Camera-prim local-to-world transform looking from `eye` at `target`.

    `Gf.Matrix4d.SetLookAt` builds the *view* matrix (world -> camera
    space); a prim's authored transform is the inverse of that.

    World-up `(0,0,1)` is degenerate for a near-vertical view direction --
    confirmed empirically, not just in theory: Camera3's authored matrix
    came out as literal FLT_MAX entries the first time this ran, because
    its top-down eye/target pair makes the view direction parallel to that
    up vector, collapsing `SetLookAt`'s internal cross product to zero
    before the inversion blows it up. Falls back to world `(0,1,0)` whenever
    the view direction is nearly parallel to `(0,0,1)`.
    """
    from pxr import Gf

    view_dir = (Gf.Vec3d(*target) - Gf.Vec3d(*eye)).GetNormalized()
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    if abs(Gf.Dot(view_dir, up)) > 0.99:
        up = Gf.Vec3d(0.0, 1.0, 0.0)
    view = Gf.Matrix4d(1.0).SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up)
    return view.GetInverse()


def _define_camera(stage: Any, spec: CameraSpec) -> Any:
    from pxr import UsdGeom

    path = f"{CAMERA_PARENT_PATH}/{spec.name}"
    camera = UsdGeom.Camera.Define(stage, path)
    UsdGeom.Xformable(camera).AddTransformOp().Set(_look_at_transform(spec.eye, spec.target))
    return camera


def _define_dome_light(stage: Any, *, texture_file: str | None) -> Any:
    """`UsdLux.DomeLight`: ambient HDRI fill. No texture asset is pinned by
    default (README's "no premature pins" -- Isaac's Nucleus asset root for
    a shipped HDRI is resolved dynamically, same reasoning as the Franka
    path in the package docstring); `texture_file` lets a caller wire one in
    once a real, resolved asset path exists. A textureless dome still gives
    a valid, uniform ambient term via `inputs:color`."""
    from pxr import Sdf, UsdLux

    light = UsdLux.DomeLight.Define(stage, DOME_LIGHT_PRIM_PATH)
    light.CreateIntensityAttr(1000.0)
    light.CreateColorAttr((0.75, 0.8, 0.85))  # cool, neutral sky fill
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
    `_bind_material`'s `UsdShade.Material.Get(stage, ...)` resolves within
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

    _set_box_prim(
        stage,
        TABLE_PRIM_PATH,
        size=TABLE_SIZE_M,
        center=(CUBE_XY[0], CUBE_XY[1], TABLE_TOP_Z),
    )
    _bind_material(stage, TABLE_PRIM_PATH, TABLE_MATERIAL_PATH)

    pedestal_size = (0.1, 0.1, TABLE_TOP_Z - 0.5 * TABLE_SIZE_M[2])
    _set_box_prim(
        stage,
        TABLE_PEDESTAL_PRIM_PATH,
        size=pedestal_size,
        center=(CUBE_XY[0], CUBE_XY[1], 0.5 * pedestal_size[2]),
    )
    _bind_material(stage, TABLE_PEDESTAL_PRIM_PATH, TABLE_MATERIAL_PATH)

    cube_z = TABLE_TOP_Z + 0.5 * TABLE_SIZE_M[2] + 0.5 * DEFAULT_CUBE_EDGE_M
    _set_box_prim(
        stage,
        CUBE_PRIM_PATH,
        size=(DEFAULT_CUBE_EDGE_M,) * 3,
        center=(CUBE_XY[0], CUBE_XY[1], cube_z),
    )
    _bind_material(stage, CUBE_PRIM_PATH, CUBE_MATERIAL_PATH)

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
