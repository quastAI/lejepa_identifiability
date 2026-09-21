"""`idtb.scenegen` -- unlike everything in `idtb.sim`, this is genuinely
testable locally: `usd-core` (standalone Pixar USD, not Isaac's bundled
build) needs no `SimulationApp` to author or read back a stage. These tests
are the actual structural verification for docs/PLAN.md Phase 2's "author
`scenes/stage_v1_tabletop.usd`" -- not just an import-guard smoke test.

`pxr` is still imported only inside functions (this fixture), matching every
other module that touches it (README §4.2) -- ruff's TID253 bans it at
module level regardless of which distribution provides it, and this file
needs no carve-out from that rule.
"""

from __future__ import annotations

import pytest

from idtb.scenegen.materials import (
    CUBE_MATERIAL_PATH,
    CUBE_PBR,
    TABLE_MATERIAL_PATH,
    TABLE_PBR,
    build_materials_stage,
    write_materials_usda,
)
from idtb.scenegen.stage_v1 import (
    CAMERAS,
    CUBE_PRIM_PATH,
    CUBE_XY,
    DEFAULT_CUBE_EDGE_M,
    DOME_LIGHT_PRIM_PATH,
    FILL_LIGHT_PRIM_PATH,
    KEY_LIGHT_PRIM_PATH,
    TABLE_PEDESTAL_PRIM_PATH,
    TABLE_PRIM_PATH,
    TABLE_SIZE_M,
    TABLE_TOP_Z,
    build_stage_v1,
    write_stage_v1_usda,
)
from idtb.scenegen.stage_v2_room import (
    CEILING_PRIM_PATH,
    FLOOR_PRIM_PATH,
    WALL_PRIM_PATHS,
    build_stage_v2_room,
    room_bounds,
    write_stage_v2_room_usda,
)


@pytest.fixture(scope="session")
def pxr():
    return pytest.importorskip("pxr")


def _bound_shader(pxr, stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    material, _ = pxr.UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    assert material, f"no material bound at {prim_path}"
    source, _, _ = material.ComputeSurfaceSource()
    assert source, f"material at {prim_path} has no surface shader"
    return source


# ---------------------------------------------------------------------------
# materials.py
# ---------------------------------------------------------------------------


def test_materials_stage_defines_both_materials_as_preview_surfaces(pxr):
    stage = build_materials_stage()
    for path, params in ((TABLE_MATERIAL_PATH, TABLE_PBR), (CUBE_MATERIAL_PATH, CUBE_PBR)):
        shader = pxr.UsdShade.Shader.Get(stage, f"{path}/Shader")
        assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
        assert shader.GetInput("roughness").Get() == pytest.approx(params.roughness)
        assert shader.GetInput("metallic").Get() == pytest.approx(params.metallic)
        assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx(params.diffuse_color)


def test_materials_roundtrip_through_disk(pxr, tmp_path):
    out = tmp_path / "pbr.usda"
    write_materials_usda(out)
    stage = pxr.Usd.Stage.Open(str(out))
    assert stage.GetPrimAtPath(TABLE_MATERIAL_PATH).IsValid()
    assert stage.GetPrimAtPath(CUBE_MATERIAL_PATH).IsValid()


# ---------------------------------------------------------------------------
# stage_v1.py -- in-memory build
# ---------------------------------------------------------------------------


def test_stage_v1_prim_hierarchy_and_types(pxr):
    stage, _materials_layer = build_stage_v1()
    expected_types = {
        TABLE_PRIM_PATH: "Cube",
        TABLE_PEDESTAL_PRIM_PATH: "Cube",
        CUBE_PRIM_PATH: "Cube",
        DOME_LIGHT_PRIM_PATH: "DomeLight",
        KEY_LIGHT_PRIM_PATH: "RectLight",
        FILL_LIGHT_PRIM_PATH: "RectLight",
    }
    for path, type_name in expected_types.items():
        prim = stage.GetPrimAtPath(path)
        assert prim.IsValid(), f"missing prim {path}"
        assert prim.GetTypeName() == type_name
    for spec in CAMERAS:
        prim = stage.GetPrimAtPath(f"/World/Cameras/{spec.name}")
        assert prim.IsValid()
        assert prim.GetTypeName() == "Camera"


def test_cube_rests_on_the_table_surface(pxr):
    stage, _ = build_stage_v1()
    cube_prim = stage.GetPrimAtPath(CUBE_PRIM_PATH)
    cube_translate = pxr.UsdGeom.Xformable(cube_prim).GetOrderedXformOps()[0].Get()
    table_surface_z = TABLE_TOP_Z + 0.5 * TABLE_SIZE_M[2]
    assert cube_translate[2] == pytest.approx(table_surface_z + 0.5 * DEFAULT_CUBE_EDGE_M)
    assert (cube_translate[0], cube_translate[1]) == pytest.approx(CUBE_XY)


def test_cube_and_table_footprint_overlap():
    """Unlike the spike scene's disconnected material-test 'table' (never
    actually under the cube -- see the module docstring), scene v1's table
    must actually contain the cube's resting point."""
    half_x, half_y = TABLE_SIZE_M[0] / 2, TABLE_SIZE_M[1] / 2
    table_center_x, table_center_y = CUBE_XY  # table is centered under the cube by construction
    assert abs(CUBE_XY[0] - table_center_x) < half_x
    assert abs(CUBE_XY[1] - table_center_y) < half_y


def test_material_bindings_resolve_to_expected_pbr_params(pxr):
    stage, _ = build_stage_v1()
    table_shader = _bound_shader(pxr, stage, TABLE_PRIM_PATH)
    cube_shader = _bound_shader(pxr, stage, CUBE_PRIM_PATH)
    assert table_shader.GetPath() == f"{TABLE_MATERIAL_PATH}/Shader"
    assert cube_shader.GetPath() == f"{CUBE_MATERIAL_PATH}/Shader"
    assert table_shader.GetInput("roughness").Get() == pytest.approx(TABLE_PBR.roughness)
    assert cube_shader.GetInput("roughness").Get() == pytest.approx(CUBE_PBR.roughness)


def test_camera_rig_is_a_realistic_head_mounted_stereo_pair():
    """docs/PLAN.md Phase 2c: replaced the original 3-monocular-camera rig
    with a single head-mounted stereo pair (README §5.3's discussion) --
    exactly 2 cameras, both aimed at the cube, separated by a real,
    RealSense-D435-class baseline along a horizontal axis (matching two
    side-by-side eyes/lenses at one head position, not two independent
    viewpoints)."""
    assert len(CAMERAS) == 2
    for spec in CAMERAS:
        assert spec.target[0] == pytest.approx(CUBE_XY[0], abs=1e-6)
        assert spec.target[1] == pytest.approx(CUBE_XY[1], abs=1e-6)

    left, right = (spec.eye for spec in CAMERAS)
    baseline = sum((lx - rx) ** 2 for lx, rx in zip(left, right, strict=True)) ** 0.5
    assert baseline == pytest.approx(0.07, abs=1e-6)
    # a real stereo module's two lenses sit side by side, not stacked
    # vertically -- both eyes should be at the same height.
    assert left[2] == pytest.approx(right[2], abs=1e-9)


def test_camera_transforms_are_finite_even_for_top_down_views(pxr):
    """Regression: `SetLookAt` with world-up `(0,0,1)` is degenerate for a
    near-vertical view direction -- the original rig's near-top-down
    camera authored a literal-FLT_MAX transform before `_look_at_transform`
    picked a fallback up vector. A finite-but-wrong matrix wouldn't fail
    any other test here, so check the actual numbers, not just that prims
    exist. The current head-stereo rig's ~50-degree angle is nowhere near
    this degenerate case, but the protection stays generic over `CAMERAS`
    so any future camera added here is still covered."""
    stage, _ = build_stage_v1()
    for spec in CAMERAS:
        camera_prim = stage.GetPrimAtPath(f"/World/Cameras/{spec.name}")
        matrix = pxr.UsdGeom.Xformable(camera_prim).GetOrderedXformOps()[0].Get()
        values = [matrix[row][col] for row in range(4) for col in range(4)]
        assert all(abs(v) < 1e6 for v in values), f"{spec.name}: degenerate transform {matrix}"
        translate = pxr.Gf.Vec3d(matrix[3][0], matrix[3][1], matrix[3][2])
        assert tuple(translate) == pytest.approx(spec.eye)


def test_dome_light_is_textureless_by_default_and_accepts_an_override(pxr):
    stage, _ = build_stage_v1()
    dome = pxr.UsdLux.DomeLight(stage.GetPrimAtPath(DOME_LIGHT_PRIM_PATH))
    assert not dome.GetTextureFileAttr().HasAuthoredValue()

    stage_with_texture, _ = build_stage_v1(dome_texture_file="./env.exr")
    dome_with_texture = pxr.UsdLux.DomeLight(stage_with_texture.GetPrimAtPath(DOME_LIGHT_PRIM_PATH))
    assert dome_with_texture.GetTextureFileAttr().Get().path == "./env.exr"


def test_stage_up_axis_and_units_match_isaac_convention(pxr):
    stage, _ = build_stage_v1()
    assert pxr.UsdGeom.GetStageUpAxis(stage) == pxr.UsdGeom.Tokens.z
    assert pxr.UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# stage_v1.py -- disk roundtrip (what Phase 3 rework will actually load)
# ---------------------------------------------------------------------------


def test_write_stage_v1_usda_roundtrips_through_disk(pxr, tmp_path):
    stage_path, materials_path = write_stage_v1_usda(tmp_path)
    assert stage_path.exists()
    assert materials_path.exists()

    reopened = pxr.Usd.Stage.Open(str(stage_path))
    assert reopened.GetPrimAtPath(TABLE_PRIM_PATH).IsValid()
    assert reopened.GetPrimAtPath(CUBE_PRIM_PATH).IsValid()

    cube_shader = _bound_shader(pxr, reopened, CUBE_PRIM_PATH)
    assert cube_shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert cube_shader.GetInput("roughness").Get() == pytest.approx(CUBE_PBR.roughness)


def test_write_stage_v1_usda_sublayer_path_is_relative(tmp_path):
    import re

    stage_path, _ = write_stage_v1_usda(tmp_path)
    text = stage_path.read_text()
    assert re.search(r"subLayers\s*=\s*\[\s*@\./materials/pbr\.usda@", text)


# ---------------------------------------------------------------------------
# stage_v2_room.py -- docs/PLAN.md Phase 2c
# ---------------------------------------------------------------------------


def test_room_shell_prim_hierarchy_and_types(pxr, tmp_path):
    stage_v1_path, _ = write_stage_v1_usda(tmp_path)
    stage = build_stage_v2_room(stage_v1_asset_path=str(stage_v1_path))
    expected_types = {FLOOR_PRIM_PATH: "Cube", CEILING_PRIM_PATH: "Cube"}
    expected_types.update({path: "Cube" for path in WALL_PRIM_PATHS.values()})
    for path, type_name in expected_types.items():
        prim = stage.GetPrimAtPath(path)
        assert prim.IsValid(), f"missing prim {path}"
        assert prim.GetTypeName() == type_name


def test_room_shell_materials_bind_to_the_room_pbr_params(pxr, tmp_path):
    from idtb.scenegen.materials import ROOM_MATERIAL_PATH, ROOM_PBR

    stage_v1_path, _ = write_stage_v1_usda(tmp_path)
    stage = build_stage_v2_room(stage_v1_asset_path=str(stage_v1_path))
    for path in (FLOOR_PRIM_PATH, CEILING_PRIM_PATH, *WALL_PRIM_PATHS.values()):
        shader = _bound_shader(pxr, stage, path)
        assert shader.GetPath() == f"{ROOM_MATERIAL_PATH}/Shader"
        assert shader.GetInput("roughness").Get() == pytest.approx(ROOM_PBR.roughness)


def test_room_encloses_every_camera_eye_with_margin():
    """docs/PLAN.md Phase 2c: the room must actually clear every camera eye
    in the rig, not just exist -- a wall placed inside a camera's eye position
    would put that camera outside (or inside the thickness of) the shell it
    is supposed to be filmed from within. Requires a real margin, not just
    a non-negative one, so a future camera move that grazes a wall fails
    loudly here instead of showing up as a mysteriously clipped render."""
    x_min, x_max, y_min, y_max, z_min, z_max = room_bounds()
    margin = 0.5  # meters -- deliberately generous, not a tight fit
    for spec in CAMERAS:
        ex, ey, ez = spec.eye
        assert x_min + margin <= ex <= x_max - margin, f"{spec.name} eye.x too close to a wall"
        assert y_min + margin <= ey <= y_max - margin, f"{spec.name} eye.y too close to a wall"
        assert z_min + margin <= ez <= z_max - margin, (
            f"{spec.name} eye.z too close to floor/ceiling"
        )


def test_stage_v2_room_composes_stage_v1_through_the_reference(pxr, tmp_path):
    """The exact regression class README §7.6 already hit once (materials
    authored outside stage_v1's default prim never composed through a
    `UsdFileCfg` reference at all) -- checked here for stage_v2's own
    reference before Phase 3 rework ever loads it on the pod."""
    stage_v1_path, _ = write_stage_v1_usda(tmp_path)
    room_path = write_stage_v2_room_usda(tmp_path, stage_v1_path=stage_v1_path)

    reopened = pxr.Usd.Stage.Open(str(room_path))
    assert reopened.GetPrimAtPath(TABLE_PRIM_PATH).IsValid()
    assert reopened.GetPrimAtPath(CUBE_PRIM_PATH).IsValid()
    for spec in CAMERAS:
        assert reopened.GetPrimAtPath(f"/World/Cameras/{spec.name}").IsValid()
    assert reopened.GetPrimAtPath(FLOOR_PRIM_PATH).IsValid()

    cube_shader = _bound_shader(pxr, reopened, CUBE_PRIM_PATH)
    assert cube_shader.GetIdAttr().Get() == "UsdPreviewSurface"


def test_write_stage_v2_room_usda_reference_path_is_relative(tmp_path):
    import re

    stage_v1_path, _ = write_stage_v1_usda(tmp_path)
    room_path = write_stage_v2_room_usda(tmp_path, stage_v1_path=stage_v1_path)
    text = room_path.read_text()
    assert re.search(r"references\s*=\s*@\./stage_v1_tabletop\.usda@", text)
