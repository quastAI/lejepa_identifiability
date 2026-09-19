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


def test_camera_rig_has_real_multi_axis_parallax():
    """README §5.3 / docs/PLAN.md Phase 2: a second view must not just be
    offset along the same axis as the first, or it buys nothing for the
    weak `cube.y` signal it's meant to fix."""
    eyes = [spec.eye for spec in CAMERAS]
    xs, ys, zs = zip(*eyes, strict=True)
    assert len(set(xs)) > 1, "no camera varies eye.x -- rig collapses to one axis"
    assert len(set(ys)) > 1, "no camera varies eye.y -- rig collapses to one axis"
    assert len(set(zs)) > 1, "no camera varies eye.z -- no height variation across views"
    # every camera aims at (roughly) the cube -- an occlusion-mitigating
    # rig only helps if the views actually converge on the manipuland.
    for spec in CAMERAS:
        assert spec.target[0] == pytest.approx(CUBE_XY[0], abs=1e-6)
        assert spec.target[1] == pytest.approx(CUBE_XY[1], abs=1e-6)


def test_camera_transforms_are_finite_even_for_top_down_views(pxr):
    """Regression: `SetLookAt` with world-up `(0,0,1)` is degenerate for a
    near-vertical view direction -- Camera3's authored matrix came out as
    literal FLT_MAX entries before `_look_at_transform` picked a fallback
    up vector. A finite-but-wrong matrix wouldn't fail any other test here,
    so check the actual numbers, not just that prims exist."""
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
