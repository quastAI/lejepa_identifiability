"""USD-authoring helpers shared by `stage_v1.py` and `stage_v2_room.py` --
axis-aligned box prims and material bindings, the only two operations both
a tabletop's furniture and a room's walls/floor/ceiling need. Split out of
`stage_v1.py` when Phase 2c's room shell needed the identical box-prim
pattern a second time (docs/PLAN.md Phase 2c).
"""

from __future__ import annotations

from typing import Any


def set_box_prim(
    stage: Any,
    path: str,
    *,
    size: tuple[float, float, float],
    center: tuple[float, float, float],
) -> Any:
    """A `UsdGeom.Cube` (default size=2, i.e. unit half-extent) scaled to an
    arbitrary axis-aligned box -- same pattern `writer.py::_write_cube_scale`
    uses for `cube.size` (a scale op on top of an authored unit edge), so
    every authored box in the project stays consistent with how the runtime
    reads one back."""
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xformable = UsdGeom.Xformable(cube)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*center))
    xformable.AddScaleOp().Set(Gf.Vec3f(*size))
    return cube


def bind_material(stage: Any, prim_path: str, material_path: str) -> None:
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    material = UsdShade.Material.Get(stage, material_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
