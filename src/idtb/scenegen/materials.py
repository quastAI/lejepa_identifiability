"""PBR material library for the scene-v1 stage: `scenes/materials/pbr.usda`.

`UsdPreviewSurface`, not MDL/OmniPBR: it's the schema `table.albedo`/
`cube.hue` already write through via `diffuseColor` on the spike scene's flat
`PreviewSurfaceCfg` (README §7.5, confirmed writable/bitwise-deterministic),
and NVIDIA's own RTX Renderer docs confirm both `PathTracing` and real-time
modes shade `UsdPreviewSurface` directly -- so keeping the same shader type
here means Phase 3 rework's `_resolve_bound_shader` needs a new prim
structure to search, not a new shader kind to handle.

Authored as its own layer (`materials.py` -> `pbr.usda`) and sublayered into
`stage_v1_tabletop.usda` by `stage_v1.py`, rather than referenced under a
scope -- sublayering is the idiomatic composition arc for a shared,
same-namespace material library (as opposed to `references`, which is for
pulling in an encapsulated external asset, e.g. the Franka robot).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Under `/World`, not a stage-root sibling of it -- `stage_v1.py` sets
#: `/World` as the stage's *default prim*, and a `sim_utils.UsdFileCfg`
#: reference (Phase 2b, Phase 3 rework) only pulls in the default prim's own
#: subtree, remapped under wherever it's referenced. Materials authored
#: outside that subtree (the original `/Looks`) never get pulled in at all --
#: confirmed on the pod: `MaterialBindingAPI(cube_prim).ComputeBoundMaterial()`
#: returned nothing, not because the binding was wrong, but because
#: `/Looks/CubeMaterial` doesn't exist anywhere in the composed result.
LOOKS_SCOPE = "/World/Looks"
TABLE_MATERIAL_PATH = f"{LOOKS_SCOPE}/TableMaterial"
CUBE_MATERIAL_PATH = f"{LOOKS_SCOPE}/CubeMaterial"


@dataclass(frozen=True)
class PbrParams:
    """One `UsdPreviewSurface`'s static (non-latent) look.

    `diffuse_color` is only the *default* -- `table.albedo`/`cube.hue`
    overwrite it per sample at runtime (README §5.2.2/§5.2.3, unchanged
    mechanism). Roughness/metallic are fixed scene-authoring, not latents.
    """

    diffuse_color: tuple[float, float, float]
    roughness: float
    metallic: float = 0.0


#: Matte, high-roughness, non-metallic -- a worktable surface, not a mirror.
TABLE_PBR = PbrParams(diffuse_color=(0.5, 0.5, 0.5), roughness=0.75, metallic=0.0)
#: Semi-gloss plastic: low enough roughness that `cube.hue` stays legible
#: under the key light, high enough to avoid a distracting specular hotspot
#: from `PathTracing`'s point-sampled area lights.
CUBE_PBR = PbrParams(diffuse_color=(0.85, 0.13, 0.11), roughness=0.4, metallic=0.0)


def _define_preview_surface_material(stage: Any, material_path: str, params: PbrParams) -> Any:
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*params.diffuse_color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(params.roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(params.metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def define_materials(stage: Any) -> None:
    """Author `/Looks/TableMaterial` and `/Looks/CubeMaterial` into `stage`'s
    current edit target.

    The one place this authoring logic lives -- both :func:`build_materials_stage`
    (a standalone `pbr.usda`) and `stage_v1.py` (authoring straight into a
    sublayer of the composed scene stage, so material *bindings* resolve
    within the same build pass) funnel through this, so the two can never
    drift apart.
    """
    stage.DefinePrim(LOOKS_SCOPE, "Scope")
    _define_preview_surface_material(stage, TABLE_MATERIAL_PATH, TABLE_PBR)
    _define_preview_surface_material(stage, CUBE_MATERIAL_PATH, CUBE_PBR)


def build_materials_stage() -> Any:
    """In-memory stage holding `/Looks/TableMaterial` and `/Looks/CubeMaterial`."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    define_materials(stage)
    return stage


def write_materials_usda(output_path: Path) -> None:
    stage = build_materials_stage()
    stage.GetRootLayer().Export(str(output_path))
