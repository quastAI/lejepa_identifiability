"""Authors `scenes/stage_v2_room.usda` (docs/PLAN.md Phase 2c): an empty
room shell -- floor, ceiling, four walls -- around scene v1's table, cube,
lights, and 3-camera rig.

Composed by *referencing* `stage_v1_tabletop.usda`'s `/World` onto this
file's own `/World` (its own default prim), not by extending
`stage_v1.py`'s build function or sublayering it in: a reference is the
idiomatic arc for pulling in an encapsulated, already-complete asset
(`materials.py`'s docstring already establishes this for the Franka), and
it keeps `stage_v1_tabletop.usda` exactly what Phase 2 shipped and tested,
unmodified, whether or not a room ever wraps it. The reference's target
primPath (`/World`) equals this file's own default prim path, so every
absolute path stage_v1 already authored (`/World/Table`,
`/World/Cameras/Camera1`, ...) survives unchanged into stage_v2 -- Phase 3
rework's runtime code that looks those paths up does not need to change
when it switches from loading stage_v1 to stage_v2.

Because a *reference* (unlike an anonymous sublayer) needs a real anchoring
file to resolve a relative asset path against, `build_stage_v2_room` always
takes a real, already-written `stage_v1_tabletop.usda` path -- there is no
"build everything fully in-memory first" step here the way `stage_v1.py`
has for its own sublayer, since the thing being composed in is itself a
file on disk (confirmed locally: an in-memory stage's own anonymous
identifier cannot anchor a relative reference -- `write_stage_v2_room_usda`
builds with an absolute reference for that reason and repoints it to a
relative one only at export, mirroring `write_stage_v1_usda`'s identical
repoint-at-export step for its materials sublayer).

Room dimensions are chosen, not guessed, to clear all three of stage_v1's
camera eyes with real margin (see `room_bounds()` and
`tests/test_scenegen.py::test_room_encloses_every_camera_eye_with_margin`,
which would fail if a future edit shrank the room or moved a camera without
re-checking this): a 5x5 m floor footprint and 3 m ceiling, centered on
`CUBE_XY` (the table's own centering point) rather than on the world
origin, which the table is not centered on.

The dome light is *not* re-authored here. `stage_v1.py`'s composed-in
`DomeLight` stays textureless by default for the same "no premature pins"
reason it started that way (README §3.1's Decision Register): no resolved
HDRI asset path exists yet. Recorded as still-open in README §3.2, not
silently assumed fixed by the room's existence -- a room shell gives a
textured or textureless dome real surfaces to bounce off either way, but
which one is correct is an open question this phase does not resolve.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from idtb.scenegen.materials import ROOM_MATERIAL_PATH, define_room_material
from idtb.scenegen.primitives import bind_material, set_box_prim
from idtb.scenegen.stage_v1 import CUBE_XY

#: Interior footprint and ceiling height -- see module docstring for why.
ROOM_SIZE_M = (5.0, 5.0, 3.0)  # (x span, y span, floor-to-ceiling height)
ROOM_CENTER_XY = CUBE_XY
WALL_THICKNESS_M = 0.1

#: `stage_v1_tabletop.usda`'s own default prim -- also this file's, by
#: construction (see module docstring).
STAGE_V1_DEFAULT_PRIM_PATH = "/World"

ROOM_PARENT_PATH = "/World/Room"
FLOOR_PRIM_PATH = f"{ROOM_PARENT_PATH}/Floor"
CEILING_PRIM_PATH = f"{ROOM_PARENT_PATH}/Ceiling"
WALL_PRIM_PATHS: dict[str, str] = {
    "north": f"{ROOM_PARENT_PATH}/WallNorth",  # +y
    "south": f"{ROOM_PARENT_PATH}/WallSouth",  # -y
    "east": f"{ROOM_PARENT_PATH}/WallEast",  # +x
    "west": f"{ROOM_PARENT_PATH}/WallWest",  # -x
}


def room_bounds() -> tuple[float, float, float, float, float, float]:
    """`(x_min, x_max, y_min, y_max, z_min, z_max)` of the room's interior
    (not counting wall thickness)."""
    half_x, half_y = ROOM_SIZE_M[0] / 2, ROOM_SIZE_M[1] / 2
    cx, cy = ROOM_CENTER_XY
    return (cx - half_x, cx + half_x, cy - half_y, cy + half_y, 0.0, ROOM_SIZE_M[2])


def _define_room_shell(stage: Any) -> None:
    x_min, x_max, y_min, y_max, z_min, z_max = room_bounds()
    room_w, room_d, room_h = ROOM_SIZE_M
    t = WALL_THICKNESS_M
    wall_center_z = (z_min + z_max) / 2

    set_box_prim(
        stage,
        FLOOR_PRIM_PATH,
        size=(room_w + 2 * t, room_d + 2 * t, t),
        center=(ROOM_CENTER_XY[0], ROOM_CENTER_XY[1], z_min - t / 2),
    )
    set_box_prim(
        stage,
        CEILING_PRIM_PATH,
        size=(room_w + 2 * t, room_d + 2 * t, t),
        center=(ROOM_CENTER_XY[0], ROOM_CENTER_XY[1], z_max + t / 2),
    )
    # North/south walls span the full x width (plus corners); east/west
    # walls span the full y depth -- together they seal all four corners
    # with no gap, rather than leaving a thickness-sized notch at each one.
    set_box_prim(
        stage,
        WALL_PRIM_PATHS["north"],
        size=(room_w + 2 * t, t, room_h),
        center=(ROOM_CENTER_XY[0], y_max + t / 2, wall_center_z),
    )
    set_box_prim(
        stage,
        WALL_PRIM_PATHS["south"],
        size=(room_w + 2 * t, t, room_h),
        center=(ROOM_CENTER_XY[0], y_min - t / 2, wall_center_z),
    )
    set_box_prim(
        stage,
        WALL_PRIM_PATHS["east"],
        size=(t, room_d + 2 * t, room_h),
        center=(x_max + t / 2, ROOM_CENTER_XY[1], wall_center_z),
    )
    set_box_prim(
        stage,
        WALL_PRIM_PATHS["west"],
        size=(t, room_d + 2 * t, room_h),
        center=(x_min - t / 2, ROOM_CENTER_XY[1], wall_center_z),
    )

    for path in (FLOOR_PRIM_PATH, CEILING_PRIM_PATH, *WALL_PRIM_PATHS.values()):
        bind_material(stage, path, ROOM_MATERIAL_PATH)


def build_stage_v2_room(*, stage_v1_asset_path: str) -> Any:
    """In-memory stage: `stage_v1_asset_path`'s `/World` referenced onto
    this stage's own `/World`, plus the room shell authored directly
    alongside it.

    `stage_v1_asset_path` must be a real, resolvable path to an already
    on-disk `stage_v1_tabletop.usda` (see module docstring for why an
    in-memory stage_v1 cannot be referenced this way) -- typically absolute
    here; :func:`write_stage_v2_room_usda` repoints it to a relative path
    only once exporting to its final location.
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().GetReferences().AddReference(
        assetPath=stage_v1_asset_path, primPath=STAGE_V1_DEFAULT_PRIM_PATH
    )

    define_room_material(stage)
    _define_room_shell(stage)

    return stage


def write_stage_v2_room_usda(
    output_dir: Path,
    *,
    stage_v1_path: Path,
    stage_filename: str = "stage_v2_room.usda",
) -> Path:
    """Writes `<output_dir>/<stage_filename>`, referencing `stage_v1_path`
    by a path relative to `output_dir` -- the same portability discipline
    `write_stage_v1_usda` uses for its own materials sublayer, and for the
    same reason: a checked-in `.usda` that only resolves against the
    machine that built it is useless once copied to the pod.

    Confirmed locally (`usd-core`): repointing an already-built stage's
    reference to a relative path and then exporting produces one benign
    "could not open asset" warning on the *in-memory* stage as it
    recomposes against its own unresolvable anonymous identifier -- the
    exact same warning `write_stage_v1_usda` already produces for its
    sublayer repoint, and, like that one, harmless: the exported file's
    relative path resolves correctly once actually reopened from
    `output_dir` (`tests/test_scenegen.py`,
    `test_stage_v2_room_composes_stage_v1_through_the_reference`).
    """
    from pxr import Sdf

    stage = build_stage_v2_room(stage_v1_asset_path=str(stage_v1_path))

    root_layer = stage.GetRootLayer()
    world_spec = root_layer.GetPrimAtPath(STAGE_V1_DEFAULT_PRIM_PATH)
    old_ref = world_spec.referenceList.prependedItems[0]
    relative_path = f"./{os.path.relpath(stage_v1_path, start=output_dir)}"
    world_spec.referenceList.prependedItems = [
        Sdf.Reference(assetPath=relative_path, primPath=old_ref.primPath)
    ]

    stage_path = output_dir / stage_filename
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    root_layer.Export(str(stage_path))
    return stage_path
