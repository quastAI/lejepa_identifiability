"""Authors the static scene-v1 USD assets under `scenes/` (docs/PLAN.md Phase 2).

Pure module: uses the standalone `usd-core` PyPI package (real Pixar USD, no
Isaac/Omniverse Kit needed) so the scene's geometry, materials, lights and
camera rig can be authored *and unit-tested* locally on macOS -- unlike
everything in `idtb.sim`, which is genuinely blind until it runs on the pod.
`pxr` imports still stay function-local, matching every other Isaac-adjacent
module (README §4.2): ruff's TID253 bans `pxr` at module level regardless of
which distribution provides it, and this module needs no carve-out from that
rule to get the benefit.

What this package deliberately does *not* author: the Franka robot. Its
Nucleus asset root is resolved dynamically by `isaaclab_assets.FRANKA_PANDA_CFG`
at runtime (README §4.4's `_resolve_franka_cfg`), never a fixed literal --
baking a reference to it into a checked-in `.usda` file would pin exactly the
kind of environment-specific path the project's Decision Register avoids
pinning everywhere else. Phase 3 rework keeps spawning the robot as an
`ArticulationCfg` alongside this stage, not inside it.
"""

from idtb.scenegen.materials import CUBE_MATERIAL_PATH, TABLE_MATERIAL_PATH, build_materials_stage
from idtb.scenegen.stage_v1 import CAMERAS, CUBE_XY, TABLE_SIZE_M, TABLE_TOP_Z, build_stage_v1

__all__ = [
    "CAMERAS",
    "CUBE_MATERIAL_PATH",
    "CUBE_XY",
    "TABLE_MATERIAL_PATH",
    "TABLE_SIZE_M",
    "TABLE_TOP_Z",
    "build_materials_stage",
    "build_stage_v1",
]
