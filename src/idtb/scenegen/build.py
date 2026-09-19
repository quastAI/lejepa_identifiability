"""Entry point: regenerates the checked-in `scenes/stage_v1_tabletop.usda`
and `scenes/materials/pbr.usda` (docs/PLAN.md Phase 2).

    python -m idtb.scenegen.build

A script, not a library import (README §4.2's "entry points are scripts;
libraries are pure") -- though nothing here actually needs `SimulationApp`,
since `usd-core` authors USD standalone. Run this after any change to
`materials.py`/`stage_v1.py` and commit the regenerated `.usda` files
alongside the code change, the same way a compiled artifact would be.
"""

from __future__ import annotations

from pathlib import Path

from idtb.scenegen.stage_v1 import write_stage_v1_usda

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENES_DIR = REPO_ROOT / "scenes"


def main() -> None:
    stage_path, materials_path = write_stage_v1_usda(SCENES_DIR)
    print(f"wrote {stage_path.relative_to(REPO_ROOT)}")
    print(f"wrote {materials_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
