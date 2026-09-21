"""Phase 2 pod-session smoke test (docs/PLAN.md): opens
`scenes/stage_v1_tabletop.usda` under Isaac Sim and renders one frame from
each of its three authored camera prims (`idtb.scenegen.stage_v1.CAMERAS`)
under the `debug` preset -- README §7.3's as-booted `RealTimePathTracing`
default, no `carb` changes: determinism is not required for this check,
only "does it look right".

    ./isaaclab.sh -p spikes/spike_scene_v1_view.py --out /idtb/data/scene_v1_view

Not a pytest target -- like `spike_api.py`/`spike_dynamic_attrs.py`, this is
a one-shot check docs/PLAN.md assigns to a human (Julian) to look at, not
something this script can assert pass/fail on its own: "renders sensibly"
is a visual judgement call.

Deliberately does not go through `idtb.sim.scene.build_rig()` /
`InteractiveScene` -- this checks the *authored* stage and its *authored*
camera prims exactly as `idtb.scenegen` wrote them, independent of
Phase 3 rework's not-yet-built `IsaacSceneBackend` rewiring (which will
spawn its own `CameraCfg` sensors, per docs/PLAN.md Phase 3 rework). Uses
`omni.replicator.core` to render from each existing camera prim directly,
since Isaac Lab's `Camera`/`CameraCfg` sensor always spawns a fresh camera
prim from a spawn config rather than wrapping one that already exists.

Renders to a local scratch directory first and copies the result to `--out`
only once everything is captured: `--out` is typically on the network
volume (`/idtb/data/...`), and per-frame writes there were measured
throttled by I/O ("Throttling generation due to I/O bottleneck") on top of
the render cost itself. `--rt-subframes` defaults low (4) rather than a
quality-grade accumulation count -- `debug` (README §7.3) only needs a
recognisable image, not the `standard` preset's determinism, so there is no
reason to pay for path-tracing accumulation depth here.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="final output directory for the rendered PNGs"
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=(512, 512), metavar=("WIDTH", "HEIGHT")
    )
    parser.add_argument(
        "--rt-subframes",
        type=int,
        default=4,
        help="path-tracing accumulation subframes per capture (debug preset: low is fine)",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from idtb.sim.app import launch

    launch(args)  # boots SimulationApp -- must precede every isaaclab/omni/pxr import below

    import omni.replicator.core as rep
    import omni.usd

    repo_root = Path(__file__).resolve().parents[1]
    stage_path = repo_root / "scenes" / "stage_v1_tabletop.usda"
    if not stage_path.exists():
        raise FileNotFoundError(f"scene not found: {stage_path}")

    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"omni.usd failed to open {stage_path}")
    stage = context.get_stage()

    camera_paths = sorted(
        str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "Camera"
    )
    if not camera_paths:
        raise RuntimeError(f"no Camera prims found in {stage_path}")

    width, height = args.resolution

    with tempfile.TemporaryDirectory(prefix="spike_scene_v1_view_") as scratch:
        scratch_dir = Path(scratch)
        for camera_path in camera_paths:
            name = camera_path.rsplit("/", 1)[-1]
            render_product = rep.create.render_product(camera_path, (width, height))
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=str(scratch_dir / name), rgb=True)
            writer.attach([render_product])
            rep.orchestrator.step(rt_subframes=args.rt_subframes)
            writer.detach()
            render_product.destroy()
            print(f"[spike_scene_v1_view] rendered {name} to local scratch")

        args.out.mkdir(parents=True, exist_ok=True)
        for name_dir in scratch_dir.iterdir():
            shutil.copytree(name_dir, args.out / name_dir.name, dirs_exist_ok=True)
        print(f"[spike_scene_v1_view] copied results to {args.out}")

    print(f"[spike_scene_v1_view] done -- {len(camera_paths)} camera(s) rendered to {args.out}")


if __name__ == "__main__":
    main()
