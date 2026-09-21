"""Phase 2c pod-session smoke test (docs/PLAN.md): opens
`scenes/stage_v2_room.usda` under Isaac Sim and renders one frame from each
of stage_v1's three authored camera prims (`idtb.scenegen.stage_v1.CAMERAS`,
composed unchanged into stage_v2 -- see `stage_v2_room.py`'s module
docstring) under the `debug` preset -- same as `spike_scene_v1_view.py`,
now checking that the room shell doesn't occlude any view and that the
still-textureless dome light still looks reasonable once there are walls
and a ceiling for it to (not) bounce off.

    ./isaaclab.sh -p spikes/spike_scene_v2_view.py --out /idtb/data/scene_v2_view

Duplicated from `spike_scene_v1_view.py` rather than parameterized over
both scenes: the two checks are visual judgement calls made once each, at
different phases, not a recurring automated check that would earn an
abstraction -- see that script's own docstring for the four rounds of
`omni.replicator.core` failures this capture approach already worked
around, which apply identically here since nothing about the capture
method changes, only which stage file is opened.

Not a pytest target -- like `spike_scene_v1_view.py`, this is a one-shot
check docs/PLAN.md assigns to a human (Julian) to look at.
"""

from __future__ import annotations

import argparse
import asyncio
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
        "--settle-frames",
        type=int,
        default=30,
        help="app updates to pump after switching camera, before capturing (lets the "
        "path tracer converge on the new view)",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from idtb.sim.app import launch

    simulation_app = launch(args)  # must precede every isaaclab/omni/pxr import below

    import omni.usd
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    repo_root = Path(__file__).resolve().parents[1]
    stage_path = repo_root / "scenes" / "stage_v2_room.usda"
    if not stage_path.exists():
        raise FileNotFoundError(f"scene not found: {stage_path}")

    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"omni.usd failed to open {stage_path}")
    stage = context.get_stage()

    from idtb.scenegen.stage_v1 import CAMERA_PARENT_PATH

    # stage.Traverse() also finds Kit's own built-in viewport cameras
    # (OmniverseKit_Front/Persp/Right/Top) -- see spike_scene_v1_view.py.
    camera_prefix = CAMERA_PARENT_PATH + "/"
    camera_paths = sorted(
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetTypeName() == "Camera" and str(prim.GetPath()).startswith(camera_prefix)
    )
    if not camera_paths:
        raise RuntimeError(f"no Camera prims found under {CAMERA_PARENT_PATH} in {stage_path}")

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("no active viewport -- headless Kit did not create one")

    width, height = args.resolution
    try:
        viewport.resolution = (width, height)
    except Exception as exc:  # not load-bearing -- default viewport size still captures
        print(f"[spike_scene_v2_view] could not set viewport resolution ({exc}); using default")

    for _ in range(args.settle_frames):  # let the freshly opened stage render in
        simulation_app.update()

    with tempfile.TemporaryDirectory(prefix="spike_scene_v2_view_") as scratch:
        scratch_dir = Path(scratch)
        for camera_path in camera_paths:
            name = camera_path.rsplit("/", 1)[-1]
            viewport.camera_path = camera_path
            for _ in range(args.settle_frames):
                simulation_app.update()  # let the new camera's view actually render

            async def _capture(file_path: Path = scratch_dir / f"{name}.png") -> None:
                helper = capture_viewport_to_file(viewport, file_path=str(file_path))
                await helper.wait_for_result()

            task = asyncio.ensure_future(_capture())
            while not task.done():
                simulation_app.update()
            task.result()  # re-raises if _capture() failed

            # wait_for_result() resolves once the capture is *requested*, not
            # once its file is finalized on disk -- see spike_scene_v1_view.py.
            for _ in range(args.settle_frames):
                simulation_app.update()

            print(f"[spike_scene_v2_view] captured {name}")

        args.out.mkdir(parents=True, exist_ok=True)
        for png in scratch_dir.glob("*.png"):
            shutil.copy2(png, args.out / png.name)
        print(f"[spike_scene_v2_view] copied results to {args.out}")

    print(f"[spike_scene_v2_view] done -- {len(camera_paths)} camera(s) rendered to {args.out}")


if __name__ == "__main__":
    main()
