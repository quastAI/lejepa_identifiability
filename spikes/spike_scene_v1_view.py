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
the render cost itself.

**Three failed attempts with `omni.replicator.core`, confirmed on the pod,
before switching approach entirely:**

1. Omitting `rep.orchestrator.set_capture_on_play(False)` left Replicator's
   default on-timeline-play capture trigger firing continuously once
   `open_stage()` auto-started the timeline -- 179k+ files in `--out`.
2. Adding that call was not sufficient: a `render_product`'s `hydra_texture`
   renders *continuously* the moment it's created, independent of
   Replicator's own frame-trigger machinery entirely -- 70k+ files even
   with `hydra_texture.set_updates_enabled()` gating applied around the
   `step()` call.
3. Still climbing after both fixes.

Rather than keep patching Replicator's writer/render-product/trigger
machinery, this now uses `omni.kit.viewport.utility.capture_viewport_to_file`
-- a synchronous, purpose-built one-shot capture (`helper.wait_for_result()`
resolves exactly once, no continuous-rendering or trigger-loop semantics to
fight) -- with the viewport's `camera_path` reassigned per capture instead
of spawning a `render_product` per camera at all.
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
    stage_path = repo_root / "scenes" / "stage_v1_tabletop.usda"
    if not stage_path.exists():
        raise FileNotFoundError(f"scene not found: {stage_path}")

    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"omni.usd failed to open {stage_path}")
    stage = context.get_stage()

    from idtb.scenegen.stage_v1 import CAMERA_PARENT_PATH

    # stage.Traverse() also finds Kit's own built-in viewport cameras
    # (OmniverseKit_Front/Persp/Right/Top) -- confirmed on the pod: without
    # this filter, all 7 got captured, not just the 3 authored ones.
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
        print(f"[spike_scene_v1_view] could not set viewport resolution ({exc}); using default")

    for _ in range(args.settle_frames):  # let the freshly opened stage render in
        simulation_app.update()

    with tempfile.TemporaryDirectory(prefix="spike_scene_v1_view_") as scratch:
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

            print(f"[spike_scene_v1_view] captured {name}")

        args.out.mkdir(parents=True, exist_ok=True)
        # Only the finished .png outputs -- capture_viewport_to_file leaves
        # its own internal staging file (".cap-XXXXXX") in the scratch dir,
        # which it cleans up itself; copying it unconditionally raced that
        # cleanup and crashed with FileNotFoundError on the pod.
        for png in scratch_dir.glob("*.png"):
            shutil.copy2(png, args.out / png.name)
        print(f"[spike_scene_v1_view] copied results to {args.out}")

    print(f"[spike_scene_v1_view] done -- {len(camera_paths)} camera(s) rendered to {args.out}")


if __name__ == "__main__":
    main()
