# Milestone 2: Scene v1 → Scene v2 (room) → First Dataset → First Real Number

> Execution checklist for the current milestone. `README.md` remains the plan of record — this file is the ordered task list for getting from a verified pipeline to a first defensible identifiability number. Checkpoints below say when to write results back into the README.

## Context

Milestone 1 is done. `README.md §9`'s Phases 0–3 are all closed: pure layers, infrastructure, Spikes 1–5 (physics-state writes and attribute writes both verified against the spike scene), and the backend seam — `SceneBackend` protocol, `MockSceneBackend`, `IsaacSceneBackend`, and the tier-1 contract suite, all green on the pod (`pytest -m isaac`: 30 passed, 2 skipped, 0 failed). Getting there took five pod-debugging rounds beyond what any isolated spike had verified, including a genuinely deep one: `sim.forward()` alone doesn't republish a physics write to a live camera, and a kinematic `RigidObjectCfg`'s tensor-API pose write permanently breaks that same prim's material rendering once combined with a real `sim.step()`. Both are fixed and documented in `README.md §5.5`/`§4.5`.

This milestone is README §9's critical path to a first defensible result: **Phase 2 (Scene v1) → Phase 2b (Spike 5 re-gate) → Phase 2c (Scene v2: room shell) → Phase 3 rework (backend against the real scene) → Phase 4 (determinism gate wired into `generate.py`) → Phase 5 (OU generator) → Phase 6 (first dataset) → Phase 7 (analysis, first real number).**

**Reordered, 2026-09-21: Scene v2's room shell moved from Phase 9 (deferred) to Phase 2c (critical path), ahead of Phase 6.** The first real dataset is meant to be the actual target scene — table, Franka, cube, in a room — not a room-less intermediate that would need regenerating once the room lands anyway. Phases 8 and 10 (sweeps, scene v3) stay **Deferred** below — they are where the scientific contribution lives, but nothing in them is actionable before Phase 7 produces a number to compare against.

**Constraint unchanged:** dev machine is macOS, Isaac doesn't run there at all. All Isaac-facing code is written blind; everything touching a GPU runs on the pod. Pod updates are a `curl`+`tar` pull to `/idtb/repo` (no `.git` there) — see `README.md §8.3`.

**Carried forward from Milestone 1 — resolve these as part of Scene v1, not before it starts:**

1. **Re-test `light.azimuth_elevation` and `table.roughness` against scene v1's real light rig and material.** Both are confirmed *blocked* on the spike scene's single `DistantLight` + flat `PreviewSurfaceCfg` — writable, exact, bitwise-deterministic, but zero measurable pixel effect. Scene v1 replaces both with an HDRI dome, area lights, and real PBR materials (README §7.4/§7.5); re-run the same `run_knob_check` recipe before deciding either latent's fate for real.
2. **Sweep `write_latent_state()`'s zero-drift claim broadly.** The real `sim.step()` fix (README §5.5) measured zero `read_state()` drift only on the states the contract suite happens to write — well-separated corner values, never an interpenetrating or near-joint-limit configuration. Sample broadly, including deliberately interpenetrating arm-cube states, and assert `read_state()` before/after the step matches to the same `1×10⁻⁴` tolerance the read-back gate already uses. Do this before Phase 6 generates anything real.
3. **Check whether `cube.y`'s weak per-view signal persists once scene v1's multi-view rig exists.** Measured on the spike scene at roughly a third of `cube.x`'s magnitude, and observed dropping to exactly `0.0` under enough compounding attribute writes in one session (README §11). Expected to improve once a second, differently-angled camera is available; not yet root-caused, and not something to assume fixed without checking.
4. **Re-run the tier-1 contract suite against scene v1**, not just against the spike scene. README §7.4 already predicts `IsaacSceneBackend`/`writer.py`/`render.py` need rework once scene v1's PBR materials, HDRI/area lights, and multi-view camera rig replace the spike scene's shipped Franka + cuboid + `DistantLight`.
5. **Root-cause `Camera3`'s near-black render**, found in Phase 2's own pod session (above). Two live hypotheses, not yet distinguished: a genuine top-down lighting/exposure gap (the `KeyLight`/`FillLight` rig is positioned obliquely, not overhead), or an artifact specific to `spike_scene_v1_view.py`'s viewport-reassignment capture method rather than the scene itself. Phase 3 rework's real `Camera` sensors are the trustworthy test either way — resolve there rather than sinking further time into the disposable smoke-test script.

---

## Legend

| | Who |
|---|---|
| 🧑 | **Julian** — pod console, billing, running commands on the pod, visual judgment calls |
| 🤖 | **Claude** — all code, tests, scripts, README updates |
| 🧑🤖 | Julian runs it, pastes the output, Claude acts on it |

---

## Phase 2 — Scene v1

> README §9: `stage_v1_tabletop.usd` — table, Franka, cube, PBR materials, HDRI + area lights, camera rig (2–3 views). Debug-preset renders look right.

- [x] 🤖 **Read the docs first** — done via a research pass over primary sources (openusd.org, NVIDIA Omniverse/Isaac docs, `isaac-sim/IsaacLab` source and issues). Findings baked into the items below; the one that changes Phase 3 rework's plan: **no `RectLightCfg` exists in Isaac Lab's own configclass set** (confirmed absent from `lights_cfg.py`) — area lights are authored directly via `pxr.UsdLux.RectLight`, a real schema Isaac Lab just has no Python wrapper for. And **prefer plain `Camera` sensors over `TiledCamera` for the multi-view rig**: `TiledCamera` has a documented history of cross-talk between multiple simultaneous cameras ([isaac-sim/IsaacLab#1070](https://github.com/isaac-sim/IsaacLab/issues/1070) — outputs overwriting each other, and even after partial fixes, ostensibly-different tiled images that were "the same image just with different noise levels"); it's designed to replicate *one* camera across parallel envs, not hold multiple distinct viewpoints in one env. Unverified from primary docs and worth checking empirically on the pod: whether Isaac Lab 3.0's `TiledCamera` fixes changed this since #1070.
- [x] 🤖 **Author `scenes/stage_v1_tabletop.usd`** — table, cube, materials, lights, camera rig, as a real USD composition built by `src/idtb/scenegen/` (new package, `python -m idtb.scenegen.build` regenerates the checked-in files). **Scope correction from this bullet's original wording: the Franka is *not* baked into this file.** Its Nucleus asset root is resolved dynamically by `isaaclab_assets.FRANKA_PANDA_CFG` at runtime (`scene.py::_resolve_franka_cfg`), never a fixed literal — baking a reference to it into a checked-in `.usda` would pin exactly the kind of environment-specific path the Decision Register avoids pinning everywhere else. Phase 3 rework keeps spawning the robot as an `ArticulationCfg` alongside this stage, not inside it. Also replaces the spike scene's table, which was never actually load-bearing (the cube always rested on the ground plane, at a position that didn't even overlap the table's footprint) — scene v1's table is real: 0.8×0.6 m top at 0.40 m, cube resting directly on its surface, table centered under the cube by construction.
  - **Found and fixed by actually running this locally** (usd-core, not blind): `Gf.Matrix4d.SetLookAt` with world-up `(0,0,1)` is degenerate for a near-vertical view direction — Camera3's authored transform came out as literal `FLT_MAX` entries the first time this ran, because its top-down eye/target pair makes the view direction parallel to that up vector. Fixed with an up-vector fallback to `(0,1,0)` when the view direction is within 0.99 of parallel to world-up; regression test added (`tests/test_scenegen.py::test_camera_transforms_are_finite_even_for_top_down_views`).
  - **New capability, worth noting for future Isaac/USD-surface work**: the standalone `usd-core` PyPI package (real Pixar USD, no Omniverse Kit/`SimulationApp`) installs and runs on macOS. Scene *authoring and structural verification* (prim hierarchy, material bindings, light/camera params) is no longer "written blind" the way `idtb.sim` is — it's a real local test loop (`tests/test_scenegen.py`, 11 tests). What is still genuinely blind: whether the RTX path tracer actually *renders* this correctly, and whether `usd-core`'s OpenUSD build opens bit-for-bit compatibly in Isaac Sim 6.0.1's bundled `pxr` — both are exactly what the pod session below checks.
- [x] 🤖 **PBR materials** — `UsdPreviewSurface` (not MDL/OmniPBR: confirmed by NVIDIA's RTX Renderer docs that both are renderer-supported, and `UsdPreviewSurface` is the schema `table.albedo`/`cube.hue` already write through via `diffuseColor`, so `_resolve_bound_shader` needs a new prim structure to search in Phase 3 rework, not a new shader kind to handle). Table: roughness 0.75, non-metallic (matte worktable). Cube: roughness 0.4, non-metallic (semi-gloss, keeps `cube.hue` legible without a distracting specular hotspot under `PathTracing`'s point-sampled area lights). Table roughness/albedo still resolve as `style` handles against this *real* material in Phase 2b (carried-forward item 1).
- [x] 🤖 **HDRI dome + two area lights**, replacing the spike's single `DistantLight`. Dome light is authored **textureless by default** (`inputs:color`, a neutral indoor-bounce tone — this is a tabletop scene with no room shell yet, so a cool sky-blue tint would misrepresent it as outdoor; caught and fixed after being flagged) rather than pinning a guessed Nucleus HDRI asset path — same "no premature pins" reasoning as the Franka decision above; `build_stage_v1(dome_texture_file=...)` accepts a real path once one is resolved and confirmed on the pod. Two `UsdLux.RectLight`s: `KeyLight` (the `light.intensity`/`light.warmth` target going forward — needs re-verifying against this light type per carried-forward item 1, since it's a different USD type than the spike's `DistantLight`) and a fixed, non-latent `FillLight` for scene depth.
- [x] 🤖 **Camera rig: 3 views**, real parallax. `Camera1` matches the spike's own placement (continuity with what §5.3/§7.4 already measured). `Camera2` sits at a distinctly different azimuth (mostly along +y) specifically targeting the weak `cube.y` signal — chosen, not just offset along `Camera1`'s axis (verified by `test_camera_rig_has_real_multi_axis_parallax`, which would fail if a future edit collapsed the rig back to one axis). `Camera3` is near-top-down, both for comparable x/y sensitivity and as README §5.3's strongest independent occlusion mitigation.
- [x] 🤖 **Debug-preset renders look right** — done, 2026-09-21, via a new one-shot smoke-test script (`spikes/spike_scene_v1_view.py`) written for this check, since none existed. Getting it working took four rounds against `omni.replicator.core`'s render-product/writer/trigger machinery, each a genuine runaway-capture bug, not a guess: (1) omitting `rep.orchestrator.set_capture_on_play(False)` left the writer capturing on every timeline tick (179k+ files); (2) a render product's `hydra_texture` renders continuously the instant it exists, independent of that trigger system entirely, and needs explicit `set_updates_enabled()` gating (still ran away, 70k+ files); (3) switched approach entirely to `omni.kit.viewport.utility.capture_viewport_to_file` — a synchronous one-shot capture with no continuous-rendering semantics to fight; (4) `wait_for_result()` resolves once a capture is *requested*, not once its file is finalized on disk, which silently dropped exactly the *last* camera's PNG every run until a settle-frame pump was added after every capture, not just before switching cameras. All four are documented in the script's own module docstring so the lesson isn't lost.
- [x] 🧑🤖 **Pod session**: pulled the updated repo, loaded `scenes/stage_v1_tabletop.usda` under `debug`, confirmed it opens without error. `usd-core`'s authored `.usda` does open correctly in Isaac Sim 6.0.1's bundled `pxr` — the thing this phase could only guess at locally. **Renders sensibly from `Camera1`/`Camera2`**: table + cube visible, reasonably lit, textureless dome's neutral tone looks fine (not blown out, not pitch black). **`Camera3` (near-top-down) does not** — renders almost entirely black. Not yet root-caused: could be a genuine lighting/exposure gap from directly overhead (the two `RectLight`s are positioned obliquely), or an artifact specific to this script's viewport-reassignment capture method rather than the scene itself (Phase 3 rework's real `Camera` sensors are a structurally different code path). Carried forward to Phase 2b/3 rework rather than debugged further now — see Context below.

## Phase 2b — Spike 5 re-gate against scene v1

> README §9: does each `full`/`style` knob still write, read back, move pixels, and stay bitwise deterministic under `standard`, against the real rig?

- [x] 🤖 **Read the docs first if a knob misbehaves against the new rig** — not needed as a separate step; the two real bugs found (below) were both root-caused directly from pod evidence (`cube_prim_tree`/`table_prim_tree` diagnostics, exact-number read-backs) without needing external docs.
- [x] 🤖 **`spike_dynamic_attrs.py` retargeted at scene v1** — `build_rig()` references `scenes/stage_v1_tabletop.usda` via `sim_utils.UsdFileCfg` at `SCENE_ROOT = "/World/SceneV1"`.
- [x] 🧑🤖 **Pod run — done, 2026-09-21, 18 passed / 2 failed.** Two real bugs found and fixed along the way (both documented in README §7.6, one of them structural enough to have hit Phase 3 rework identically): `write_cube_scale` wrote the scale multiplier as if it were the absolute size (a `0.09` m request silently became `1.5` m), and `idtb.scenegen.materials`' `/Looks` scope sat outside `stage_v1.py`'s default prim (`/World`), so a `UsdFileCfg` reference never pulled the materials in at all ("shader was never resolved" on every shader-bound knob) — fixed by moving materials to `/World/Looks`. Of the 2 remaining FAILs: `motion_blur_under_teleport_no_step` is expected by design (no velocity between teleports, exactly as intended); `camera_jitter_per_capture` has a small, new, not-yet-root-caused residual non-determinism (`mad` 4.07×10⁻⁵, ≈2 pixel-units) — tracked in README §11, not blocking.
- [x] 🤖 **Carried-forward item 1 resolved**: `table.roughness` is no longer a dead knob against scene v1's real PBR material (`mad_vs_base` 1.86) — the old scene's exactly-zero effect was scene-specific, not structural. `light.azimuth_elevation` is no longer *exactly* zero either, but at ~2,500–120,000× weaker than every other knob it's still not practically usable as rigged (full numbers in README §7.6). **Item 3 (`cube.y`'s weak-signal question) stays out of scope for this phase**, as already noted — needs Phase 3 rework's 3-camera rig.
- [x] 🤖 Updated README §3.1/§3.2/§5.2.3/§7 (new §7.6)/§11 with this phase's measurements.
- [ ] 🤖 **Cube x/y squash radii (§5.2.1) remain provisional, deliberately not done here** — measuring them against real Franka reach needs the arm actually spawned, which is Phase 3 rework's job, not this phase's. Tracked there, not blocking Phase 2b's close-out.

## Phase 2c — Scene v2: room shell

> README §9, reordered 2026-09-21: an empty room shell around stage v1, moved onto the critical path ahead of Phase 6 rather than deferred as an ablation — the first real dataset should be the actual target scene (table + Franka + cube, in a room), not a room-less intermediate that gets regenerated once the room lands anyway. Authoring done, 2026-09-21; pod-session visual check still open.

- [x] 🤖 **Read the docs first** — no stock Isaac Sim room/interior asset is worth referencing here (would reintroduce exactly the unresolved-Nucleus-path problem the Franka and the dome already avoid); a handful of large axis-aligned boxes is genuinely sufficient USD for an "empty room" shell, consistent with how the table/cube were authored.
- [x] 🤖 **Decided the room's scope** (README §3.1): 5×5 m floor footprint, 3 m ceiling, centered on `CUBE_XY` (not the world origin — the table's own centering point is what the room needs to contain), sized to clear all three camera eyes with ≥0.5 m real margin (`tests/test_scenegen.py::test_room_encloses_every_camera_eye_with_margin`). Simple matte `UsdPreviewSurface` for floor/ceiling/walls (`materials.py::ROOM_PBR`), not load-bearing for anything. **`stage_v2_room.usda` *references* `stage_v1_tabletop.usda`** (README's original Phase 9 language, not an extension of `stage_v1.py`'s own build function) — same idiomatic reference arc already used for the Franka, keeps `stage_v1_tabletop.usda` exactly what Phase 2 shipped and tested either way.
- [x] 🤖 **Decided the dome light's fate**: stays textureless, unchanged, deferred (README §3.1/§3.2) — no resolved HDRI asset path exists yet, and the room's existence doesn't change that. Whether it needs one is exactly what the pod session below and the knob re-check are for, not assumed either way.
- [x] 🤖 **Authored the room via `idtb.scenegen`** — new `idtb/scenegen/stage_v2_room.py` (+ a small shared `idtb/scenegen/primitives.py`, factored out of `stage_v1.py`, for the box-prim/material-bind helpers both files now use); `build.py` regenerates `scenes/stage_v2_room.usda` alongside the existing two files. Structural tests via `usd-core` (`tests/test_scenegen.py`, 6 new tests) confirm the room's prim hierarchy, material bindings, the camera-margin property above, and — the exact regression class README §7.6 already hit once for `UsdFileCfg` — that the reference actually composes stage v1's table/cube/cameras through to a real reopened file, not just in-memory. All local tests + ruff clean.
- [ ] 🧑🤖 **Pod session**: debug-preset render check, same recipe as Phase 2's (`spikes/spike_scene_v2_view.py`, duplicated from `spike_scene_v1_view.py` for the room-inclusive stage — written, not yet run) — confirm the room doesn't occlude any of the 3 camera views and the lighting looks reasonable from all of them, `Camera3` included (still an open finding from Phase 2, see Context item 5 — resolve it here or in Phase 3 rework, not left unexplained).
      ```
      ./isaaclab.sh -p spikes/spike_scene_v2_view.py --out /idtb/data/scene_v2_view
      ```
- [ ] 🤖 **Re-check Phase 2b's knob verdicts against the room's lighting**, at least for `table.albedo`/`table.roughness`/`light.azimuth_elevation` — a room's walls/ceiling give the dome light real surfaces to bounce off, which could plausibly change what was measured against scene v1 alone (§7.6). Not assumed to change; not assumed to stay the same either.
- [ ] 🤖 Update README §9's Phase 2c row from "authored, pod pending" to "built" once the pod session above lands, and close out this phase's row in the table.

## Phase 3 rework — Backend against the real scene

> README §9 Phase 3 is *closed* against the spike scene. This is the re-verification against scene v1 (now scene v2, room included — Phase 2c) that §7.4 already predicted would be needed, not new scope.

- [ ] 🤖 **Read the docs first** — `_resolve_bound_shader` was written against the spike scene's single flat material; before patching it for failures, check Isaac Sim's material-binding/shader-graph docs for how multi-material USD references (table/arm/cube each with their own PBR graph) expose bound shaders, so the fix generalizes instead of chasing this one scene's structure.
- [ ] 🤖 Update `src/idtb/sim/scene.py::build_rig()` to load the Phase 2c room-inclusive stage (not `stage_v1_tabletop.usd` alone, once Phase 2c lands) instead of authoring the spike scene inline, and keep spawning the Franka as an `ArticulationCfg` alongside it (Phase 2's authored file deliberately excludes the robot — see that phase's notes).
- [ ] 🤖 Wire up the 3-camera rig with **plain `Camera` sensors, one per view, not `TiledCamera`** — Phase 2's docs research found a documented history of `TiledCamera` cross-talk between multiple simultaneous cameras in one env ([isaac-sim/IsaacLab#1070](https://github.com/isaac-sim/IsaacLab/issues/1070)); it's designed to replicate one camera across parallel envs, not hold several distinct viewpoints in one. Check all three `CameraCfg.OffsetCfg.convention`s match — the same issue thread found a camera-orientation flip traced to a `"opengl"`/`"ros"` mismatch between cameras.
- [ ] 🤖 Re-run the full tier-1 contract suite (`pytest -m isaac`) against it. Expect at least the light/material-path tests to need attention (new prim structure, possibly new shader resolution paths for `_resolve_bound_shader`).
- [ ] 🤖 Carried-forward item 2 (zero-drift sweep) lands here, as a permanent addition to the contract suite — not a one-off pod check.
- [ ] 🤖 Carried-forward item 5: confirm `Camera3` (near-top-down) actually renders sensibly once it's a real `Camera` sensor here, not the viewport-reassignment hack `spike_scene_v1_view.py` used. If it's still near-black, that's a real lighting/exposure gap in the rig, not a script artifact, and needs fixing before Phase 6.
- [ ] 🤖 Measure cube x/y squash radii (README §5.2.1, still provisional) against real Franka reach now that the arm is actually spawned — carried forward from Phase 2b, which couldn't measure this without the arm.

## Phase 4 — Determinism gate in `generate.py`

> README §9: the §7.1 acceptance test passing for every preset intended for dataset use, called by `generate.py` itself, not only by the test suite.

- [ ] 🤖 `src/idtb/gen/generate.py` does not exist yet. First write: the generation loop sketched in README §6.4, calling `idtb.gates.determinism_gate` before any shard is written, exactly as `tests/contract/` already does.
- [ ] 🤖 Confirm `standard`'s bitwise determinism against the real scene specifically (scene v1 + Phase 2c's room) — README §7.3 already flags that `standard`'s numbers were measured on the spike scene, and N (the `totalSpp` needed to converge) "almost certainly" changes with scene complexity even if the mechanism doesn't; a room adds more geometry/lighting complexity again on top of scene v1 alone.
- [ ] 🤖 **Read the docs first if N needs retuning** — Isaac Lab/RTX render-settings docs on path-tracing SPP accumulation and denoiser convergence, before guessing a new N by trial-and-error render sweeps on the pod.

## Phase 5 — OU generator

> README §9: sharded writer storing (x, x′, z, z′, visibility, collision, ρ, seed, intrinsics); per-shard checkpointing.

- [ ] 🤖 **Read the docs first** — HDF5 (`h5py`, chunking/parallel-write docs) vs. WebDataset (sharding/streaming docs) before deciding: the tradeoffs (random access vs. sequential streaming, concurrent shard writers for spot-preemption checkpointing) are documented by each project and shouldn't be re-derived from scratch.
- [ ] 🤖 Dataset storage format — open per README §3.2, decided here: HDF5 vs. WebDataset vs. other, once per-sample payload size and the training-side read pattern are actually known from Phase 2's real scene.
- [ ] 🤖 Per-shard checkpointing, so a spot-instance preemption mid-generation resumes rather than restarts (README §11 risk register).
- [ ] 🤖 Wire in the group-tag/style-probe-split machinery README §6.4 already specifies (the third "style dims resampled" frame for the invariance probe).

## Phase 6 — First dataset

> README §9: ~100k pairs at `standard`, ρ_task = 0.95 (a starting point, not a finding), **`base` group only, no style variation** — the clean control every later configuration is compared against.

- [ ] 🧑🤖 Generate, visually audit a random sample grid.
- [ ] 🤖 Re-measure the GPU-hours/100k-pairs cost against the real scene (scene v1 + room) — README §11 flags the spike-scene measurement (`~0.44` GPU-hours/100k at B=2, 128×128) as not transferable, and it now also costs a second render call per capture (the settle-render fix, README §5.5), plus whatever the room adds on top.

## Phase 7 — Analysis: first real number

> README §9: LeJEPA/SIGReg training; metrics: R²(h→z), R²(z→h), ‖Q̂ᵀQ̂−I‖_F/√n, ε, δ, bound D + (ε+D)².

- [ ] 🤖 **Read the docs/paper first** — the LeJEPA/SIGReg paper's training recipe and metric definitions, and the authors' reference implementation (`github.com/klindtlab/lejepa-identifiability`), before writing training/metrics code from memory. Getting the SIGReg regularizer or the R²(h→z)/R²(z→h) definitions subtly wrong invalidates the milestone's headline number.
- [ ] 🤖 Training/metrics code against `MockSceneBackend`-generated data first (no GPU needed for this layer at all — README §8.4).
- [ ] 🤖 Run against Phase 6's real dataset. This is the milestone's actual deliverable: a first measured R² on photorealistic observations, comparable to the paper's Tables 1–2.

---

## Verification

**Locally (macOS):**
```
pytest          # everything except tier 2 / isaac-marked contract tests
ruff check .
```

**Pod:**
```
pytest -m isaac                          # tier-1 contract suite against scene v1
./isaaclab.sh -p src/idtb/gen/generate.py --group base --n 100000   # Phase 6
```

**Done means:** scene v1 and its room (scene v2) both exist, and every `full`/`style` knob against them either has a measured range and confirmed write, or is dropped with the reason recorded (mirroring Milestone 1's own "done means", now against the real rig instead of the spike scene); the tier-1 contract suite is green against the room-inclusive scene, including the zero-drift sweep; `generate.py` exists, calls the determinism gate itself, and has produced a first `base`-only, 100k-pair dataset in that scene; a first R²(h→z) number exists and is written into README §10.2/§1 as the milestone's headline result.

---

## Deferred

- **README §9 Phases 8 and 10** — the group-matrix and ρ/λ/realism/resolution sweeps, scene v3 (additional manipulands). Not actionable before Phase 7 produces a number to compare against. (Scene v2's room shell is no longer deferred — moved to Phase 2c, on the critical path.)
- The exact-zero mechanism behind `cube.y`'s weak-signal finding (README §11), if it turns out *not* to be resolved by scene v1's multi-view rig. Root-causing it further only if it's still live once Phase 2b's real-rig re-test lands.
- LeJEPA/SIGReg trainer implementation details beyond "runs against the mock, then against real data" — the authors' own code (`github.com/klindtlab/lejepa-identifiability`) is a live option once Phase 7 starts in earnest.
