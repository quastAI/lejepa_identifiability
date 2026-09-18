# Milestone 2: Scene v1 → First Dataset → First Real Number

> Execution checklist for the current milestone. `README.md` remains the plan of record — this file is the ordered task list for getting from a verified pipeline to a first defensible identifiability number. Checkpoints below say when to write results back into the README.

## Context

Milestone 1 is done. `README.md §9`'s Phases 0–3 are all closed: pure layers, infrastructure, Spikes 1–5 (physics-state writes and attribute writes both verified against the spike scene), and the backend seam — `SceneBackend` protocol, `MockSceneBackend`, `IsaacSceneBackend`, and the tier-1 contract suite, all green on the pod (`pytest -m isaac`: 30 passed, 2 skipped, 0 failed). Getting there took five pod-debugging rounds beyond what any isolated spike had verified, including a genuinely deep one: `sim.forward()` alone doesn't republish a physics write to a live camera, and a kinematic `RigidObjectCfg`'s tensor-API pose write permanently breaks that same prim's material rendering once combined with a real `sim.step()`. Both are fixed and documented in `README.md §5.5`/`§4.5`.

This milestone is README §9's critical path to a first defensible result: **Phase 2 (Scene v1) → Phase 2b (Spike 5 re-gate) → Phase 3 rework (backend against the real scene) → Phase 4 (determinism gate wired into `generate.py`) → Phase 5 (OU generator) → Phase 6 (first dataset) → Phase 7 (analysis, first real number).** Phases 8–10 (sweeps, scene v2, scene v3) are explicitly **Deferred** below — they are where the scientific contribution lives, but nothing in them is actionable before Phase 7 produces a number to compare against.

**Constraint unchanged:** dev machine is macOS, Isaac doesn't run there at all. All Isaac-facing code is written blind; everything touching a GPU runs on the pod. Pod updates are a `curl`+`tar` pull to `/idtb/repo` (no `.git` there) — see `README.md §8.3`.

**Carried forward from Milestone 1 — resolve these as part of Scene v1, not before it starts:**

1. **Re-test `light.azimuth_elevation` and `table.roughness` against scene v1's real light rig and material.** Both are confirmed *blocked* on the spike scene's single `DistantLight` + flat `PreviewSurfaceCfg` — writable, exact, bitwise-deterministic, but zero measurable pixel effect. Scene v1 replaces both with an HDRI dome, area lights, and real PBR materials (README §7.4/§7.5); re-run the same `run_knob_check` recipe before deciding either latent's fate for real.
2. **Sweep `write_latent_state()`'s zero-drift claim broadly.** The real `sim.step()` fix (README §5.5) measured zero `read_state()` drift only on the states the contract suite happens to write — well-separated corner values, never an interpenetrating or near-joint-limit configuration. Sample broadly, including deliberately interpenetrating arm-cube states, and assert `read_state()` before/after the step matches to the same `1×10⁻⁴` tolerance the read-back gate already uses. Do this before Phase 6 generates anything real.
3. **Check whether `cube.y`'s weak per-view signal persists once scene v1's multi-view rig exists.** Measured on the spike scene at roughly a third of `cube.x`'s magnitude, and observed dropping to exactly `0.0` under enough compounding attribute writes in one session (README §11). Expected to improve once a second, differently-angled camera is available; not yet root-caused, and not something to assume fixed without checking.
4. **Re-run the tier-1 contract suite against scene v1**, not just against the spike scene. README §7.4 already predicts `IsaacSceneBackend`/`writer.py`/`render.py` need rework once scene v1's PBR materials, HDRI/area lights, and multi-view camera rig replace the spike scene's shipped Franka + cuboid + `DistantLight`.

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

- [ ] 🤖 **Author `scenes/stage_v1_tabletop.usd`** — table, Franka, cube, as USD composition (not authored ad hoc inside `build_rig()` the way the spike scene is). Cube stays `AssetBaseCfg` (README §5.5) — the tensor-API/Fabric interaction that decision fixed is a property of the write path, not of this specific scene, and there is no reason to reopen it here.
- [ ] 🤖 **PBR materials** — table, arm, cube get roughness/metallic/normal maps in place of the spike's flat `PreviewSurfaceCfg`. Table roughness/albedo stay `style` handles against the *real* material this time (carried-forward item 1 above).
- [ ] 🤖 **HDRI dome + one or two area lights**, replacing the spike's single `DistantLight`. `light.intensity`/`light.warmth` need re-verifying against the new light type even though they were clean on the spike scene — a different light type is plausibly a different code path (README §7.5's own reasoning for why `light.azimuth_elevation` was carried forward rather than dropped).
- [ ] 🤖 **Camera rig: 2–3 views**, real parallax between them (README §5.3's multi-view occlusion mitigation). Placement should specifically avoid replicating whatever made `cube.y` weak on the spike scene's single camera — i.e., don't place a second view along the same axis as the first. High, oblique, near-top-down for at least one view, matching what already reduces arm-over-cube occlusion (README §5.3).
- [ ] 🤖 **Debug-preset renders look right** — visual smoke test before any determinism/sensitivity work, per README §12's closing note ("resist building the scene first" was about *not deciding physics/render policy against a scene that doesn't exist yet" — the scene itself still needs a first visual pass before anything else touches it).
- [ ] 🧑🤖 **Pod session**: load the new stage, confirm it boots and renders under `debug`.

## Phase 2b — Spike 5 re-gate against scene v1

> README §9: does each `full`/`style` knob still write, read back, move pixels, and stay bitwise deterministic under `standard`, against the real rig?

- [ ] 🧑🤖 Re-run `run_knob_check` (already written, reusable) for every `full`/`style` handle against scene v1's actual prims — not a rewrite of `spike_dynamic_attrs.py`, a re-target of it.
- [ ] 🤖 **Carried-forward items 1 and 3** (see Context) are resolved here: `light.azimuth_elevation`, `table.roughness`, and `cube.y`'s weak-signal question all get their real-rig verdict in this phase, not deferred further.
- [ ] 🤖 Update README §5.2's radii table and §3.2's Decision Register with whatever this phase measures — cube x/y radii in particular were always provisional pending "scene v1 has a table" (§5.2.1).

## Phase 3 rework — Backend against the real scene

> README §9 Phase 3 is *closed* against the spike scene. This is the re-verification against scene v1 that §7.4 already predicted would be needed, not new scope.

- [ ] 🤖 Update `src/idtb/sim/scene.py::build_rig()` to load `stage_v1_tabletop.usd` instead of authoring the spike scene inline.
- [ ] 🤖 Re-run the full tier-1 contract suite (`pytest -m isaac`) against it. Expect at least the light/material-path tests to need attention (new prim structure, possibly new shader resolution paths for `_resolve_bound_shader`).
- [ ] 🤖 Carried-forward item 2 (zero-drift sweep) lands here, as a permanent addition to the contract suite — not a one-off pod check.

## Phase 4 — Determinism gate in `generate.py`

> README §9: the §7.1 acceptance test passing for every preset intended for dataset use, called by `generate.py` itself, not only by the test suite.

- [ ] 🤖 `src/idtb/gen/generate.py` does not exist yet. First write: the generation loop sketched in README §6.4, calling `idtb.gates.determinism_gate` before any shard is written, exactly as `tests/contract/` already does.
- [ ] 🤖 Confirm `standard`'s bitwise determinism against scene v1 specifically — README §7.3 already flags that `standard`'s numbers were measured on the spike scene, and N (the `totalSpp` needed to converge) "almost certainly" changes with scene complexity even if the mechanism doesn't.

## Phase 5 — OU generator

> README §9: sharded writer storing (x, x′, z, z′, visibility, collision, ρ, seed, intrinsics); per-shard checkpointing.

- [ ] 🤖 Dataset storage format — open per README §3.2, decided here: HDF5 vs. WebDataset vs. other, once per-sample payload size and the training-side read pattern are actually known from Phase 2's real scene.
- [ ] 🤖 Per-shard checkpointing, so a spot-instance preemption mid-generation resumes rather than restarts (README §11 risk register).
- [ ] 🤖 Wire in the group-tag/style-probe-split machinery README §6.4 already specifies (the third "style dims resampled" frame for the invariance probe).

## Phase 6 — First dataset

> README §9: ~100k pairs at `standard`, ρ_task = 0.95 (a starting point, not a finding), **`base` group only, no style variation** — the clean control every later configuration is compared against.

- [ ] 🧑🤖 Generate, visually audit a random sample grid.
- [ ] 🤖 Re-measure the GPU-hours/100k-pairs cost against scene v1 — README §11 flags the spike-scene measurement (`~0.44` GPU-hours/100k at B=2, 128×128) as not transferable, and it now also costs a second render call per capture (the settle-render fix, README §5.5).

## Phase 7 — Analysis: first real number

> README §9: LeJEPA/SIGReg training; metrics: R²(h→z), R²(z→h), ‖Q̂ᵀQ̂−I‖_F/√n, ε, δ, bound D + (ε+D)².

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

**Done means:** scene v1 exists and every `full`/`style` knob against it either has a measured range and confirmed write, or is dropped with the reason recorded (mirroring Milestone 1's own "done means", now against the real rig instead of the spike scene); the tier-1 contract suite is green against scene v1, including the zero-drift sweep; `generate.py` exists, calls the determinism gate itself, and has produced a first `base`-only, 100k-pair dataset; a first R²(h→z) number exists and is written into README §10.2/§1 as the milestone's headline result.

---

## Deferred

- **README §9 Phases 8–10** — the group-matrix and ρ/λ/realism/resolution sweeps, scene v2 (room shell), scene v3 (additional manipulands). Not actionable before Phase 7 produces a number to compare against.
- The exact-zero mechanism behind `cube.y`'s weak-signal finding (README §11), if it turns out *not* to be resolved by scene v1's multi-view rig. Root-causing it further only if it's still live once Phase 2b's real-rig re-test lands.
- LeJEPA/SIGReg trainer implementation details beyond "runs against the mock, then against real data" — the authors' own code (`github.com/klindtlab/lejepa-identifiability`) is a live option once Phase 7 starts in earnest.
