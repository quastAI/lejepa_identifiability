# Milestone 1: Isaac API green + pure layers

> Execution checklist for the current milestone. `README.md` remains the plan of record — this file is the ordered task list for getting to first verified Isaac contact. Checkpoints below say when to write results back into the README.

## Context

Starting point: the repo had **no source code** — only README.md, LICENSE, and the paper PDF. Phase 1 is built and Phase 2's version question is resolved; what remains of Phase 2 is a pod session, and Phase 3 onwards is open.

Every Isaac API signature in README §4.4 came from reading docs, **never from running anything**. So: meet the Isaac API first in one spike script, then design the protocol against verified reality. The README's current order (protocol + mock first, Isaac at Phase 3) would mean designing the central abstraction against guesses.

**Scope:** stops at "Isaac API verified on the pod + pure layers green locally + protocol and mock built." ~3–4 days.

**Constraint:** dev machine is macOS. Isaac doesn't run on macOS at all. All Isaac-facing code is written blind; **everything touching a GPU runs on the pod.**

---

## Legend

| | Who |
|---|---|
| 🧑 | **Julian** — pod console, billing, running commands on the pod, visual judgment calls |
| 🤖 | **Claude** — all code, tests, scripts, README updates |
| 🧑🤖 | Julian runs it, pastes the output, Claude acts on it |

---

## Phase 1 — Local foundations (no GPU, start now) — ✅ done

- [x] 🤖 **Scaffolding** — `pyproject.toml` (hatchling, src layout), pytest config with the `isaac` marker deselected by default, ruff, `src/idtb/` tree
  - Single package `idtb`, not top-level `sim`/`eval`/`gen` — those collide with Kit extensions on Isaac's `sys.path`, and `eval` shadows a builtin
  - Isaac packages must never be pip-installable, so a local install can't misrepresent what's runnable
  - No Python version pin anywhere: it follows the Isaac Sim release, resolved in Phase 2
- [x] 🤖 **Import guard** — ruff TID253 bans module-level `isaaclab`/`omni`/`carb`/`pxr`/`isaacsim`; `tests/test_import_guard.py` imports every module in a subprocess with those roots blocked by a meta-path finder, and asserts the blocker itself fires first
  - Guards a [known unresolved upstream bug](https://github.com/isaac-sim/IsaacLab/issues/3239): pytest collection imports Isaac before `SimulationApp` starts. A violation is otherwise only discovered on the pod, costing a container restart.
- [x] 🤖 **OU sampler + tests** — `idtb.latents.sample_ou_pairs`, paper Eq. (1), single scalar ρ. Tests: `Cov(z)≈I`, `Cov(z,z′)≈ρI`, per-dim KS normality, ρ=0 → independent, ρ=1 → bitwise equality, seeded reproducibility, ρ outside [0,1] rejected
- [x] 🤖 **LatentSpec + squash + tests** — `idtb.latents.{Handle, LatentSpec}`, both corrections to README §5.1 built in:
  - Handles keyed by **role** (`arm.j0`, `cube.x`); `Handle.from_limits(role, lo, hi, fraction)` so no radius is ever an absolute number in code
  - `saturated(phi, atol=…)` flags where float32 `tanh` has collapsed injectivity, in **physical** space. A test pins the real boundary: `squash(9.0) == squash(12.0)` in float32, and both are flagged
- [x] 🤖 **Pod scripts** — `infra/preflight.sh` (read-only, never fails fast, hard-fails an RT-core-less GPU) and `infra/bootstrap.sh` (chown volume, relocate all caches onto it by symlink, checkout)

**README updated in the same change:** §3.1 gained four decided rows, §4.6 the real layout, §4.7 the dev loop, §5.1 the float32 caveat, §6.1–6.2 the shipped code, §8.3 the vendor-image decision, §9/§12 the re-sequencing.

**Done when:** `pytest` green locally, zero collection errors.

---

## Phase 2 — Pod setup — 🟡 versions resolved, pod not yet on the vendor image

> **No Dockerfile needed.** NVIDIA ships a prebuilt headless `nvcr.io/nvidia/isaac-lab` image on NGC. README §8.3's custom image is unnecessary — pull the vendor image, put our code on the volume.

- [x] 🧑 **Pick the datacenter by RT-core GPU availability, *then* create the network volume**
- [x] 🧑 **Launch the pod**, confirm the allocated GPU is the advertised part — RTX 4090, 24 GB, CC 8.9
- [x] 🧑🤖 **Run `infra/preflight.sh`, paste the output** — all checks passed; the full survey is recorded in README §8.1
- [x] 🤖 **Resolve versions from the driver reading** — **Isaac Sim 6.0.0, Isaac Lab 3.0.0-beta2, Python 3.12**, image `nvcr.io/nvidia/isaac-lab:3.0.0-beta2`
  - The driver *does* decide it, one layer deeper than expected: 580.178.04 clears Isaac Sim 5.1.0 (580.65.06) and 6.0.0 (580.95.05) but **not** 6.0.1 or 6.1.0, which both test at 595.58.03. It sets a ceiling, and 6.0.0 is the newest release under it.
  - **First answer was wrong and got reversed.** The initial call was stable 5.1.0 / Isaac Lab 2.3.2, on the reasoning that a beta simulator is not worth the risk — citing IsaacLab#6200 as an open dependency conflict. That issue had been **closed on 2026-08-04**, and Isaac Sim 6.1.0 had shipped three days before this was written. Checking the tracker reversed the decision; see README §3.4.
  - **[IsaacSim#367](https://github.com/isaac-sim/IsaacSim/issues/367) is Spike 4.** `TiledCamera` returns broken tiles under ray tracing; NVIDIA reproduced it and closed it with *"can confirm the issue in 5.1, it is however fixed in 6.0"*. 5.1.0 is terminal (Oct 2025, no patch line), so that never gets fixed there. Spiking on 5.1 and later moving to 6.x would also discard the determinism verdict and N.
  - Python 3.12, not 3.11: the `isaacsim` 6.0.0.0 wheels declare `requires_python == 3.12.*` where 5.1.0.0 declared `3.11.*`.
  - ⚠️ **`3.0.0-beta2` vs `3.0.0-beta2-post1`.** The `-post1` tag is `v3.0.0-beta2.patch1`, which moved to Isaac Sim 6.0.1 → driver 595.58.03 → **not runnable on this host**. Four characters apart, pulls cleanly, fails at a layer that never mentions drivers.
- [x] 🤖 **README checkpoint 1** — §3.1 rewritten around the 6.0.0 stack, new §3.4 recording why a beta beats the stable line here, §8.1 gained the measured survey, §8.3 rewritten, §9/§11/§12 updated. Three upstream issues the plan cited as live hazards (#251, #367, IsaacLab#3239) are all closed; §4.4 and §11 now say *how* each closed, because #251 closed by reassignment to the Isaac Lab layer rather than by a fix.
- [x] 🤖 **Fix the pod scripts against the vendor layout** *(not in the original plan — reading `docker/.env.base`, `Dockerfile.base` and `docker-compose.yaml` to resolve the versions turned up four defects in code written blind)*
  - **`/workspace` collision.** The image unpacks Isaac Lab into `/workspace/isaaclab`; both scripts defaulted the volume to `/workspace` and would have shadowed it. The symptom is a missing `isaaclab.sh`, which reads as a broken image. Default is now `/idtb` and `/workspace` is refused outright.
  - **Missing caches.** `bootstrap.sh` only relocated `$HOME` paths, so it missed `/isaac-sim/kit/cache` — the Kit extension cache, the largest — and `/isaac-sim/kit/data`, which is new in Isaac Lab 3.0.
  - **Non-root container.** `Dockerfile.base` ends on `USER isaaclab` (uid/gid 1000) and ships no `sudo`, while a provider volume arrives root-owned. `bootstrap.sh` now stops with `chown -R 1000:1000 /idtb` as the instruction instead of failing later inside Kit as a `PermissionError` on `logs/` or an `omni.datastore` lock error.
  - **`OMNI_KIT_ALLOW_ROOT` is conditional.** 2.3.2 runs as root and needs it; 3.0 does not. It is emitted only when the uid really is root, so the script stays correct across both images.
  - `tests/test_infra_scripts.py` pins all of it: the vendor cache list, the `/workspace` refusal, both uid regimes, and an end-to-end idempotent bootstrap against a fake home, fake Isaac root and local git origin.

**Remaining — 🧑 Julian, next pod session:**

- [ ] 🧑 **NGC account + API key, then `docker login nvcr.io`.** The survey's `401` means reachable-and-unauthenticated; the pull fails until this is done.
- [ ] 🧑 **Recreate the pod** on image **`nvcr.io/nvidia/isaac-lab:3.0.0-beta2`** — *not* `-post1`, see above — with:
  - the network volume mounted at **`/idtb`** — *not* `/workspace`; the mount path is a pod setting, so the volume's contents are unaffected
  - pod env var `ACCEPT_EULA=Y`
  - `chown -R 1000:1000 /idtb` from a root shell once, before the Isaac container needs the volume
- [ ] 🧑 **Re-run `infra/preflight.sh` inside the container.** It now also checks that `isaaclab.sh` survived the mount and that the env vars are set. Paste the output.
- [ ] 🧑 **Run `infra/bootstrap.sh`**, then stop/restart the pod and confirm caches survived (no re-download). That restart is the only proof the relocation works — a cache that is silently not persisting looks exactly like a slow first run.
- [ ] 🧑 **Run a shipped Isaac Lab tutorial and look at the PNG** — separates "environment broken" from "my blind code broken". 2 minutes, zero code, saves an ambiguous debugging session later.

  ```
  source /idtb/env.sh
  cd $ISAACLAB_PATH
  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
  ```

---

## Phase 3 — The spike

- [ ] 🤖 **Write `spikes/spike_api.py`** — one standalone file. No protocol, no USD authoring. Isaac Lab's shipped `FRANKA_PANDA_CFG` + a cube + one camera.

  Three structural rules:
  1. **Never fail fast.** Every check isolated, all run, PASS/FAIL table + `facts.json` at the end. `SimulationApp` is one-shot per process and boot is slow — an assert-and-die script gives one failure per boot.
  2. **`num_envs=2`, widely spaced.** Issue #251 is "frozen *at the env origin*". With one env the origin is `(0,0,0)` and a plausible cube target is centimetres away — "frozen" and "correct" sit inside any sloppy tolerance. Two envs make it a metres-scale, unmissable error.
  3. **Determinism/convergence as reusable `(scene, preset)` functions**, not inline asserts — they get re-run against scene v1 later.

  | Check | What it answers |
  |---|---|
  | Boot, versions, GPU recorded | §3.2 rows |
  | Scene builds; **measured** joint names + limits, cube state, intrinsics | promotes §5.2 from provisional |
  | Write → read back matches, **zero `sim.step()`** | Spike 2, issue #251 |
  | Same + one `step()`: **how far** does state move | §5.5 validity policy |
  | Render non-degenerate (not all-black) | issue #367 |
  | **Sensitivity: arm-only and cube-only deltas ≫ noise floor** | **see below — rank first** |
  | Same state twice → equal; **and buffers not aliased** | Spike 1a |
  | A/B/B/A order-independent | Spike 1b, temporal leakage |
  | Convergence depth: Δ curve, smallest k below tol | Spike 3 — the GPU budget |
  | Repeat across render modes × denoiser/AA settings | Spike 1 verdict |
  | `TiledCamera` vs `Camera`, throughput at B ∈ {1,2,8,32} | Spike 4 |
  | ms/sample: preset × resolution × B | prices "100k pairs" |
  | **Segmentation annotator available + non-degenerate, per mode** | the occlusion-conditioned metric dies without it — and the mode is chosen *now* |
  | Cross-process determinism (hash to volume, second process matches) | makes spot/checkpoint resume sound |

  > **Why sensitivity ranks above determinism.** Read-back + non-degenerate + same-state-equal + A/B/B/A are *all satisfied by a pipeline that renders the same stale frame every time* — Fabric flush never reaching the render graph, or a cached annotator buffer. That's the most deterministic possible pipeline and it's 100% garbage, discovered weeks later when R² comes out zero. Same logic for the aliasing check: if Isaac returns a view onto a reused buffer, A1 and A2 compare equal *because they're the same memory*.

- [ ] 🧑 **Run it, paste the whole table.** Expect 2–4 iterations; keep the pod up between fixes under ~20 min (a cold boot costs more than the idle time)
- [ ] 🤖 **Fix and re-run** until every check has a verdict. A FAIL is fine if *understood* — "TiledCamera black under RealTimePathTracing, matches #367, use `Camera`" is a completed check.
- [ ] 🧑 **Save a few rendered arrays + their generating state, pull them down** — so the mock is built to real conventions, not assumed ones
- [ ] 🤖 **README checkpoint 2** — render mode + preset definitions (§7.3 rewritten), whether a physics step is required (**edit §5.5 in the same change if so** — its recommendation is conditional on this), convergence depth N (as a *procedure + the scene measured on*, never a bare constant), TiledCamera verdict, measured joint limits, ms/sample table
- [ ] 🧑 **Stop the pod**

---

## Phase 4 — Build against verified reality

- [ ] 🤖 **`SceneBackend` protocol** — now, not before. The risk was never the method *names*; it was granularity and semantics, which the spike just settled.
  - Batched from the start (`B=1` valid) — absorbs the TiledCamera question either way
  - Segmentation remapped to **our own stable ids** — Isaac's annotator ids aren't stable across runs, which would silently turn the visibility column into noise
- [ ] 🤖 **`MockSceneBackend`** — numpy only, built to the captured conventions. Anti-aliased analytic shapes (hard edges would make two nearby latents render identically and fail the injectivity test on the one backend where it must pass)
- [ ] 🤖 **Determinism/read-back gates as library code in `src/`**, not test-only — README §7.1 says the gate runs before every dataset generation, so `generate.py` and the tests must call the same function
  - **Acceptance is bitwise, not a threshold.** A temporal denoiser leak sits at ~0.3/255 and passes any sane threshold — while being exactly the structure a contrastive encoder is trained to find. A threshold doesn't weaken this gate, it disables it.
- [ ] 🤖 **Negative controls** — inject faults into the mock (jitter, temporal leak, frozen write simulating #251, aliased buffers) and assert the gates **fire**
  - Without these, a gate that passes is a detector nobody has ever seen detect anything
- [ ] 🤖 **Tier-1 contract suite** — one set of tests parametrized over both backends
- [ ] 🧑 **Final short pod session:** run the suite against Isaac, paste results

---

## Verification

**Locally (macOS):**
```
pytest          # tier 0 + mock half green, zero collection errors
```

**Pod:**
```
./isaaclab.sh -p spikes/spike_api.py --out /workspace/data/spike
./isaaclab.sh -p -m pytest --isaac-mode=require
```
`--isaac-mode=require` matters: without it, a broken container that can't import `isaaclab` silently deselects the whole Isaac tier and reports green.

**Done means:** all four §7.2 spikes answered and recorded in README §3.1; protocol backed by running code on both sides; README describes nothing in this milestone as unknown.

---

## Known limit — don't treat the spike verdict as final

Denoiser and accumulation behaviour is **scene-, material- and light-dependent**. The spike uses a shipped Franka + cuboid + default light; scene v1 adds PBR materials, HDRI, area lights, real resolution. The determinism verdict may not transfer and **N almost certainly won't** (it'll be larger). That's why the checks are written as reusable functions and N is recorded as a procedure. Re-gate on scene v1 before generating any dataset — next milestone, but the functions must exist now or it's a rewrite.

---

## Deferred

- USD scene authoring, custom Dockerfile, dataset storage format (prefer **lossless** — compression artifacts would be indistinguishable from a real identifiability effect)
- LeJEPA/SIGReg trainer, behind an interface. Note: the authors published code at `github.com/klindtlab/lejepa-identifiability`, which makes "use theirs" a live option when we get there.
- Everything in README Phases 5–10
