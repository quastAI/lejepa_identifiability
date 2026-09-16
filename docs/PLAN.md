# Milestone 1: Isaac API green + pure layers

> Execution checklist for the current milestone. `README.md` remains the plan of record — this file is the ordered task list for getting to first verified Isaac contact. Checkpoints below say when to write results back into the README.

## Context

Starting point: the repo had **no source code** — only README.md, LICENSE, and the paper PDF. Phase 1 and Phase 2 are both done: versions resolved onto the pod already running rather than a re-selected one (README §3.4.1 — two attempts to select a host by driver both failed to land the target branch, so the plan stopped chasing it), and a real headless Isaac Lab tutorial has since completed successfully on that driver. Phase 3 onwards is open.

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

## Phase 2 — Pod setup — ✅ done

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

- [x] 🧑 **NGC account + API key, then `docker login nvcr.io`** — done; the image pulled.
- [x] 🧑 **Recreate the pod**, volume at `/idtb`, `ACCEPT_EULA=Y`, volume chowned to 1000. All four confirmed by the second survey: `/isaac-sim` present, `/workspace/isaaclab/isaaclab.sh` intact (the mount did not shadow it), EULA set, volume `ubuntu:ubuntu` and writable. *Correction after the fact:* the image that actually came up was `3.0.0-beta2-post1`, not plain `beta2` as intended at the time — unnoticed until the version-check row below. See the reversal bullet further down for why it's being kept rather than replaced.
- [x] 🧑🤖 **Re-run `infra/preflight.sh` on the vendor image** — run on 2026-09-16. It printed *all checks passed*, and it was wrong on two counts. Both are recorded in README §8.1:
  - **Driver 570.195.03**, where the stack was resolved against 580.178.04. `nvidia-smi` in a container reports the *host* driver, and a recreate is a fresh allocation — so §3.1's release was resolved against a machine we no longer have. 6.0.0 is tested at 580.95.05. **It is not a floor**: from 5.1.0 on NVIDIA only publishes "tested on", and Kit refuses below 535.129, so 6.0.0 would very likely boot here and the difference would surface as unexplainable §7.1 numbers. The fix is host selection — RunPod's **CUDA Version = 13.0** filter — not a different tag.
  - **No egress**, all four endpoints, where the first survey reached all four. `bootstrap.sh` cannot clone and Isaac cannot stream assets without it.
- [x] 🤖 **Fix the two defects the survey exposed** *(not in the original plan)*
  - **The false green.** `curl` writes `000` for `%{http_code}` on a failed transfer *and* exits non-zero, so the `|| echo "000"` fallback appended a second one — `000000`, which compared unequal to `000`, so a total blackout scored green. The exit status decides it now and the code (6 = DNS, 7 = refused, 28 = timeout, 35/60 = TLS) is reported, which is the only thing that says *why*.
  - **Nothing checked the driver against the resolved release**, because when the script was written the driver was the input to that decision rather than a requirement of it. It now fails below 580.95.05 and warns at 595+ ([IsaacSim #537](https://github.com/isaac-sim/IsaacSim/issues/537) — newer is not safer).
  - Also: the survey now reads the **image's own Isaac Sim version**, so `beta2` vs `beta2-post1` is confirmed from inside the container instead of trusted; probes Isaac's bundled interpreter (the image ships no `python3` on `PATH`, which the old script reported as a raw shell error inside the value); and gained a non-blocking `WARN` tier.
  - `tests/test_infra_scripts.py` pins all of it, including the blackout regression, the octal trap in comparing driver fields like `.08`, and that an image shipping no `VERSION` file is reported as absent rather than failed.
- [x] 🧑🤖 **Three more `bootstrap.sh` defects, found by running it on the pod** — all were invisible from macOS, and all are now in README §8.3 with tests:
  - **`HOME=/root` while running as uid 1000.** Every `$HOME` cache path was unwritable, so the relocation aborted midway under `set -e` — some caches moved, some not, which reads as half-finished rather than failed. The script now falls back to the password database when `HOME` is not writable, and exports the corrected value into `env.sh` so Isaac does not read caches at a path nothing was relocated to.
  - **It cloned the repo, which was circular** — the script ships inside the repo, so anything that can run it already has one, and the clone landed at a *second* path (`$VOL/repo`), so the checkout being edited was not the checkout Isaac ran. Removed: it reports the checkout it runs from, warns when that is not on the volume (gone at the next pod start), and touches the network nowhere. That also disposes of the DNS race — Docker's embedded resolver is briefly not forwarding after container boot, and one lookup failure used to kill a bootstrap that had already relocated every cache.
  - **`/isaac-sim/kit` is root-owned, and the §8.3 `chown` instruction never covered it** — that instruction is about the *volume*. Measured live: `rm: cannot remove '/isaac-sim/kit/cache': Permission denied`, and the script died there under `set -e` with `kit-data` and every `$HOME` cache after it left un-relocated. A symlink replaces a *parent directory* entry, so this bites regardless of permissions on the cache contents themselves. Fixed by probing the nearest existing ancestor of each target before touching it and collecting every path that fails instead of stopping at the first — everything relocatable still gets relocated — then exiting non-zero at the very end with the exact `chown -R <uid>:<gid> <path>` needed, because a run that quietly skipped the single largest cache and reported "done" would be worse than the crash.
  - Deliberately *not* changed: the `chown -R` logic on the volume itself. A non-root user really can take ownership there, which is unusual but verified on the pod — defensive changes would be guarding against a case that does not occur.
- [x] 🧑🤖 **Version decision reversed: staying on 6.0.1 (`-post1`), not recreating for 6.0.0.** Two recreates aimed at a 580-branch host for plain `beta2` both landed on 570-branch hosts instead — the provider's allocation decides the driver, not a CUDA-version filter chosen after the fact. The pod already runs `beta2-post1` (Isaac Sim 6.0.1), pulled before the tag mismatch was noticed, and `bootstrap.sh` has run against it. Re-pulling `beta2` now would trade a working image for an unverified one to chase a driver branch two attempts already failed to control, over a one-patch-release difference (6.0.0 → 6.0.1) smaller than the beta-vs-stable exposure already accepted in §3.4. Full reasoning in README §3.4.1; §3.1, §8.1, §8.3, §11, §12 updated to match — driver 570.195.03 vs. 6.0.1's tested 595.58.03 is now a recorded, accepted gap, not a defect to fix by recreating.

**Remaining — 🧑 Julian, next pod session, same pod:**

- [x] 🧑🤖 **Confirm egress properly, don't just route around it.** Confirmed transient: re-ran `infra/preflight.sh` with no `--resolve` workaround and all four endpoints reached cleanly (`nvcr.io` 401, the rest 200), matching the Docker-embedded-resolver theory rather than a dead network. Same run also confirmed the tag check now passes (`isaac sim version 6.0.1-rc.7+release.42383.32955d8d.gl` matches the updated `WANT_SIM=6.0.1`) and measured **Python 3.12.13** on the pod's bundled interpreter — turns the §3.1 Python row from an inference into a confirmed fact. Only remaining `FAIL` is the accepted driver gap (now 580.159.04 vs. tested 595.58.03 — narrower than the 570.195.03 seen earlier, still below, still accepted per §3.4.1).
- [x] 🧑🤖 **Get the fixed `infra/bootstrap.sh` onto the pod, then run it.** Done via the tarball re-fetch — confirmed the fix works exactly as designed: `/isaac-sim/kit` reported `BLOCKED` (root-owned, as predicted) while every other cache still relocated, including `/root/.cache/*` — which turned out writable by uid 1000 on this image, unlike `/isaac-sim/kit`, so no HOME fallback was even needed here.
- [x] 🧑🤖 **Chown `/isaac-sim/kit` — found to be impossible, then found not to matter.** `su` and `sudo` both fail from inside the container (no `sudo` binary at all), so the suggested root-shell fix doesn't exist on this image. Checked what's actually lost: `ls -ld` / `stat` on `/isaac-sim/kit/cache` confirmed it's `ubuntu:ubuntu` and writable — only the *parent* `/isaac-sim/kit` blocks the relocation, so Isaac's own runtime writes into the cache are unaffected. The real cost is that `kit/cache` (the largest persisted cache) doesn't survive a pod restart/recreate, paid as a slower cold start each time that happens, not per command. A custom-image rebuild (`chown` + the already-known `ENTRYPOINT []` fix) was considered and **declined** — same reasoning as §3.4.1: new surface (a registry to maintain) for a minor, already-mitigated cost. `bootstrap.sh` now reports this as a non-fatal `NOTE` and exits `0` — no skip flag added; there's nothing to configure, it's just no longer treated as a failure with a fix to chase. README §8.3 and `tests/test_infra_scripts.py` updated to match.
- [x] 🧑🤖 **Confirm caches survive, properly** — a same-container restart wouldn't prove anything (the container's own ephemeral layer survives a restart regardless of relocation), so the pod was **deleted and recreated**, not restarted. Landed on a new container (`f0e893a3661e`) at driver **580.159.04** — narrower gap than 570.195.03, still below 595.58.03, still accepted (§3.4.1). Both `infra/preflight.sh` and `infra/bootstrap.sh` re-ran clean: `bootstrap.sh` correctly recreated the `$HOME` symlinks from scratch (a fresh container has no symlinks of its own yet — `already linked` only applies to a same-container restart) pointing at the same `/idtb/cache/*` content, and `/isaac-sim/kit` reported its `NOTE` again as expected. Also re-confirmed the tag check passes and Python is 3.12.13 on this pod too. **Directly verified**, not just inferred from clean symlinking: `ls -la /idtb/cache/ov` shows `_cache.lock` timestamped **before** the previous (now-deleted) pod's own preflight run, with a fully populated `DerivedDataCache`/`Kit`/`shaders`/`texturecache`/`ogn_generated` structure — content that could only exist by surviving at least one prior delete+recreate cycle.
- [x] 🧑🤖 **Run a shipped Isaac Lab tutorial** — `create_empty.py --headless` completed cleanly (`[INFO]: Setup complete...`) on the new pod's driver 580.159.04. This is the first **direct, empirical** evidence against a driver-mismatch crash, not just the documentation-based reasoning in §3.4.1. **Correction: no PNG.** This tutorial is scene composition, not rendering — the "look at the PNG" framing assumed the wrong tutorial existed to check against. A real visual/determinism check is Spike 1 (§7.2), not a separate step. One thing to watch later, not now: Kit's `OmniHub` helper failed to launch and retried ~44 times (~14s) before continuing without it — harmless here (no external assets needed), worth attention once a script streams the Franka asset.

  ```
  source /idtb/env.sh
  cd $ISAACLAB_PATH
  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
  ```

> **The driver gap (now 580.159.04 vs. tested 595.58.03) is accepted, not pending.** No further step in this checklist is about closing it. README §3.4.1 explains why: it's a "tested on" figure, not a hard floor (Kit itself only refuses below 535.129), and the real check is the §7.2 spike measuring the renderer directly on this exact host — reinforced now by a real headless boot completing on this driver. If a render or determinism result later looks wrong in a way nothing else explains, this is the first thing to revisit — and the escape hatch (Isaac Sim 5.0.0 + Isaac Lab 2.2.0 + Python 3.11, README §3.4.1) is recorded for that case, not for routine use.

**Phase 2 is done.** Every item above is closed. Next is Phase 3 — the spike.

---

## Phase 3 — The spike

- [x] 🤖 **Write `spikes/spike_api.py`** — one standalone file. No protocol, no USD authoring. Isaac Lab's shipped `FRANKA_PANDA_CFG` + a cube + one camera. **Written; every check in the table below is implemented and unrun.** Four things about how it was built, all consequences of writing it blind:
  - **The detectors are pure and tested locally.** Everything above the "Isaac layer" banner (`determinism_report`, `convergence_report`, `sensitivity_report`, the `Report` registry, the uint8-safe diffs) imports no Isaac, so `tests/test_spike_api.py` runs it on macOS. That file is mostly **negative controls** — a stale renderer, an aliased buffer, a temporal leak, free-running MC noise — each asserting the matching report *fires*. Phase 4 wants these anyway (§10.1); having them now means the spike's verdict comes from detectors that have been watched detecting.
  - **A threshold test pins why §7.1 is bitwise:** a one-grey-level leak passes `tol=1.0` and fails `bitwise`, in the same assertion.
  - **Every Isaac symbol is resolved, not assumed.** `resolve()` tries the known spellings of `FRANKA_PANDA_CFG` and records which answered; the scene builds down a ladder (semantics+tiled+rich → plain rgb) so one unknown kwarg costs a recorded note, not the run; each preset's carb settings are read back, because carb silently creates unknown keys and only equality proves the path exists.
  - **Arm perturbation is a fraction of the *measured* half-range** (§5.2), not a hardcoded radian value.
- [ ] 🤖 **Deliberate deviation to confirm on the pod: the `B ∈ {1,2,8,32}` sweep is across runs, not within one.** `num_envs` is fixed when the scene is built and `SimulationApp` is one-shot per process, so the script takes `--num-envs` and records it in `facts.json`; the table is assembled by running it four times. Cheaper than rebuilding a second scene mid-process, and each run stays a clean boot.
- [x] 🧑🤖 **First real run on the pod, on 2026-09-16 — found two defects, both in the spike script itself, before any real check could run.** `boot_and_versions` passed clean (torch 2.10.0+cu128, isaaclab 6.1.16, CUDA 12.8, RTX 4090); `scene_builds_and_measures` failed on all three ladder rungs. Both fixed and covered by `tests/test_spike_api.py`, not just patched blind:
  - **The retry ladder contaminated its own retries.** A failed `InteractiveScene()` call leaves whatever prims it already created sitting on the stage — USD construction has no transactional rollback — so rungs 2 and 3 died on `A prim already exists at path: '/World/ground'` instead of their own errors, masking the real problem behind two copies of a bug in the harness. Fixed by giving every prim path a per-rung suffix (`/World/ground_r0`, `_r1`, `_r2`, …) instead of trying to reset the stage between attempts — safer than guessing whether `omni.usd.get_context().new_stage()` plays cleanly with an already-constructed `SimulationContext`, which nothing here was confident about.
  - **`FRANKA_PANDA_CFG`'s shipped `usd_path` 404s.** The real error, once the ladder could actually show it: `FileNotFoundError` on `.../Robots/FrankaEmika/panda_instanceable.usd`. Researched rather than guessed at — confirmed by direct HEAD request against the real asset tree that the object moved under a `Legacy/` subfolder and the shipped config was never updated to match. No tracked issue number found for this one specifically; the code (`franka.py`) and the asset tree have simply diverged. Franka is the only shipped robot config with this split — Unitree, ANYbotics, UR and the Kuka/Allegro configs were checked and are fine. `build_rig()` now patches the one known-broken suffix via a pure, tested `correct_franka_usd_path()` and records what it did either way — corrected, left alone because the suffix didn't match (upstream already fixed it, or a different path entirely), or failed to patch — rather than assuming the fix still applies on a future isaaclab version.
  - **Unrelated but fixed at the same time:** the ~35 `OmniHub` retry warnings (~10s) at every boot are a *different*, cosmetic bug — the image sets `HUB__ARGS__DETECT_ONLY=true` while `omni.client` defaults to asking Hub to launch anyway, and the two settings contradict each other. Root-caused and merged upstream as [IsaacLab#6971](https://github.com/isaac-sim/IsaacLab/pull/6971); reported as still open for this exact image as issue #7732 (repo not independently re-verified — treat the number as a lead, not a confirmed link). Hub is a local USD caching layer, not required for `https://` resolution at all — confirmed via IsaacLab's own kitless runs, which disable it the same way ([#6985](https://github.com/isaac-sim/IsaacLab/pull/6985)). `spike_api.py` now sets `OMNICLIENT_HUB_MODE=disabled` via `os.environ.setdefault` at module load, before any Isaac import — an operator's own env export still wins.
  - Lesson for the rest of Phase 3: **the "never fail fast" design worked exactly as intended even though the run itself failed** — one boot produced a full report that correctly separated a harness bug (ladder contamination) from a real upstream one (the 404) instead of one opaque crash. Worth remembering when reading the next run's `facts.json` too.
- [x] 🧑🤖 **Second run, same day — both fixes confirmed, scene builds clean, 10/12 checks pass.** No `OmniHub` noise at all; `franka_usd_path` shows the `Legacy/` correction applying; `scene_builds_and_measures` passes on the first ladder rung (semantics+tiled+rich) in 17.9s. `boot_and_versions`: torch 2.10.0+cu128, isaaclab reports `6.1.16`, CUDA 12.8. Two `FAIL`s, read together rather than separately:
  - **`determinism_and_aliasing` FAILs on `buffers_aliased: true` — real, and answers a question this project needed answered, not a defect.** Traced the mechanism exactly: `determinism_report()` holds a live reference (`a1_live`) into `camera.data.output[...]` deliberately, and a *later* `capture(state_b)` call overwrites the same underlying buffer before `a1_live` is ever compared — proving Isaac's camera sensor returns a reused, mutable tensor across calls, not a fresh one. **Consequence for §4.5/§6.3/`writer.py`/`generate.py`: every capture must be `.clone()`d immediately, before the next capture call, or a caller silently observes the wrong frame's data.** Everywhere else in the spike already does this; only the aliasing probe deliberately doesn't, by design, specifically to catch this.
  - **`sensitivity_arm_and_cube` FAILs — first theory was `--render-depth 8` being under-converged; re-run at 64 falsified it.** `order_independent_mad` was `48.5` at depth 8 and `48.1` at depth 64 — statistically identical, no improvement from an 8× increase in `sim.render()` calls. If depth genuinely drove path-tracing accumulation, that gap should have collapsed. It didn't, which pointed at the real bug below instead.
  - **Two encouraging side-findings, unforced:** `tiled_vs_camera` shows `Camera` and `TiledCamera` producing identical stats with `tiles_distinct_mad: 58.8` for both — no sign of the #367 tile-corruption bug on this image. And `pathtracing_denoiser_off` already achieves `order_independent_bitwise: true` and `back_to_back_bitwise: true` *even at depth 8*, because its own `totalSpp=64` setting drives internal accumulation independent of the external render-depth loop.
- [x] 🧑🤖 **Third run confirmed the depth-64 re-run showed zero improvement — traced to two real bugs in the spike's own measurement, not the renderer. Both fixed:**
  - **`convergence_depth`'s "converged at 64" was a tautology.** `convergence_report()` built `deepest = frames[depths[-1]]` and then compared *every* depth, including 64 itself, against that same object — depth 64 vs. depth 64 is mad=0 by construction, proving nothing. The real signal was depths 1–32 sitting flat at ~48 with **zero improving trend**, meaning `sim.render()` call count isn't reducing noise for the default render mode at all. Fixed: an independent second capture at the deepest depth is now the reference, so even the deepest row can show it hasn't converged.
  - **`buffers_aliased` and rendering-quality determinism were conflated into one `deterministic` flag.** Aliasing (confirmed real, §above) is a sensor-API fact, true on *every* preset tried, regardless of render mode. Folding it into `deterministic` meant `render_mode_sweep`'s `deterministic_presets` list could never be non-empty no matter how good a preset actually was — which is exactly what buried `pathtracing_denoiser_off`'s already-bitwise-clean numbers on the last two runs. Fixed: `determinism_report()` now reports `content_reproducible` (from the cloned a1/a2/b1/b2 comparisons, trustworthy regardless of aliasing) separately from `buffers_aliased`; `deterministic` stays the strict AND of both for a non-cloning caller, but `render_mode_sweep` and `content_reproducible` are what actually answer "which preset works."
  - **Consequence for the dedicated `sensitivity_arm_and_cube`/`determinism_and_aliasing` checks**: they were testing under whatever the sim booted into (noisy `RealTimePathTracing`, `as_booted`), which README §7.1 already warned against — "naming a realtime mode does not buy determinism." Both now apply `pathtracing_denoiser_off` explicitly before capturing, matching the one preset with actual evidence behind it, and record `preset_applied` in their facts so this choice is visible, not assumed. `check_determinism`'s failure message now separates "content not reproducible" (a settings problem) from "buffers aliased" (a permanent, always-clone caller obligation) instead of one blob.
  - `tests/test_spike_api.py` gained two regression tests: one proving the convergence self-comparison bug can't recur (a fake that never settles, even at the deepest depth, must not read as converged), one proving aliasing and content-reproducibility are measured independently.
  - `throughput`'s `projected_hours_per_100k_pairs: ~0.4` was measured under the noisy default mode at both depth 8 and 64 and should be **re-measured under `pathtracing_denoiser_off`** — its cost profile (fixed internal `totalSpp`) may differ from a `sim.render()`-loop-driven number entirely.
- [x] 🧑🤖 **Fourth run confirms it — 12/13 pass, Spike 1 answered.** `sensitivity_arm_and_cube` PASS (arm mad 78.7, cube mad 76.5, against a literal `0.0` noise floor). `determinism_and_aliasing` FAILs on exactly one reason now — `buffers_aliased` — nothing else. `render_mode_sweep.deterministic_presets = ["as_booted", "pathtracing_denoiser_off"]`, and `cross_process_determinism` PASS on the repeat run. **Verdict recorded for §3: `pathtracing_denoiser_off` (`spp=1, totalSpp=64, optixDenoiser off`) is the deterministic setting; the sensor buffer must always be `.clone()`d immediately, permanently, on this Isaac version.** Two things worth knowing, not worth more pod time:
  - `as_booted` showing deterministic in this run is an artifact, not independent confirmation — carb settings are global and persist across checks, so `pathtracing_denoiser_off`'s settings from the two earlier checks were still active when `render_mode_sweep` measured "as_booted" (its frame stats are identical to `pathtracing_denoiser_off`'s). The true boot default's own noise (~48 mad) was already established in the first run, before any preset had been touched.
  - `convergence_depth` now shows every depth 1–64 bitwise identical, because `totalSpp=64` converges fully in a single `sim.render()` call — the external depth loop is irrelevant once these carb settings are set. The real cost lever going forward is `spp`/`totalSpp`, not "render depth."
  - Spike 1 is done. No further pod runs needed for it — the one remaining FAIL is permanent and correct, same category as the `/isaac-sim/kit` NOTE in `bootstrap.sh`.

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
- [ ] 🧑 **Save a few rendered arrays + their generating state, pull them down** — so the mock is built to real conventions, not assumed ones. Still open — `spike_api.py` doesn't currently save frames to disk, only stats to `facts.json`.
- [x] 🤖 **README checkpoint 2** — §7.2 rewritten with all four Spike answers; §7.3 `standard` preset now a real measured carb config (`PathTracing`, `spp=1`, `totalSpp=64`, denoiser off); §5.5 confirmed (zero `sim.step()`, measured, not assumed); §5.2 joint limits promoted from provisional to measured for the four active arm joints + gripper (cube position stays provisional — no table in the spike scene); §3.1/§3.2 Decision Register updated (four Spike questions moved from deferred to decided, plus the unplanned camera-aliasing finding added); §11 Risk Register rows closed/updated to match.
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
./isaaclab.sh -p spikes/spike_api.py --out /idtb/data/spike
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
