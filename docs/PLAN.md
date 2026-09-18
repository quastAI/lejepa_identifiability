# Milestone 1: Isaac API green + pure layers

> Execution checklist for the current milestone. `README.md` remains the plan of record — this file is the ordered task list for getting to first verified Isaac contact. Checkpoints below say when to write results back into the README.

## Context

Starting point: the repo had **no source code** — only README.md, LICENSE, and the paper PDF. Phase 1 and Phase 2 are both done: versions resolved onto the pod already running rather than a re-selected one (README §3.4.1 — two attempts to select a host by driver both failed to land the target branch, so the plan stopped chasing it), and a real headless Isaac Lab tutorial has since completed successfully on that driver. Phase 3 onwards is open.

Every Isaac API signature in README §4.4 came from reading docs, **never from running anything**. So: meet the Isaac API first in one spike script, then design the protocol against verified reality. The README's current order (protocol + mock first, Isaac at Phase 3) would mean designing the central abstraction against guesses.

**Scope:** stops at "Isaac API verified on the pod + pure layers green locally + protocol and mock built." ~3–4 days, now ~5 with the latent-group work below.

**Amended 2026-09-17 — three latent groups.** The latent set was a single flat
vector of 7 handles. README §5.2 now splits every simulator knob into `base`
(what single-cube pick-and-place needs), `full` (⊇ `base`, every task-related
factor — cube size, cube colour), and `style` (what the task must never depend on
— lighting, camera jitter, table material), with `style` resampled *within* each
pair at ρ = 0 so the encoder is pushed to be invariant to it. Two consequences for
this milestone, both below: the pure layer needs group support (**Phase 3a**,
local, no GPU, no blockers), and every `full`/`style` knob writes through an API
that **Spikes 1–4 verified nothing about** (**Phase 3b**, one pod session).

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

- [x] 🤖 **Write `spikes/spike_api.py`** — one standalone file. No protocol, no USD authoring. Isaac Lab's shipped `FRANKA_PANDA_CFG` + a cube + one camera. **Written, and run four times below — every check in the table has a verdict.** Four things about how it was built, all consequences of writing it blind:
  - **The detectors are pure and tested locally.** Everything above the "Isaac layer" banner (`determinism_report`, `convergence_report`, `sensitivity_report`, the `Report` registry, the uint8-safe diffs) imports no Isaac, so `tests/test_spike_api.py` runs it on macOS. That file is mostly **negative controls** — a stale renderer, an aliased buffer, a temporal leak, free-running MC noise — each asserting the matching report *fires*. Phase 4 wants these anyway (§10.1); having them now means the spike's verdict comes from detectors that have been watched detecting.
  - **A threshold test pins why §7.1 is bitwise:** a one-grey-level leak passes `tol=1.0` and fails `bitwise`, in the same assertion.
  - **Every Isaac symbol is resolved, not assumed.** `resolve()` tries the known spellings of `FRANKA_PANDA_CFG` and records which answered; the scene builds down a ladder (semantics+tiled+rich → plain rgb) so one unknown kwarg costs a recorded note, not the run; each preset's carb settings are read back, because carb silently creates unknown keys and only equality proves the path exists.
  - **Arm perturbation is a fraction of the *measured* half-range** (§5.2), not a hardcoded radian value.
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

- [x] 🧑 **Run it, paste the whole table.** Done — see the four run entries above; it took exactly the expected 2–4 iterations.
- [x] 🤖 **Fix and re-run** until every check has a verdict. Done — 12/13 PASS, the one FAIL (`buffers_aliased`) is understood and permanent, same standard as "TiledCamera black under RealTimePathTracing, matches #367, use `Camera`" would have been.

> **Two loose ends, both moved to Phase 3b's pod session rather than left dangling here.** Spike 1 itself is done (line above) and needs no further pod time on its own account, but two small measurements never got taken and both need the pod up — which Phase 3b needs anyway, so they ride along instead of costing a second spin-up:
> - the `B ∈ {1,2,8,32}` throughput sweep (`spike_api.py --num-envs {1,8,32}`) — the *mechanism* is confirmed (`num_envs` fixed per process, `facts.json` records it), only the sweep itself is missing; all four runs so far used the default `--num-envs 2`
> - saving a few rendered arrays + their generating state to `/idtb/data/spike`, pulled down locally, so `MockSceneBackend` (Phase 4) is built to real conventions instead of assumed ones
>
> Tracked as their own checklist items under **Phase 3b**, not here, so Phase 3 closes clean.

- [x] 🤖 **README checkpoint 2** — §7.2 rewritten with all four Spike answers; §7.3 `standard` preset now a real measured carb config (`PathTracing`, `spp=1`, `totalSpp=64`, denoiser off); §5.5 confirmed (zero `sim.step()`, measured, not assumed); §5.2 joint limits promoted from provisional to measured for the four active arm joints + gripper (cube position stays provisional — no table in the spike scene); §3.1/§3.2 Decision Register updated (four Spike questions moved from deferred to decided, plus the unplanned camera-aliasing finding added); §11 Risk Register rows closed/updated to match.
- [x] 🧑 **Stop the pod** — done, 2026-09-16.

---

## Phase 3a — Latent groups in the pure layer (local, no GPU, no blockers)

> Everything here is decided structure (README §5.2, §5.4.1, §6.1–6.2). Only the
> *radii* and the group membership of individual knobs wait on Phase 3b, and those
> are data, not code. This is Track B work: it can land before the pod is next up.

- [x] 🤖 **`Handle.group`** — a fourth field, `"base" | "full" | "style"`, defaulting to `"base"` so every existing call site and test keeps working unchanged. `from_limits` grows a `group=` keyword. Reject any other value at construction; three groups is the decision (README §3.1), not a placeholder for an open vocabulary.
- [x] 🤖 **Ordering contract in `LatentSpec.__post_init__`** — handles must appear `base`, then `full`, then `style`. Enforced, not documented: it is what makes `dims("base")` a *prefix* of `dims("full")`, which in turn is what lets latent index *i* mean the same physical thing in a `base` run and a `full` run. A spec that interleaves them raises.
- [x] 🤖 **Group views on `LatentSpec`**
  - `dims(group) -> tuple[int, ...]` — **cumulative**: `dims("full")` returns base *and* full-tagged dims; `dims("style")` returns only style dims (disjoint).
  - `subset(group) -> LatentSpec` — a narrower spec preserving dimension order.
  - `group_of_dim -> tuple[str, ...]` — per-dimension tags, stored with every shard (README §6.4) so the analysis slices by tag instead of re-deriving the split from role-name prefixes.
  - `rho_vector(rho_task, *, rho_style=0.0) -> Tensor[n]` — `rho_task` on base/full dims, `rho_style` on style dims. A *parameter*, not a hardcoded zero: §5.4.1 predicts a spectrum crossing at ρ_style = ρ_task² that only a sweep can test.
- [x] 🤖 **Vector ρ in `sample_ou_pairs`** — accept a scalar *or* a length-*n* tensor. The update `z' = ρz + √(1−ρ²)η` is already elementwise, so this is broadcasting plus validation; the scalar path must stay bitwise identical to today's.
- [x] 🤖 **Tier-0 tests**, in the same change:
  - cumulative membership (`dims("full")` ⊇ `dims("base")`), and `dims("base")` is a *prefix*
  - an out-of-order spec raises; an unknown group name raises
  - `subset("base").squash(z)` agrees element-for-element with the full spec's squash on those dims
  - **block cross-covariance**: with a mixed spec and `rho_vector(0.95)`, `Cov(z, z′)` measures 0.95·I on the task block and **exactly 0** on the style block (this is README §10.1's group-wiring gate, and it is the only direct evidence ρ = 0 reached the dimensions it was meant to)
  - scalar-ρ regression: same seed, scalar `0.95` vs. a constant vector `0.95` → bitwise equal
  - ρ vector of the wrong length, and any element outside [0, 1], both rejected

**Done when:** `pytest` green, and a mixed `base`/`full`/`style` spec round-trips through sample → squash → group-slice with the style block measurably decorrelated. **Done, 2026-09-17** — `src/idtb/latents/spec.py` and `ou.py`, 90/90 tests and `ruff check` green locally.

---

## Phase 3b — Spike 5: the attribute write paths (`full` + `style`)

> **This is the gate on the entire group design — and it did not fully pass.**
> Spikes 1–4 measured `write_joint_state_to_sim` and `write_root_state_to_sim` —
> §5.2's `base` group and nothing else. `cube.size` is a geometry write, `cube.hue`
> and `table.*` are material writes, `light.*` is a light-attribute write,
> `cam.jitter.*` is a per-capture camera re-aim, `exposure` is a post-process
> setting. **Result: `cube.size` gated clean; every other tested `full`/`style`
> write path did not, at a small but confirmed, reproducible non-determinism with
> no root cause found (README §7.5, §11).** `full` beyond `cube.size` and all of
> `style` are blocked pending that cause or a different render configuration; the
> `base`-only pipeline is not affected.

- [x] 🤖 **Write `spikes/spike_dynamic_attrs.py`.** Reuses `spikes/spike_api.py`'s pure layer directly — `determinism_report`, `sensitivity_report`, `Report`, the uint8-safe diffs — rather than duplicating it. Only the Isaac-layer rig is new, and it can be *smaller*: no Franka is needed, since every open question is about the cube, the lights, the table and the camera, so a minimal cube + light + camera scene boots faster and does not re-touch the already-solved Franka-asset problem.

  Three structural rules, adapted from Spike 1:
  1. **Never fail fast** — one PASS/FAIL table, one `facts.json`, every check isolated. `SimulationApp` is one-shot per process and boot is slow.
  2. **One write path per check.** A failure writing cube colour must not block finding out whether writing light intensity works — isolating *which mechanisms exist* is the whole point.
  3. **Resolve, don't guess.** Where the exact call for an attribute (scale, material colour, light intensity, exposure) is unknown, try candidates and record which answered — `spike_api.py`'s `resolve()` pattern, not a single guess that kills the boot.

  **Three questions per knob, and the order matters.** Sensitivity ranks above determinism, for a reason specific to the `style` group:

  | Order | Question | Why it ranks there |
  |---|---|---|
  | 1 | **Does the write land?** Per-attribute read-back. | The §6.3 silent-write-failure gate, on a path where `robot.data.*` does not apply. A material write that is silently ignored looks exactly like a working one. |
  | 2 | **Does varying it alone move pixels, ≫ the noise floor?** | A knob that writes cleanly and renders deterministically but changes *nothing visible* is a dead dimension. For a `style` knob that is worse than useless: the encoder scores perfect invariance for free and the headline result is an artefact of an unchecked write. |
  | 3 | **Still bitwise deterministic under `standard`?** Varying only that knob between captures. | Spike 1's verdict was measured with only physics state changing. A material or light write that re-triggers shader compilation or resets accumulation is a different question with the same acceptance test. |

  | Check | Group | What it answers |
  |---|---|---|
  | Cube scale write + read-back; does the collision volume follow the visual scale | `full` | Is `cube.size` writable per-sample at all |
  | Cube scale **sensitivity** and **determinism** under `standard` | `full` | Does the geometry path preserve Spike 1's bitwise result |
  | **Measured** size range the gripper can actually close on | `full` | `cube.size`'s radius, from the measured `[0.0, 0.04]` m finger aperture — not a guessed constant (README §5.2.2) |
  | Cube material colour write + read-back, at fixed saturation/value | `full` | Is `cube.hue` writable via the USD material API |
  | Cube hue **sensitivity** and **determinism** under `standard` | `full` | Same, for the material path |
  | Light intensity / warmth write + read-back + sensitivity + determinism | `style` | The first nuisance write path, and whether it is compatible with `standard` |
  | **Measured** intensity range that clips at neither end | `style` | A blown-out or black frame destroys injectivity for *every* dimension at once, not just this one |
  | Light direction (azimuth/elevation) write + determinism | `style` | Shadow direction is the most salient style cue, so the strongest invariance test |
  | Camera re-aimed **every capture**, not once at boot, + determinism | `style` | `set_world_poses_from_view` was only ever called once in Spike 1; per-sample jitter is a different usage pattern |
  | Exposure: does a carb / post-process setting exist and accept a value | `style` | Locates the lever, or establishes there isn't one and the handle gets dropped rather than faked |
  | **Write-order independence**: size-then-colour vs. colour-then-size → identical frame | both | USD attribute writes may cache or bind lazily; if order matters, the writer needs a fixed canonical order and that must be *known*, not assumed |
  | **Cross-talk**: writing style knobs leaves the `base` read-back unchanged | both | A camera re-aim or a scale write that perturbs physics state would silently corrupt the task latents — cheap to check, expensive to discover later |
  | Motion blur under teleport-no-step | `style` | **Expected clean FAIL, not a bug to force past.** Blur implies motion over time; this pipeline has none by design (§5.5). A recorded reason is a completed check, same as Spike 1's aliasing FAIL. |
  | Sensor noise injection point | — | Confirms synthetic noise, if ever wanted, is a `generate.py` post-process on an already-deterministic frame, never a render setting (README §5.2.3) |

  **Written blind, three choices worth flagging before the first pod run:**
  - **No `InteractiveScene`, no env-namespace templating.** Every prim (`/World/Ground`, `/World/Light`, `/World/Cube`, `/World/Camera`) sits at a fixed absolute path this script chose itself, and the `Camera` sensor is instantiated directly rather than pulled from a scene registry. Spike 1's `{ENV_REGEX_NS}` multi-env resolution existed only to make the #251 frozen-at-origin bug unmissable at metres scale — not a concern here, so the whole mechanism is sidestepped rather than reused.
  - **Raw `pxr` (`UsdGeom.XformCommonAPI`, `UsdShade.MaterialBindingAPI`, `UsdLux.LightAPI`) is the primary write mechanism**, not higher-level Isaac Lab convenience wrappers — chosen because these are stable, long-standing OpenUSD schema APIs rather than Isaac-Lab-version-specific ones, which is exactly the kind of guess that broke `FRANKA_PANDA_CFG`'s `usd_path` in Spike 1. `try_candidates()` (a generalisation of `spike_api.resolve()` to arbitrary write attempts, not just import paths) is the fallback wherever more than one call could plausibly be right — e.g. resolving which prim under the cube actually holds its shader.
  - **One reusable `run_knob_check()` procedure** runs the three-question recipe (write → read-back, sensitivity, determinism) for every `full`/`style` attribute, rather than five near-duplicate check functions — mirrors Spike 1's own "determinism/convergence as reusable functions" rule, generalised one level further since this spike repeats the same three-step recipe five times where Spike 1 only repeated two.

  Two things this file does **not** attempt, both flagged inline as the parts a real pod run should sanity-check first: the exact carb key names for `exposure`/`motion blur` (read from documentation, never run — that's exactly what "resolve, don't guess" is for), and the light-rotation Euler decomposition's angle order (correct relative to the `Gf.Rotation` math, but worth a visual check that a shadow actually moves where azimuth/elevation say it should).

  Pure-layer tests in `tests/test_spike_dynamic_attrs.py` cover `try_candidates`, `hue_to_rgb`, `azel_to_direction`, and `cube_size_radius_from_aperture` — everything below the "Isaac layer" banner, same split as Spike 1. 101/101 tests and `ruff check` green locally as of 2026-09-17.

- [x] 🧑 **Run it on the pod, paste the whole table.** Took ~7 iterations, more than Spike 1's 2–4 — see the fix log below.
- [x] 🤖 **Fix and re-run** until every check has a verdict. Done, but not clean: `cube.size` and `exposure`'s lever are confirmed good; `cube.hue`/`light.*`/`cam.jitter` have a **recorded FAIL whose reason was not found** despite four targeted, each-falsified fix attempts (render depth 1→8, a discard-first warm-up render, preset applied once vs. per-check, widened light angular size) — see README §7.5 for the full verdict and §11 for the risk. `--save-frames` + a `diff_summary()` diagnostic (added mid-investigation) localized the non-determinism to ~15% of the frame on the varied object's own surface, ruling out a uniform accumulation leak or a background/shadow-edge artefact, without finding the actual cause.
- [x] 🧑 **While the pod was up anyway, closed out Spike 1's two loose ends:**
  - `spike_api.py --num-envs {1,8,32}` — throughput table filled in, but also surfaced a **new, separate finding**: `sensitivity_arm_and_cube` intermittently fails at `--num-envs 1` and `32` with the noise floor exploding to ~75 mad (vs. the verified 0.0) — confirmed *flaky*, not deterministic, by an identical repeat run passing cleanly. Not yet root-caused; a warm-up-capture mitigation was added to `check_sensitivity` but not verified to fix it (untested after the fix landed).
  - `spike_api.py --save-frames` — added and run; frames pulled via base64 for visual inspection.
- [x] 🤖 **README checkpoint 3** — §7.5 rewritten with the verdicts; §3.2's two relevant deferred rows resolved; §11's risk row updated to Critical with the confirmed finding.

---

## Phase 3c — Spike 5 diagnosis, round 2 (external research lead, untested)

> `docs/research_task_full_style_determinism.md` was sent out as a research
> question after the four falsified fixes above; `docs/answer.md` is the reply.
> **This is a set of new hypotheses and a specific experiment order, not a
> fix** — nothing below has been run. Per the no-premature-pins rule, none of
> these carb keys go into `standard` or into code until a pod run confirms
> them; they're a checklist, not a decision.

**What's actually new versus the four already-falsified attempts** (render
depth 1→8, discard-first warm-up, preset-once-vs-per-check, wider light
angular size): those varied *when*/*how often* the same four carb keys were
applied. Nothing before touched the path tracer's **caches** or its **AA
jitter**, and nothing checked whether the four carb keys are even the
attribute the camera's RenderProduct reads. Two of the falsified results turn
out to *corroborate* the new leading theory rather than sit unexplained:
- the warm-up-render regression of `cube.size` (`0.0` → `~1.87`) is exactly
  what a cross-frame-cache theory predicts (a warm-up hands the cube a warm
  cache too), where a "needs more convergence" theory predicted the opposite.
- the widened-light-angle result, which moved 3 of 5 knobs but left
  `light.azimuth_elevation` and `cam.jitter` bit-for-bit unchanged, is
  consistent with a cache/change-detection mechanism that isn't uniformly
  triggered by shadow softness.

**One flagged concern that does *not* apply to us:** the reply worries that
`cube.size`'s clean `0.0` could be a Fabric-override no-op (the kinematic
cube's transform might be written to Fabric while the USD scale op is
ignored). It couldn't see our code. But `run_knob_check`
(`spikes/spike_dynamic_attrs.py:675`) already runs `sensitivity_report` on
every knob, `cube.size` included, and gates the check on `all_responsive`
*before* the determinism check even runs — Spike 5's own three-question
ordering (§7.5, "sensitivity ranks above determinism") was designed for
precisely this failure mode. `cube.size` could not have reported "confirmed
good" if it were a no-op. Worth re-confirming the actual mad number on the
next run since it was never written down, but there's no reason to doubt the
clean verdict.

**Next pod session — run in this order** (each falsifies a specific claim
before spending time on the next):

- [ ] 🧑🤖 **A. Audit what's actually configured (~10 min).** Dump every
  `omni:rtx:*` attribute on the camera's RenderProduct prim; compare against
  the four carb keys `standard` sets. Isaac Sim 6.x's per-product render
  settings (`omni:rtx:rendermode`, `omni:rtx:pt:samplesPerPixel`, etc.) are a
  separate mechanism from the global/deprecated carb keys, and reading a carb
  key back only proves carb stored it — not that the RenderProduct used it.
  **Falsifies:** "the `standard` preset is actually live on this render path."
  If it isn't, everything measured under §7.5 needs re-interpreting, not just
  patching.
- [ ] 🧑🤖 **B. Disable every path-tracer cache and stochastic filter, re-run
  the five blocked knobs.** New settings, none tried before:
  `/rtx/pathtracing/cached/enabled=False`,
  `/rtx/pathtracing/lightcache/cached/enabled=False`,
  `/rtx/pathtracing/adaptiveSampling/enabled=False`,
  `/rtx/pathtracing/fireflyFilter/enabled=False`,
  `/rtx/pathtracing/aa/op=0`, `/rtx/pathtracing/aa/filterRadius=0.0` — via the
  per-product attributes instead of carb if A shows carb isn't authoritative.
  (`antialiasing_mode="Off"` in `RenderCfg` is an Isaac Lab **Real-Time**-mode
  call and very likely does nothing under `PathTracing`, where AA is the
  `/rtx/pathtracing/aa/*` jittered pixel filter — sub-pixel jitter may have
  been on the whole time.) **Prediction:** the five nonzero `back_to_back_mad`
  values drop to `0.0` or by an order of magnitude. If none move at all, the
  cache/AA-jitter theory is dead and C/D are next.
- [ ] 🧑🤖 **C. Force cold accumulation, matching our exact capture pattern.**
  Set `/rtx/resetPtAccumOnAnimTimeChange=True`; call
  `RenderContext.reset_transform_cadence()` (or
  `reset_scene_state_cadence()`) before each capture; try
  `omni.kit.app.get_app().update()` in place of `sim.render()`. This is the
  **version-matched lead**: IsaacLab#6609, reported against our exact
  Isaac Lab build, is that `RenderContext` publishes renderer scene state at
  most once per physics-step count, and a caller that calls `forward()`
  without advancing that count — which is precisely §5.5's zero-`sim.step()`
  capture loop — can skip publication entirely.
- [ ] 🧑🤖 **D. Escape hatch: Minimal render mode.** If B and C both fail to
  move the numbers, re-run the full Spike 5 table under Isaac's `Minimal`
  render mode (hard shadows, first distant light only, no Monte Carlo state,
  no caches, no accumulation) instead of `PathTracing`. Loses little for a
  128×128 cube-on-plane scene; the material/light/camera knobs still change
  the image. This changes `standard` itself, so it needs its own §7.1 gate
  re-run against `base`, not just the five blocked knobs.
- [ ] 🤖 **If none of A–D land bitwise-clean:** record a fallback acceptance
  criterion instead of giving up the render path — render each state twice in
  shuffled order and test whether a linear probe on `b1 − b2` predicts the
  varied factor above chance. This is weaker than bitwise equality but matches
  what §7.1 actually needs (no *information* about the latent in the
  residual, not literal bit-identity). Record as a Decision Register candidate
  in README §3.1, not an adopted change — a real decision needs the pod data
  from A–D first.
- [ ] 🤖 **README checkpoint 4** — whichever of A–D resolves it (or the
  fallback criterion, if adopted) gets written into §7.3's `standard` preset,
  §7.5's verdict, and §11's risk row. If A–D all fail, §7.5 records that too,
  with the fallback criterion as the new plan of record for `full`/`style`.

---

## Phase 4 — Build against verified reality

> **Scope note on "verified": Spike 1/2 only cover physics-state writes** (joint angles, rigid-body pose) — README §5.2's `base` group. Every `full` and `style` latent (cube size and colour; lighting, per-sample camera jitter, exposure, materials) uses a write path that hasn't been touched. **Phase 3b is the gate**: don't wire any of those into `SceneBackend`/`generate.py` as a first-class latent before that spike has a verdict. The `base`-only pipeline does not wait on it.

- [ ] 🤖 **`SceneBackend` protocol** — now, not before. The risk was never the method *names*; it was granularity and semantics, which the spike just settled.
  - Batched from the start (`B=1` valid) — absorbs the TiledCamera question either way
  - Segmentation remapped to **our own stable ids** — Isaac's annotator ids aren't stable across runs, which would silently turn the visibility column into noise
  - `write_state(phi)` dispatches **by write path, not by group** (README §6.3): joint write, root-state write, attribute write. Three paths, three read-backs — `robot.data.*` does not cover a material or light write, so one blanket assertion would pass while a colour write was being silently dropped.
  - A backend declares which roles it supports; an active spec naming a role the backend can't write must fail loudly at bind time, not render a scene missing a latent
- [ ] 🤖 **`MockSceneBackend`** — numpy only, built to the captured conventions. Anti-aliased analytic shapes (hard edges would make two nearby latents render identically and fail the injectivity test on the one backend where it must pass)
- [ ] 🤖 **Determinism/read-back gates as library code in `src/`**, not test-only — README §7.1 says the gate runs before every dataset generation, so `generate.py` and the tests must call the same function
  - **Acceptance is bitwise, not a threshold.** A temporal denoiser leak sits at ~0.3/255 and passes any sane threshold — while being exactly the structure a contrastive encoder is trained to find. A threshold doesn't weaken this gate, it disables it.
- [ ] 🤖 **Negative controls** — inject faults into the mock (jitter, temporal leak, frozen write simulating #251, aliased buffers) and assert the gates **fire**
  - Without these, a gate that passes is a detector nobody has ever seen detect anything
  - **Add one for the group design: a disconnected `style` knob.** A mock that accepts the write and renders identically must fail the §10.1 style-sensitivity gate. This is the failure mode that would otherwise report perfect encoder invariance and be believed.
- [ ] 🤖 **Tier-1 contract suite** — one set of tests parametrized over both backends
  - Parametrized over **group configuration** too (`base`, `base+style`, `full`, `full+style`), so the group plumbing is exercised on both backends rather than only in tier 0
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
./isaaclab.sh -p spikes/spike_dynamic_attrs.py --out /idtb/data/spike_attrs   # Phase 3b
./isaaclab.sh -p -m pytest --isaac-mode=require
```
`--isaac-mode=require` matters: without it, a broken container that can't import `isaaclab` silently deselects the whole Isaac tier and reports green.

**Done means:** all four §7.2 spikes answered and recorded in README §3.1; **every `full`/`style` knob in §5.2 either has a measured range and a confirmed write call, or is dropped with the reason recorded** (§7.5); the pure layer carries the group split with tier-0 tests; protocol backed by running code on both sides; README describes nothing in this milestone as unknown.

---

## Known limit — don't treat the spike verdict as final

Denoiser and accumulation behaviour is **scene-, material- and light-dependent**. The spike uses a shipped Franka + cuboid + default light; scene v1 adds PBR materials, HDRI, area lights, real resolution. The determinism verdict may not transfer and **N almost certainly won't** (it'll be larger). That's why the checks are written as reusable functions and N is recorded as a procedure. Re-gate on scene v1 before generating any dataset — next milestone, but the functions must exist now or it's a rewrite.

---

## Deferred

- USD scene authoring, custom Dockerfile, dataset storage format (prefer **lossless** — compression artifacts would be indistinguishable from a real identifiability effect)
- LeJEPA/SIGReg trainer, behind an interface. Note: the authors published code at `github.com/klindtlab/lejepa-identifiability`, which makes "use theirs" a live option when we get there.
- Everything in README Phases 5–10
