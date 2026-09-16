# Isaac Sim Identifiability Testbed

*Implementation plan for a LeJEPA linear-identifiability pipeline — photorealistic tabletop manipulation, OU-sampled positive pairs, remote GPU backend. Extensible from single-cube tabletop to full room-scale scenes.*

> **Status: living plan.** This document is the project's plan of record. It is kept correct as work proceeds — when a spike answers a question, a decision is made, or an API turns out to differ from what is written here, this file is updated in the same change. Anything not yet known is recorded as a deferred decision in §3 rather than guessed at.

---

## 1. Executive Summary

The goal is a simulator that produces photorealistic image pairs *(x, x′)* whose ground-truth latents *(z, z′)* are drawn from a known Gaussian process with Ornstein–Uhlenbeck correlation, so that LeJEPA's linear-identifiability claim (Thm. 1–3 of Klindt, LeCun & Balestriero) can be tested on realistic observations rather than on 2D toy mixings or the low-fidelity DMC Reacher.

**Recommendation: Isaac Sim + Isaac Lab, running headless in a container on a rented RT-core GPU.** Isaac Sim satisfies every hard requirement: it exposes direct state writes (teleport-then-render, no policy rollout needed), it has an RTX path-tracing renderer for genuine photorealism, and its USD scene graph is built for exactly the incremental scene-growth path we want (tabletop → room → many objects). The *version* of Isaac Sim is deliberately not pinned here — see §3.

> ### ⚠ Two infrastructure facts that constrain everything
>
> **1. RT cores are mandatory.** Isaac Sim's requirements state that GPUs without RT Cores — explicitly A100 and H100 — are **not supported**. These are the default "big GPU" options on most cloud providers. An RTX-class GPU is required. Budget and availability planning starts from this constraint, not from raw FLOPs.
>
> **2. There is no local development loop.** Isaac Sim supports Ubuntu 22.04/24.04 and Windows 11. **macOS is not supported**, the container is Linux-only, and there is **no CPU-only or software-rendering fallback**. On an Apple-silicon development machine, no Isaac code can execute at all. Every line of Isaac-touching code is written blind locally and exercised remotely. This is not a minor inconvenience — it is the single strongest argument for the backend seam in §4.3, and it should be treated as a first-class design driver rather than an operational detail.

Four decisions matter more than the rest. Three are about protecting the *mathematical* validity of the experiment; the fourth is about being able to work at all.

1. **Latent parameterisation via an absorbed squashing map.** The theory requires *z* to be exactly Gaussian with unbounded support, but joints and table surfaces are bounded. The fix is to keep *z ~ N(0, I)* genuinely unbounded and push a fixed deterministic squashing function into the mixing map *g*. Since the theory permits *g* to be an arbitrary nonlinear measurable map, this is free — and it eliminates the joint-limit wrapping artefact that degraded the paper's own Reacher results (their Table 2).
2. **Renderer determinism.** The theory assumes *x = g(z)* is a deterministic function. A path tracer with a moving random seed or a temporal denoiser makes *g* stochastic and history-dependent, which silently invalidates the setup. **In current Isaac Sim releases this is the default behaviour, not an opt-in risk** — see §7.1. It must be engineered explicitly and verified numerically.
3. **Injectivity of *g* under occlusion and object symmetry.** If the arm hides the cube, or if a symmetric cube looks identical at 0° and 90° yaw, then distinct *z* map to identical *x* and no encoder can recover them. This is the deepest scientific risk in the project and needs design mitigation from day one.
4. **A mockable backend seam.** Because nothing Isaac-related runs locally, the pipeline is built against a narrow `SceneBackend` protocol with two implementations: the real Isaac backend and an analytic mock. The mock gives a local test loop, and — more importantly — gives the correctness gates a known-good reference, so that a gate which passes is actually evidence.

**Effort:** roughly **2 weeks** to a first defensible identifiability number on the single-cube scene, and **4–5 weeks** to a fully extensible room-scale pipeline with a full sweep. Detailed phasing in §9. These are estimates, not commitments.

---

## 2. Requirements Traceability

| Requirement | How Isaac Sim satisfies it | Residual work / caveat |
|---|---|---|
| Latents of a simple scene (cube pick-and-place, fixed arm) | Franka Panda USD ships with Isaac Sim; cube is a primitive rigid body. Full joint and pose state is readable and writable. | We must *define* what counts as the latent vector — a modelling decision, not a default. See §5. |
| Extensible: load scenes, add objects later | USD composition (references, sublayers, payloads) is designed for this. Room-scale assets and the Omniverse asset library plug in directly. | Latent dimension *n* grows with each object; encoder output dim *m* must track it. The paper flags *m ≠ n* as an open problem (their §7). |
| Highly realistic rendering from the start | RTX path tracing with physically based materials, HDRI domes, area lights. This is the strongest renderer of any robotics sim. | Path tracing costs far more per frame than rasterisation, and the realtime modes are themselves temporally accumulating. Determinism must be forced. See §7. |
| OU sampling capability | `write_joint_state_to_sim` and `write_root_pose_to_sim` allow arbitrary state teleport without physics rollout. Render immediately after. | Need a collision/validity policy for teleported states, and isotropic ρ across all latent dimensions. |
| Rented GPU backend | Official NGC container `nvcr.io/nvidia/isaac-sim` runs headless on Linux; providers such as RunPod support custom images and persistent network volumes. | RT-core GPU mandatory; host driver version must match the chosen release; asset cache must be persisted or every pod start re-downloads gigabytes. |
| Cost | Free for internal and commercial R&D under NVIDIA's licence FAQ. Source is Apache 2.0. | Paid NVIDIA AI Enterprise licence only if Isaac Sim is *redistributed* or delivered as a service to third parties. Publishing research and datasets does not trigger this. |

---

## 3. Decision Register

The purpose of this section is to keep the plan honest about the difference between *decided*, *provisional*, and *unknown*. Nothing below the first table is pinned anywhere else in this document, in the Dockerfile, or in configs. Where an earlier draft of this plan pinned a version, a driver, a GPU model or a render mode, that pin has been removed and replaced by an entry here.

### 3.1 Decided — these shape the code and are not expected to change

| Decision | Rationale |
|---|---|
| Latents are exactly Gaussian; boundedness handled by an absorbed `tanh` squash | §5.1. Preserves Theorem 1's premise and injectivity of *g*. |
| A single scalar ρ shared across all latent dimensions | §5.4. Required for simultaneous (non-sequential) identifiability, per the paper's Appendix F. |
| State is written by teleport; no policy rollout, no physics settling by default | §5.5. Keeps *g* deterministic and injective. |
| Isaac Lab is used for the asset/state layer; raw `carb` settings for the render layer | §4.4. Isaac Lab gives batched tensorised state writes with read-back; it does not expose path-tracing controls. |
| The pipeline is built against a `SceneBackend` protocol with a mock implementation | §4.3. Forced by the absence of any local Isaac runtime. |
| Tests accompany every module; the §10.1 correctness gates are executable tests | §10.3. |
| One installable package `src/idtb/`, never top-level `sim`/`gen`/`eval` | Those names collide with Kit extensions on Isaac's `sys.path`, and `eval` shadows a builtin. §4.6. |
| Handles are keyed by **role** (`arm.j0`, `cube.x`), not by Franka joint name | §6.2. A spec keyed on asset-specific names cannot be shared with the mock, which makes the tier-1 contract suite impossible. |
| Module-level `isaaclab`/`omni`/`carb`/`pxr` imports are banned mechanically | §4.2. Ruff TID253 catches the syntax; `tests/test_import_guard.py` imports every module with those roots blocked, catching the rest. A violation is otherwise only discoverable on the pod. |
| Isaac is not pip-installable into our environment, by construction | A local install would misrepresent what is actually runnable, and the constraint in §1 is the whole reason the seam exists. |
| **Isaac Sim 6.0.1, staying on the image already pulled** | **Reversed from 6.0.0 on 2026-09-16.** Two recreates aimed at landing a 580-branch host for 6.0.0 instead produced two different 570-branch hosts (§8.1) — the version was being chased by the infrastructure rather than the other way round. The image already on the pod, `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`, is Isaac Sim 6.0.1: it keeps the `TiledCamera` fix §3.4's whole argument turns on (fixed in 6.0, present in every 6.x since), needs no further pull, and is one patch release, not a minor, away from what was actually spiked against. |
| **The driver requirement moved to 595.58.03 — and the host measured is below it, accepted knowingly** | 6.0.1 (and 6.1.0) test at 595.58.03, a step up from 6.0.0's 580.95.05. The host surveyed 2026-09-16 is **570.195.03** — now two branches short, not one. This is accepted, not overlooked: NVIDIA's requirements page says only "tested on" from 5.1.0 onward, not a floor, and Kit's own hard refusal sits far lower (rejects only < 535.129, per [IsaacSim #4244](https://github.com/isaac-sim/IsaacLab/issues/4244)). An untested driver **starts**; whatever it changes shows up only in what it renders — which is exactly what §7.1 measures, so the spike (Phase 3) is the check that actually closes this, not another pod recreate. `infra/preflight.sh` still fails on it by default (`IDTB_MIN_DRIVER`) so the gap stays visible on every session rather than being silently normalized; override deliberately when running anyway. If a future recreate (for unrelated reasons) happens to land on 595+, that closes the gap for free — but it is not being chased. |
| **Isaac Lab 3.0.0-beta2.patch1** | Its `docker/.env.base` pins `ISAACSIM_VERSION=6.0.1`. Same beta line as `beta2`, one patch release later — the exposure accepted in §3.4 (breaking changes before 3.0 stable) is unchanged in kind, not compounded. |
| **Container image `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`; no custom Dockerfile** | §8.3. This is the *sibling* tag flagged as "the wrong one" in an earlier draft of this table, back when the target was 6.0.0 specifically and the plan was to recreate the pod onto plain `beta2`. That plan is dropped (see the row above): the pod already runs `-post1`, plain `beta2` was never pulled, and re-pulling now would trade a known image for an unverified one to chase a driver branch two recreates have failed to land. |
| **Python 3.12** | Forced by Isaac Sim 6.x: the `isaacsim` 6.0.0.0 wheels declare `requires_python == 3.12.*` (5.1.0.0 was `3.11.*`). **Not independently re-verified for 6.0.1** — inferred from 6.0.1 being a patch release, which does not typically move a `requires_python` constraint; confirm against the pod's own interpreter (§8.1's "isaac sim version" survey line now measures it) rather than trust this by extension. Recorded, not pinned to an equality: `pyproject.toml` sets `requires-python = ">=3.12"` so the pure tier-0 layers keep running on the local interpreter, and nothing selects a Python version independently of the Isaac release. |
| **GPU: GeForce RTX 4090, 24 GB, compute capability 8.9** | Survey verdict (§8.1), and the advertised part was allocated on both surveyed hosts. RT cores present; above 6.0.0's minimum spec of RTX 4080 / 16 GB. The *rest* of the host — OS, glibc, core count, driver — is not a property of this decision and moves with every pod recreate; §8.1 records what each one measured. |
| **The network volume mounts at `/idtb`, never at `/workspace`** | The image unpacks Isaac Lab into `/workspace/isaaclab` (`DOCKER_ISAACLAB_PATH`). A volume mounted at `/workspace` shadows it, and the symptom is a missing `isaaclab.sh` — which reads as a broken image, not a mount problem. Both pod scripts refuse `/workspace` outright. |
| **The container runs as uid/gid 1000 — not root. Key on the uid, never the name** | `Dockerfile.base` does its setup as root and ends on `USER isaaclab`. A provider-supplied volume arrives root-owned, so uid 1000 cannot write it, and the image ships no `sudo`. The ownership fix has to happen from a root shell *before* the Isaac container needs the volume. `ACCEPT_EULA=Y` is required as a pod env var. The survey measured the account as **`ubuntu`**, not `isaaclab` — same uid, different name, because the 3.0 image's Ubuntu 24.04 base already ships a user at 1000. Both pod scripts use `id -u`, so the discrepancy is inert; a script that had matched on the username would not be. |

### 3.2 Deferred — recorded here so that de-pinning does not lose the question

| Open question | What decides it | When |
|---|---|---|
| **Render mode and preset definitions** | Spike 1 (§7.2): which modes can be made deterministic, and at what cost. The preset table in §7.3 is a set of candidates, not a configuration. | Phase 1 |
| **Whether a physics step is required before rendering** | Spike 2 (§4.5). If a step turns out to be unavoidable, §5.5's "accept interpenetration" recommendation must be revisited. | Phase 1 |
| **Accumulation depth N** (render calls / `rt_subframes` / SPP to convergence) | Spike 3 (§7.2). This is the per-sample cost multiplier and therefore the entire GPU-hour budget. | Phase 1 |
| **`TiledCamera` vs. `Camera`** | Spike 4 (§7.2). Worth an order of magnitude in throughput. | Phase 1 |
| **Dataset storage format** (HDF5 / WebDataset / other) | Write behind a small writer interface; decide once the per-sample payload size and the training-side read pattern are known. | Phase 5 |
| **Image resolution** | Candidates 128×128 and 224×224. Cheap to ablate; treat as an experimental variable rather than a configuration decision. | Phase 8 |

### 3.3 Provisional — chosen to make progress, expected to be revised

- The Stage-1 latent assignment in §5.2 (which four arm joints, which squash radii). Provisional until the Franka asset actually loads and joint limits are read back from it.
- ρ = 0.95 as the first generation configuration (§9, Phase 6). A starting point for the sweep, not a finding.
- Cost figures in §8.4. Indicative only and known to move; verify at provisioning time.

### 3.4 Why a beta simulator, when §3.2 said 5.1 was the conservative choice

That earlier judgement was made before reading the issue tracker, and the evidence reversed it.

- **[IsaacSim #367](https://github.com/isaac-sim/IsaacSim/issues/367) is Spike 4.** `TiledCamera` returns broken tiles under ray tracing — correct total resolution, but the per-camera tiles are gone. An NVIDIA engineer reproduced it and closed the issue with *"can confirm the issue in 5.1, it is however fixed in 6.0"*. Path tracing is unaffected. TiledCamera is worth an order of magnitude in throughput, and `standard` — the preset intended for the ~100k-pair dataset — is precisely where ray tracing would be used.
- **5.1.0 is terminal.** Released 2025-10-21, with no 5.1.x patch line in the eleven months since. That bug will never be fixed there.
- **6.0 changes the renderer** — RT 2.0 by default, Kit SDK 109.0.2 — and fixes a Replicator path where async rendering re-enabled on timeline pause/stop while annotators were attached, skipping writer frames. Both sit directly on top of §7's determinism question.
- **Sequencing is the real argument.** The spike's entire value is measuring the renderer we ship on. Spiking against 5.1 and later moving to 6.x discards the determinism verdict and N — which §7.4's closing note already warns do not transfer.

What we are accepting: breaking changes between beta2 and 3.0 stable. The exposure is bounded by how little of Isaac Lab we use — `FRANKA_PANDA_CFG`, batched state writes with read-back, and a camera wrapper — all of it behind the §4.3 `SceneBackend` seam with a contract suite on both sides. That seam was built for the no-local-runtime problem; it absorbs this too.

**3.4.1 — Why 6.0.1 (the pulled image), not 6.0.0 (the originally resolved one)**

The first version of this decision chased the driver: read 580.178.04, resolve 6.0.0, recreate the pod expecting to land on another 580-branch host. Two recreates later, both landed on 570-branch hosts instead (§8.1) — the provider's allocation, not this project, decides what driver a pod gets, and there is no way to reserve a specific branch, only to keep re-rolling and hoping. Meanwhile the pod that came up was already running `beta2-post1` (Isaac Sim 6.0.1), pulled before the tag mismatch was even noticed, and it works: `bootstrap.sh` completed against it (§8.3).

Re-pulling `beta2` to get back to exactly-6.0.0 would trade a known-working image for an unverified one, in service of a driver number that two attempts have failed to control, for a difference (6.0.0 vs 6.0.1, one patch release) smaller than the difference already accepted between stable 5.1.0 and the beta line (§3.4 above). So: stop chasing the driver, keep the image that is actually on the pod, and let the Phase 3 spike — which measures the renderer directly — be the thing that closes the open question, rather than a driver-version proxy for it.

**Consequence to keep in view:** the driver number is now informational, not a gate this project can act on by recreating pods. `infra/preflight.sh` still reports the gap against 595.58.03 by default, and still fails on it unless overridden — not because another recreate is planned, but so the gap stays visible in every session's output rather than being quietly forgotten. If a future recreate for an unrelated reason happens to land on 595+, good; it is not the plan.

**The escape hatch, still recorded in case both the version *and* the risk acceptance above turn out to be wrong on the pod:** Isaac Sim **5.0.0** + Isaac Lab **2.2.0** (`nvcr.io/nvidia/isaac-lab:2.2.0`, Python **3.11**), documented driver 535.216.01, independently confirmed running on the 570 branch. A real downgrade, not a lateral move: *older* than 5.1.0, so it loses the `TiledCamera` fix this whole section turns on, and Python 3.11 would move `requires-python`. Take it only with that understood, and re-run the §7.2 spikes on whatever is chosen — the determinism verdict and N do not transfer across a renderer change.

---

## 4. Architecture

### 4.1 Component stack

```
┌──────────────────────────────────────────────────────────────┐
│  Analysis layer  (any GPU / local)                           │
│   LeJEPA encoder training · SIGReg · R² · orthogonality      │
│   ε, δ, bound D + (ε+D)²  ·  planning eval                   │
└──────────────────────────▲───────────────────────────────────┘
                           │  sharded dataset
                           │  (x, x′, z, z′, ρ, diagnostics, metadata)
┌──────────────────────────┴───────────────────────────────────┐
│  Dataset generation  (remote, RT-core GPU)                   │
│   ┌────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│   │ OU sampler │──▶│ Latent→State │──▶│  SceneBackend    │   │
│   │  z, z′     │   │  map  φ      │   │  (protocol)      │   │
│   └────────────┘   └──────────────┘   └────────┬─────────┘   │
│      pure python       pure python             │             │
│                            ┌───────────────────┴──────────┐  │
│                            ▼                              ▼  │
│                   IsaacSceneBackend              MockSceneBackend
│                   (remote, GPU)                  (local, analytic)
└──────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┴───────────────────────────────────┐
│  Scene layer  (USD)                                          │
│   stage_v1: table + Franka + cube                            │
│   stage_v2: + room shell                                     │
│   stage_v3: + N distractor & manipuland objects              │
└──────────────────────────────────────────────────────────────┘
```

Two architectural properties matter:

- **The scene layer is swappable without touching the sampler.** The latent-to-state map *φ* is a registry of named handles (an articulation's joint, a rigid body's root pose) with per-handle squashing ranges. Adding an object means appending entries to that registry and incrementing *n*; the OU sampler, the writer and the renderer are unchanged.
- **The backend is swappable without touching anything above it.** Everything above the `SceneBackend` line is pure Python and runs locally.

### 4.2 The constraint that dictates the module layout

`SimulationApp` must be constructed **before any `isaaclab.*` or `omni.*` import**. It is process-global, one-shot, and cannot be restarted within a process. Three consequences shape every file:

1. **No Isaac imports at module top level, anywhere in reusable code.** Sim-touching modules import Isaac *inside* functions. A top-level `import isaaclab` in `src/idtb/sim/writer.py` would kill `pytest` at collection time on a developer machine — where, per §1, Isaac cannot be installed at all.
2. **Entry points are scripts; libraries are pure.** `src/idtb/gen/generate.py` boots the app and *then* imports. `src/idtb/latents/*` and `src/idtb/analysis/*` never touch Isaac.
3. **One app per process.** Isaac-dependent tests run in a single session under Isaac's own interpreter, with a session-scoped fixture owning the app.

### 4.3 The backend seam

Defined in `src/idtb/sim/backend.py` — pure Python, zero Isaac imports:

```python
class SceneBackend(Protocol):
    def handles(self) -> Mapping[str, HandleInfo]: ...
    def write_state(self, phi: Tensor) -> None:        ...  # [B, n], teleport
    def read_state(self) -> Tensor:                    ...  # [B, n], for the read-back gate
    def render(self, sample_idx: int) -> dict:         ...  # {cam_id: {"rgb":…, "seg":…}}
    def diagnostics(self) -> dict:                     ...  # visibility, collision
    def close(self) -> None:                           ...
```

- **`IsaacSceneBackend`** — real, lazy-imports Isaac, runs remotely.
- **`MockSceneBackend`** — an analytic toy renderer (project handles to 2D, draw sprites). Deterministic and injective *by construction*.

The mock is not a convenience. It does two jobs:

1. It gives a local development and test loop for the OU sampler, the squash, the shard writer, the generation driver, the metrics and the LeJEPA training code — which is most of the project.
2. It gives the correctness gates a **known-good reference**. A determinism gate that has never been observed to pass on something that should pass is not evidence. Running the same contract suite against the mock (must pass) and against Isaac (must also pass) is what makes the Isaac result meaningful.

### 4.4 The Isaac API surface we actually need

Deliberately small. This is the complete list; anything outside it is out of scope.

**Boot** — `src/idtb/sim/app.py`, the only module permitted to do this
- `isaaclab.app.AppLauncher`, `AppLauncher.add_app_launcher_args(parser)`, `app_launcher.app`, `simulation_app.close()`
- Flags: `headless=True`, `enable_cameras=True` (required for sensor rendering in standalone scripts), `renderer=…`

**Simulation context**
- `isaaclab.sim.SimulationCfg`, `RenderCfg`, `SimulationContext`
- `sim.reset()`, `sim.render()`, `sim.step(render=False)`, `sim.forward()`

**Scene and assets**
- `isaaclab.scene.InteractiveScene`, `InteractiveSceneCfg`; `scene["robot"]`, `scene.env_origins`, `scene.write_data_to_sim()`, `scene.update(dt)`
- `isaaclab.assets.ArticulationCfg`, `RigidObjectCfg`, `AssetBaseCfg`
- `isaaclab.sim` spawners: `UsdFileCfg`, `CuboidCfg`, `GroundPlaneCfg`, `DomeLightCfg`

**State write** — signatures verified against `articulation.py` on `main`
- `robot.write_joint_state_to_sim(position, velocity, joint_ids=None, env_ids=None)` — shapes `(len(env_ids), len(joint_ids))`
- `robot.write_root_pose_to_sim(root_pose, env_ids=None)` — `(B, 7)`, position + quaternion **(w, x, y, z)**, **world frame**
- `obj.write_root_state_to_sim(root_state, env_ids=None)` — `(B, 13)`
- `obj.write_root_velocity_to_sim(...)` — zeroed on every write
- Index resolution: `robot.find_joints(["panda_joint1", …])`. **Joint indices are never hardcoded.**

**Read-back** — the gate
- `robot.data.joint_pos`, `cube.data.root_pos_w`, `cube.data.root_quat_w`, `robot.data.default_joint_pos`, `cube.data.default_root_state`

**Cameras**
- `isaaclab.sensors.CameraCfg` / `Camera` / `TiledCamera`, `CameraCfg.OffsetCfg`, `sim_utils.PinholeCameraCfg`
- `camera.update(dt, force_recompute=True)`, `camera.data.output["rgb"]` → `(B, H, W, 3) uint8`
- `data_types`: `"rgb"`, `"semantic_segmentation"`, `"instance_segmentation_fast"`, `"distance_to_image_plane"`
- `camera.data.intrinsic_matrices` — logged to dataset metadata

**Render control** — drops below Isaac Lab into `carb`, contained entirely in `src/idtb/sim/render.py`
- `carb.settings.get_settings().set("/rtx/rendermode", …)`
- `/rtx/pathtracing/spp`, `/rtx/pathtracing/totalSpp`, `/rtx/post/dlss/execMode`
- `RenderCfg(antialiasing_mode=…, enable_dl_denoiser=…, samples_per_pixel=…, carb_settings={…})`
- Possibly `omni.replicator.core`: `rep.orchestrator.set_capture_on_play(False)`, `rep.orchestrator.step(delta_time=0.0, rt_subframes=N)`, `rep.set_global_seed(seed)`

> **Why both Isaac Lab and raw `carb`?** Isaac Lab provides exactly the thing this project needs most — batched, tensorised state teleport with read-back — which would be substantial work to reimplement against raw USD/PhysX. But `RenderCfg` exposes no path-tracing SPP or accumulation control, so the render layer must drop to raw `carb` settings and possibly the Replicator orchestrator. The split is clean and is confined to one module.

### 4.5 The capture sequence

This ordering is the part that must be exactly right.

```
write_joint_state_to_sim / write_root_pose_to_sim
  → scene.write_data_to_sim()
  → sim.forward()                        # flush USD/Fabric, no time advance
  → assert read_state() ≈ phi            # hard gate
  → camera.update(dt=0.0, force_recompute=True)
  → N × sim.render()                     # N = accumulation convergence depth
  → read camera.data.output[...]
```

Two unknowns, both Phase-1 spikes (§7.2): whether `sim.forward()` alone suffices with no `sim.step()` at all, and what N must be for the chosen preset to converge to a fixed point.

### 4.6 Repository layout

```
lejepa_identifiability/
├── src/idtb/                        # one package: top-level `sim`/`gen`/`eval`
│   │                                # would collide with Kit extensions on
│   │                                # Isaac's sys.path, and `eval` is a builtin
│   ├── latents/                     # pure python, no Isaac          ✅ built
│   │   ├── spec.py                  # Handle, LatentSpec: roles, squash φ
│   │   └── ou.py                    # OU pair sampler
│   ├── sim/                         # backend.py (protocol) · mock.py · app.py
│   │                                # scene.py · writer.py · render.py
│   ├── gen/                         # generate.py — dataset driver, sharded
│   └── analysis/                    # LeJEPA training · metrics
├── spikes/
│   └── spike_api.py                 # one standalone Isaac contact script
├── tests/
│   ├── test_ou.py, test_spec.py     # tier 0 — pure                  ✅ built
│   ├── test_import_guard.py         # tier 0 — §4.2 enforced         ✅ built
│   ├── contract/                    # tier 1 — parametrized over backend
│   └── isaac/                       # tier 2 — pod only
├── scenes/                          # stage_v1_tabletop.usd, materials/
├── configs/                         # ρ, λ, n, render preset
├── infra/
│   ├── preflight.sh                 # read-only pod survey           ✅ built
│   └── bootstrap.sh                 # caches onto the volume, checkout ✅ built
└── docs/PLAN.md                     # ordered checklist for the current milestone
```

No Dockerfile: NVIDIA ships a prebuilt headless `isaac-lab` image on NGC, so the
image is pulled and our code lives on the volume (§8.3).

### 4.7 Local development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # tier 0; the isaac marker is deselected by default
.venv/bin/ruff check .    # includes TID253, the module-level Isaac import ban
```

Both are run by hand before a commit — there is no hosted CI (§10.3).

The Isaac runtime is Python 3.12 (§3.1), but that is the pod's constraint, not
ours: `pyproject.toml` asks only for `>=3.12` so the tier-0 layer keeps running on
whatever local interpreter is to hand, and nothing selects a Python version
independently of the Isaac Sim release. Torch installs on macOS; Isaac does not,
and deliberately cannot — see §1.

---

## 5. Latent Design — The Scientific Core

This section is where the experiment is won or lost. Everything downstream is engineering.

### 5.1 The bounded-support problem and its clean solution

Theorem 1 requires *z ~ N(0, Iₙ)* exactly. But a Franka joint lives in a bounded interval, and a cube must stay on the table. The naive approaches all damage the theory:

- **Clipping** *z* to a box — destroys Gaussianity, and Theorem 2 says non-Gaussian latents break linear identifiability. This is the mechanism behind the paper's own poor trajectory-condition results.
- **Sampling uniformly in joint space** — same problem; α → ∞ in their gennorm sweep, far from the α = 2 optimum.
- **Wrapping angles** — makes *g* non-injective and introduces exactly the "joint-limit wrapping" artefact they name as a theory violation.

> ### Recommended approach: absorb the squash into g
>
> Keep *z ~ N(0, Iₙ)* genuinely unbounded. Define a fixed, deterministic, strictly monotonic map *φ : Rⁿ → PhysicalState*, e.g. a scaled `tanh` into each joint's safe range. Then the observation is
>
> ```
> x = render(φ(z)) ≡ g(z)
> ```
>
> Because the theory allows *g* to be *any* measurable nonlinear map (their Hermite argument needs only measurability, not diffeomorphism), the squash is legitimately part of the unknown mixing. The latents stay exactly Gaussian, every sampled state is physically valid by construction, and there is no wrapping. The encoder's job is simply harder — it must invert the squash too, which is a genuine test of the theory rather than a violation of it.

Use `tanh` rather than a hard clip: `tanh` is a bijection onto the open interval, so *φ* is injective and *g* retains injectivity. A clip would map an entire tail to a single point and destroy it.

> **The float32 caveat.** `tanh` is injective in exact arithmetic, but in float32 the physical step per unit of latent falls below any write tolerance long before overflow — past |z| ≈ 5 — and rounds to the bound exactly past |z| ≈ 9. Distinct latents then teleport the scene to the *same* state, and *g* is non-injective there for a purely numerical reason. `LatentSpec.saturated(phi, atol=…)` flags it, measured in **physical** units against the same tolerance as the read-back gate. Never in latent space: `atanh` turns a micron of solver tolerance into an unbounded latent error. Saturation is logged per sample like visibility and collision, so it can be conditioned on rather than guessed at.

### 5.2 Stage-1 latent specification (n = 7)

**Provisional** (see §3.3) — the joint selection and radii below are placeholders until the Franka asset loads and its limits are read back via `robot.find_joints(...)` and the articulation's joint-limit data. Radii are expressed as fractions of the *measured* half-range, never as absolute numbers baked into code.

| Index | Semantic factor | Physical handle | Squash φᵢ |
|---|---|---|---|
| 0–3 | Arm configuration (4 principal DoF) | Franka `panda_joint1,2,4,6` (by name) | cᵢ + rᵢ·tanh(zᵢ), rᵢ = a fraction of measured half-range |
| 4 | Gripper aperture | `panda_finger_joint1/2` | mapped into the measured aperture range |
| 5–6 | Cube position on table | cube root pose x, y | x₀ + r·tanh(z₅), y₀ + r·tanh(z₆), r set from the table extent |

Cube *z*-height is held fixed at the table surface plus half the cube edge — a deterministic function of the other coordinates, not a free latent. Remaining Franka joints are held at a fixed nominal pose in Stage 1 so that *n* stays small and the arm configuration is uniquely determined by the active joints.

> ### ⚠ Do not include cube yaw in Stage 1
>
> A cube has 90° rotational symmetry about its vertical axis. Yaw values of 0 and π/2 render to *identical pixels*, making *g* non-injective and that latent dimension formally unrecoverable. Either (a) omit yaw, (b) restrict it to a range narrower than the symmetry period via the squash, or (c) replace the cube with a visually asymmetric object (a textured block, a mug, a toy) before adding yaw. Option (c) is the right long-term answer and should be the Stage-2 change.

### 5.3 The occlusion problem

This is the most serious threat to a clean result and it is intrinsic to manipulation scenes, not to Isaac Sim.

When the arm passes in front of the cube from the camera's viewpoint, the cube's position is no longer observable. Two different latent vectors — same arm pose, different hidden cube position — produce the same image. *g* is then not injective, *h = f ∘ g* cannot be a bijection, and Theorem 1's conclusion is unreachable in principle for those regions of latent space. The encoder will look like it is failing when in fact the data-generating process is degenerate.

Mitigations, in order of preference:

1. **Multi-view observation.** Render 2–3 cameras at well-separated viewpoints and concatenate (or stack as channels). Occlusion from all views simultaneously is rare. This is the cleanest fix and costs proportionally more render time.
2. **Camera placement.** A high, oblique, near-top-down view makes arm-over-cube occlusion much less frequent than an eye-level view. Cheap and worth doing regardless.
3. **Spatial separation in Stage 1.** Restrict the cube's squashed region to a table area the arm rarely sweeps over. Scientifically a bit of a dodge, but useful as a control condition to isolate occlusion as the cause of any measured gap.
4. **Measure it.** Render a segmentation pass alongside RGB, compute the fraction of cube pixels visible, and log it per sample. Identifiability can then be reported conditioned on visibility — which turns a confound into a finding.

Default configuration: (1) + (2), with (4) always on, because per-sample visibility is cheap to record and enormously clarifying when a number comes out low.

### 5.4 Isotropy of the transition

Appendix F of the paper makes a point that is easy to miss and expensive to get wrong: **the simultaneous (non-sequential) optimisation used by LeJEPA requires isotropic transitions**. If different latent dimensions have different autocorrelations ρ_α, the eigenvalue ordering interleaves, and the encoder recovers the *second* Hermite component of a slow latent instead of the first component of a fast one. Their formal condition is max_α K_α < 2 min_β K_β.

Because we sample *z* directly rather than rolling out a policy, this is trivially satisfiable: **use one scalar ρ for all dimensions**. Resist any temptation to give the cube a different correlation from the arm. Conversely, an *anisotropic* ρ sweep is a cheap and valuable ablation — it should reproduce the eigenvalue-interleaving failure and would be a genuine contribution beyond the paper's own experiments.

### 5.5 Validity policy for teleported states

Teleporting can produce arm-cube interpenetration. Note carefully that this does *not* break the theory: *g* only needs to be deterministic and injective, and an interpenetrating configuration is still a well-defined deterministic render. It looks physically wrong but is mathematically admissible.

Three options, and the first is recommended:

- **Accept interpenetration, log it.** Maximally faithful to the theory, zero sampling bias. Record a collision flag per sample so it can be ablated later. Some reviewers will find the images odd; the flag lets that be answered with data.
- **Rejection sampling.** Discard colliding pairs. *This biases the latent distribution away from Gaussian* and therefore partially undermines Theorem 1's premise. Avoid unless the collision rate is tiny.
- **One settling step.** Write the state, step physics once, then render. This makes *g* depend on the physics solver and introduces a non-injective many-to-one map (different pre-settle states settle to the same post-settle state). Worst option for identifiability; useful only for physically plausible imagery in a figure.

> **Dependency on Spike 2.** This recommendation assumes rendering is possible with no physics step at all (§4.5). If the spike shows a step is unavoidable, this section must be revisited, because "accept interpenetration" and "a step happens anyway" are not compatible.

---

## 6. OU Sampling Implementation

### 6.1 The sampler

Direct transcription of Eq. (1) of the paper. Trivial code, but note the batching and the single shared ρ.

```python
# src/idtb/latents/ou.py  (shipped)
def sample_ou_pairs(n, batch, rho, *, device="cpu", dtype=torch.float32, generator=None):
    """z ~ N(0, I_n);  z' = rho*z + sqrt(1-rho^2)*eta,  eta ~ N(0, I_n).

    Single scalar rho across all dims -> isotropic transition, required
    for simultaneous (parallel) identifiability. See paper App. F.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must lie in [0, 1], got {rho!r}")
    kwargs = {"device": device, "dtype": dtype, "generator": generator}
    z = torch.randn(batch, n, **kwargs)
    eta = torch.randn(batch, n, **kwargs)
    return z, rho * z + (1.0 - rho**2) ** 0.5 * eta
```

Sanity assertions kept in the test suite: `Cov(z) ≈ I`, `Cov(z′) ≈ I`, `Cov(z, z′) ≈ ρI`, and marginal normality per dimension. If any fails, the entire downstream analysis is meaningless.

### 6.2 Latent specification and squash

```python
# src/idtb/latents/spec.py  (shipped)
@dataclass(frozen=True)
class Handle:
    role: str          # backend-independent: "arm.j0", "cube.x"
    center: float
    radius: float      # tanh amplitude
    # Handle.from_limits(role, lo, hi, fraction) builds these from *measured*
    # limits, so no radius is ever an absolute number baked into code.

@dataclass(frozen=True)
class LatentSpec:
    """Registry mapping latent dims -> physical handles.
    Adding an object == appending handles. n grows, nothing else changes."""
    handles: tuple[Handle, ...]

    def squash(self, z):                 # phi: R^n -> physical values
        center, radius = self._params(z)
        return center + radius * torch.tanh(z)

    def saturated(self, phi, *, atol):   # float32 injectivity diagnostic, §5.1
        center, radius = self._params(phi)
        return radius - (phi - center).abs() < atol
```

Handles carry **roles**, not joint names and never indices. The backend owns the role → target mapping — the Isaac backend resolves `arm.j0` to a joint index once at bind time via `robot.find_joints(...)`, the mock to a sprite parameter. A spec keyed on `panda_joint1` could not be shared with the mock, which has no Franka, and the tier-1 contract suite (§10.3) exists precisely to run one set of tests against both. Hardcoded indices are a silent-corruption hazard: they change with asset revisions.

### 6.3 State writer

Sketch, structured to match §4.5. Two points of care: root poses are in **world** frame so the environment origin must be added, and the root pose is a 7-vector whose orientation is a **normalised (w, x, y, z) quaternion** — not something to leave uninitialised while writing only x and y.

```python
# src/idtb/sim/writer.py  (sketch — imports Isaac lazily, inside the function)
def write_latent_state(scene, bound_spec, phi_vals):
    """Teleport the scene to the physical state encoded by phi_vals.
    phi_vals: [B, n] already squashed. No physics stepping."""
    robot = scene["robot"]
    cube  = scene["cube"]

    # --- arm joints (indices resolved at bind time from names) ---------
    joint_pos = robot.data.default_joint_pos.clone()
    joint_pos[:, bound_spec.joint_cols] = phi_vals[:, bound_spec.joint_dims]
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

    # --- cube root pose, world frame, normalised wxyz quaternion -------
    root = cube.data.default_root_state.clone()          # [B, 13]
    root[:, bound_spec.cube_cols] = phi_vals[:, bound_spec.cube_dims]
    root[:, 0:3] += scene.env_origins                    # local -> world
    cube.write_root_pose_to_sim(root[:, :7])             # quat already set
    cube.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))

    # --- flush; do NOT call scene.reset(), which restores defaults -----
    scene.write_data_to_sim()
```

> ### ⚠ Silent write failures are the real API risk
>
> `write_root_pose_to_sim` was reported leaving objects frozen at the environment origin on Isaac Sim 5.0 despite working on 4.5 ([IsaacSim #251](https://github.com/isaac-sim/IsaacSim/issues/251)). That issue is closed, but **not because it was fixed**: NVIDIA judged the behaviour to originate in the Isaac Lab asset layer rather than Isaac Sim core and sent it there. It was reassigned, not resolved — and we are moving to a *different* Isaac Lab major version than it was filed against. A failure of this kind is silent: the dataset generates normally and the cube latents are simply noise. **The read-back assertion in §4.5 is therefore not optional and not a debug aid — it is a correctness gate that runs on every sample during development and on a sampled basis in production runs.**
>
> Assets configured with `fix_root_link` or `kinematic_enabled` can also silently ignore root-pose writes. Same gate catches it.
>
> For the record, `write_joint_state_to_sim` is **not** deprecated as of `main`; it takes `joint_ids`/`env_ids` directly. Deprecations elsewhere in the asset API (`set_external_force_and_torque`, `write_joint_friction_to_sim`) do not affect this project.

### 6.4 Generation loop

```python
# src/idtb/gen/generate.py  (sketch — backend is a SceneBackend, real or mock)
for shard in range(n_shards):
    z, z_next = sample_ou_pairs(spec.n, batch, rho, generator=g)

    backend.write_state(spec.squash(z))
    x  = backend.render(sample_idx=idx)
    d1 = backend.diagnostics()

    backend.write_state(spec.squash(z_next))
    x2 = backend.render(sample_idx=idx + 1)
    d2 = backend.diagnostics()

    store(shard, x=x, x_next=x2, z=z, z_next=z_next,
          visibility=(d1["visibility"], d2["visibility"]),
          collision=(d1["collision"], d2["collision"]),
          rho=rho, seed=seed)
```

Because the driver speaks only to `SceneBackend`, this entire loop is exercised locally against the mock before it ever runs on a GPU.

---

## 7. Rendering: Photorealism vs. Determinism

### 7.1 The determinism requirement

> ### ⚠⚠ This is the default behaviour, and it will silently corrupt results
>
> Theorem 1 assumes *x = g(z)* is a deterministic function of *z*. Two failure modes break this:
>
> **(a) Sampling noise.** Monte Carlo path tracing with a free-running seed gives a different image for the same state on every call. The encoder then sees *x = g(z) + noise*, which shifts the setup into a different (noisy-observation) regime not covered by the theorem.
>
> **(b) Temporal accumulation.** Neural denoisers and DLSS ray reconstruction accumulate across frames. This makes the render of *x′* depend on the previously rendered *x* — a direct, artificial correlation between the two members of every positive pair. An encoder can exploit this leakage to score well on alignment without learning anything about the world. This is the single most dangerous artefact in the whole pipeline.
>
> **The trap is that this is the out-of-the-box configuration.** Isaac Sim's default "RTX Real-Time 2.0" mode is itself path tracing with DLSS neural rendering (`RealTimePathTracing`) — it is temporally accumulating by default. There is no "just use the fast rasteriser" escape hatch; naming a realtime mode does not buy determinism. Determinism has to be constructed and then measured.

Required mitigations:

- Fix the render seed per sample (a deterministic function of the sample index), or use enough samples per pixel that residual MC noise is below quantisation.
- Disable temporal denoising and DLSS ray reconstruction, or force an accumulation reset between every capture. Candidate controls: `antialiasing_mode="Off"`/`"DLAA"`, `enable_dl_denoiser=False`, `/rtx/pathtracing/spp`, `/rtx/pathtracing/totalSpp`. Which combination actually achieves determinism is Spike 1.
- Render *x* and *x′* with independent accumulation buffers, never back-to-back within one accumulating sequence.

**Determinism acceptance test (a hard gate; run before generating any dataset):**

```
render(z_a) -> A1 ;  render(z_b) -> B1
render(z_b) -> B2 ;  render(z_a) -> A2
assert mean_abs_diff(A1, A2) < tol   # order-independence
assert mean_abs_diff(B1, B2) < tol   # no temporal leakage
```

If this fails, nothing else in the project is worth running. The same test runs against `MockSceneBackend`, where it must pass trivially — that is what makes a pass on the Isaac backend meaningful rather than merely reassuring.

### 7.2 Phase-1 spikes

These four questions gate the render design. None of them can be answered from documentation; all require a GPU pod.

| # | Question | Why it matters |
|---|---|---|
| **1** | **Determinism.** Same state, two renders → equal? Then A/B/B/A order-independence, per candidate mode. | Decides whether any realtime mode is usable, or whether full `PathTracing` with a fixed SPP cap and the denoiser off is mandatory. |
| **2** | **Minimal capture sequence.** Does `write → write_data_to_sim → forward → render` yield a correct image with *zero* `sim.step()` calls? | If a physics step is required, §5.5's validity policy must be rethought, because a step means settling. |
| **3** | **Convergence depth N.** How many `sim.render()` calls (or what `rt_subframes` / `totalSpp`) until the image stops changing. | This is the per-sample cost multiplier and therefore the entire GPU-hour budget. |
| **4** | **`TiledCamera` × the mode chosen by Spike 1.** Does tiled rendering work there? | Worth an order of magnitude in throughput. The known failure is the *opposite* of the intuitive guess: under **ray tracing** the tiles come back wrong — correct total resolution, per-camera tiles gone — while path tracing is fine ([IsaacSim #367](https://github.com/isaac-sim/IsaacSim/issues/367)). NVIDIA reproduced it, confirmed it in 5.1 and **fixed it in 6.0**, which is one of the reasons we are on 6.0 (§3.4). Verify the fix rather than trusting it — this spike is now a regression check with a known-good expectation, which makes a FAIL *more* informative, not less. |

Spikes 1 and 2 are the ones that can retroactively invalidate weeks of work. They run first.

### 7.3 Render presets — candidates, not a configuration

Preset *definitions* are deliberately left open until Spike 1 and Spike 3 return. What is fixed is the *shape*: three presets, spanning a realism/cost axis, all of which must pass the §7.1 gate before use.

| Preset | Intent | Defined by |
|---|---|---|
| `debug` | Fastest thing that renders a recognisable image. Scene authoring, camera placement, smoke tests. Determinism not required. | Whatever is cheapest that works |
| `standard` | Main dataset generation: good realism at tractable cost, provably deterministic. | Spikes 1 + 3 |
| `photoreal` | Headline figures and a smaller high-fidelity dataset for the realism ablation. | Spikes 1 + 3, with SPP set for convergence |

A genuinely interesting experiment falls out of having more than one: **does linear identifiability degrade as rendering realism increases?** Same latents, same ρ, same encoder, three presets. If *R²* drops with realism, that is a real and publishable finding about the gap between the theory's idealisation and realistic observation. Cost is one extra dataset generation run.

### 7.4 Scene and lighting setup

- **Lighting:** HDRI dome plus one or two area lights. Fixed across the dataset — lighting variation is *not* a latent unless deliberately made one.
- **Materials:** proper PBR (roughness, metallic, normal maps) on table, arm and cube. Flat diffuse materials would make the "photorealistic" claim indefensible.
- **Camera:** fixed extrinsics and intrinsics, high oblique angle, 2–3 views per §5.3. Intrinsics logged to dataset metadata.
- **Deliberate non-latents:** anything not in the latent vector must be held exactly constant, or it becomes uncontrolled nuisance variation the encoder must marginalise. Fix random seeds for any procedural material or placement.

---

## 8. Infrastructure

### 8.1 GPU selection criteria

The GPU model is now decided (§3.1); the criteria below are what decided it and what any future re-provisioning must satisfy again:

| Criterion | Rule |
|---|---|
| RT cores | **Mandatory.** A100 and H100 are explicitly unsupported. Verify the allocated GPU is the advertised part, not a substituted datacenter card. |
| VRAM | Enough headroom for the target scene stage. Room-scale (Stage 2/3) wants more than the tabletop scene. |
| Driver | Must satisfy the requirements page *of the Isaac Sim release being pinned*. This is a matching problem between the release and the provider's pool, not a fixed number — check both at provisioning time. |
| Availability / price | Real constraints; an ideal part that is never available is not a choice. |

Practically, this narrows to RTX-class parts (GeForce RTX 40/50 series, L40S, RTX A6000, RTX 6000 Ada, RTX PRO 6000 Blackwell).

**Provisioned and surveyed** (`infra/preflight.sh`). Two surveys, because the pod
was recreated onto the vendor image between them — and **the host underneath is
not stable across a recreate**, which is the single most important thing this
section now records:

| | Survey 1 — bare pod, 2026-09-13 | Survey 2 — vendor image, 2026-09-16 | Wanted *(at the time)* |
|---|---|---|---|
| GPU | RTX 4090, 24 GB, CC 8.9 | RTX 4090, 24 GB, CC 8.9 | RT cores; ≥ RTX 4080 / 16 GB |
| **Driver** | **580.178.04** | **570.195.03** ❌ | ≥ 580.95.05, for 6.0.0 — **superseded, §3.4.1**: the target moved to Isaac Sim 6.0.1 (≥ 595.58.03), and 570.195.03 is now the accepted, known-below-tested driver this project actually runs on |
| OS | Ubuntu 22.04.5 | Ubuntu 24.04.3 | either is supported |
| glibc | 2.35 — *exactly at the floor* | 2.39 | ≥ 2.35 |
| CPU / RAM | 48 vCPU, 251 GiB | 32 vCPU, 124 GiB | not a constraint |
| Account | root (uid 0) | `ubuntu` (uid 1000) | uid 1000 in the Isaac container |
| System `python3` | 3.11.10 | **absent** | irrelevant — Isaac ships its own |
| Volume | 409 TiB free, root-owned | 643 TiB free, `1000:1000`, writable | writable by uid 1000 |
| Isaac Lab | — | `/workspace/isaaclab/isaaclab.sh` present, `ACCEPT_EULA=Y` | the mount did not shadow it |
| Egress | CDN / PyPI / GitHub 200, `nvcr.io` 401 | **all four unreachable** ❌ | 401 from `nvcr.io` is reachable-and-unauthenticated |

Survey 2 is two blockers and one lesson, and it reported **"all checks passed"**
for all three:

- **The driver went backwards, from 580.178.04 to 570.195.03.** Nothing about the
  image or the volume caused this — `nvidia-smi` inside a container reports the
  *host's* driver, and a recreate is a fresh allocation. §3.1's release was
  resolved against a machine we no longer had. The fix is to select for it (CUDA
  Version = 13.0), not to change the tag. **A driver below the tested version
  does not announce itself**: Kit only refuses below 535.129, so 6.0.0 would very
  likely have booted and rendered, and the difference would have surfaced as
  numbers in §7.1 that nobody could explain.
- **Egress was gone** — all four endpoints, where survey 1 had reached all four.
  The re-run reports `curl`'s exit code, which separates the causes that matter
  (6 = DNS, 7 = refused, 28 = timeout, 35/60 = TLS). Without egress, `bootstrap.sh`
  cannot check out the repo and Isaac cannot stream Omniverse assets on first run.
- **The survey said all checks passed anyway**, because `curl` writes `000` for
  `%{http_code}` on a failed transfer *and* exits non-zero, so the script's
  `|| echo "000"` appended a second one; `000000` then compared unequal to `000`
  and a total blackout scored green. Exit status decides it now, and
  `tests/test_infra_scripts.py` pins the regression. The lesson generalises: a
  check whose failure path has never been exercised is not a check, which is the
  same argument §10.1's negative controls make about the determinism gates.

Survey 1's numbers are kept above because they are what §3.1 was resolved
against, and a decision register that quietly loses its own evidence is worse
than one that shows the evidence moving.

### 8.2 Pre-flight checks (before writing any Isaac-facing code)

1. **Driver version.** Spin up a candidate pod and run `nvidia-smi`. **The reading constrains the Isaac Sim release** (§3.1), not the other way round; getting this backwards is the most common cause of a failed first day. Where the driver clears more than one release — as it did here — say so and record the tiebreak explicitly, rather than letting a preference pass for a measurement.
   **This project has since stopped trying to reverse that arrow (§3.4.1).** Two attempts to *select hosts by* driver both failed to land the intended branch — the provider's allocation decides it, not a filter — so the driver reading is recorded and checked (`preflight.sh` still fails below the current release's tested version, by default) but is no longer something a pod recreate is aimed at fixing. It is re-read on every recreate anyway, because the host changes underneath regardless of intent, and a driver below the tested version still boots without saying so.
2. **RT core presence.** Confirm the allocated GPU is what was advertised.
3. **NGC account and API key.** Free; required for `docker login nvcr.io`. Set up before it is needed.
4. **Outbound network.** Isaac Sim streams assets from an Omniverse CDN endpoint on first run, and `bootstrap.sh` checks the repo out over HTTPS. Confirm unrestricted egress — and read the *exit code*, not the HTTP code, since a failed transfer still prints one (§8.1).
5. **Headless render smoke test.** Render one frame to disk and inspect it. Do not proceed until an image file exists.

### 8.3 Container and volumes

**No custom image** — with one mechanical exception, below. NVIDIA ships prebuilt headless Isaac Lab images on NGC with Isaac Sim and Isaac Lab already installed *and already matched to each other*; building our own would only re-do that work and add a second thing to keep matched to the driver.

> ### The one thing a rebuild may still be needed for: `ENTRYPOINT`
>
> The image's `ENTRYPOINT` is `runheadless.sh`, and RunPod's "Container Start Command" template field does **not** override an `ENTRYPOINT` — so the pod starts Isaac's streaming app instead of dropping you at a shell. Found on the pod, not in any documentation.
>
> The remedy is a two-line image (`FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1` + `ENTRYPOINT []`) or a provider template that clears it. That is not the "custom image" this section rejects: it rebuilds nothing, matches nothing to the driver, and adds no version to keep in sync — it only removes a default command. Keep it that way; the moment a `pip install` appears in it, §8.3's argument is lost.

The tag is `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1` (§3.1) and appears nowhere else in this repo. Our code is not baked into an image at all: it lives on the network volume, checked out once by hand, so a code change is a `git pull`, not a rebuild.

> ### Two tags, four characters apart, different Isaac Sim — and which one we actually run
>
> `3.0.0-beta2` is Isaac Sim **6.0.0**. `3.0.0-beta2-post1` is Isaac Lab's `v3.0.0-beta2.patch1`, which moved to Isaac Sim **6.0.1**, tested at driver 595.58.03. Both matter because they are four characters apart, pull equally cleanly, and a version mismatch between them fails at a layer that never mentions drivers.
>
> **We are on `-post1` (6.0.1), and that is now the decision, not an accident.** The pod that came up on 2026-09-16 was already running it — pulled before anyone was watching for the tag — on a 570.195.03 host, two driver branches below what 6.0.1 tests at. §3.4.1 records why that pod is being kept rather than replaced: two recreates aimed at a 580-branch host for plain `beta2` both landed on 570-branch hosts instead, so the driver was never actually controllable from this end, and re-pulling `beta2` now would trade a working image for an unverified one over a one-patch-release difference. `preflight.sh` reads `/isaac-sim/VERSION` and checks it against `-post1`/6.0.1 (`IDTB_ISAACSIM_VERSION`), not `beta2`/6.0.0, because the tag itself is invisible from inside a running container and memory is not evidence.

Everything below was read out of Isaac Lab `v3.0.0-beta2`'s own `docker/.env.base`, `docker/Dockerfile.base` and `docker/docker-compose.yaml` — the same release line as `-post1`, which changes the pinned Isaac Sim version but not the docker layout — not inferred. `tests/test_infra_scripts.py` pins the same facts so our scripts and the vendor layout cannot drift apart unnoticed.

> ### The volume must not mount at `/workspace`
>
> The image unpacks Isaac Lab into `/workspace/isaaclab` (`DOCKER_ISAACLAB_PATH`). A network volume mounted at `/workspace` — the default on most providers — shadows it, and what you see is a missing `isaaclab.sh`. That reads as a broken image, and the mount is the last thing anyone suspects.
>
> Mount the volume at **`/idtb`** instead. The mount path is a pod setting, not a volume setting, so fixing it costs a pod recreate and leaves the volume's contents alone. Both scripts refuse `/workspace` rather than letting it proceed.

> ### The container is *not* root — and this is where a first session gets lost
>
> `Dockerfile.base` does its setup as root and then ends on `USER isaaclab`, uid/gid **1000**, with `$HOME=/root` chowned to that user. A provider-supplied network volume arrives owned by `root:root` (the survey in §8.1 confirms ours does), so uid 1000 cannot write to it — and the image ships **no `sudo`**, so the container cannot fix this itself.
>
> Take ownership from a root shell *before* the Isaac container needs the volume:
>
> ```
> chown -R 1000:1000 /idtb      # from a root shell on the pod, once per volume
> ```
>
> `infra/bootstrap.sh` attempts the chown and, when it cannot, stops with that instruction rather than continuing and failing later inside Kit. The failure it prevents is indirect — `PermissionError` creating `logs/`, or `omni.datastore` lock errors under `kit/cache` — and reads as an Isaac bug.
>
> Note this reverses between image generations: Isaac Lab 2.3.2 runs as root and needs `OMNI_KIT_ALLOW_ROOT=1`, which 3.0's uid-1000 user does not. `bootstrap.sh` sets it only when it is actually running as root, so it stays correct either way.

**Pod environment.**

| Variable | Why | Set where |
|---|---|---|
| `ACCEPT_EULA=Y` | The image requires it; without it the container does not come up. | Pod env var, at creation |
| `OMNI_KIT_ALLOW_ROOT=1` | Only when the uid *is* root — Kit refuses to start as root without it. Not needed on the 3.0 image. | `$IDTB_VOL/env.sh`, written conditionally by `bootstrap.sh` |

Two scripts, both idempotent and both written to run on a pod nobody has logged into yet:

- **`infra/preflight.sh`** — read-only survey: driver version, GPU part, VRAM, OS, egress to NGC / the Omniverse CDN / PyPI, volume ownership and free space, and (when run inside the image) whether `isaaclab.sh` survived the mount, whether the current uid can actually write the volume, and whether the env vars are set. Every check runs even after one fails, and the verdict block is what gets pasted back. It hard-fails an RT-core-less GPU (A100/H100 and friends) rather than letting a cheap allocation look fine, and — since §3.1 is now resolved — also hard-fails a driver below 580.95.05, warns above the 595 boundary, reads the **image's own Isaac Sim version** so the `beta2` / `beta2-post1` tag is confirmed from inside rather than remembered, and decides egress on `curl`'s exit status rather than on the code it prints (§8.1 explains why that distinction cost a survey).
- **`infra/bootstrap.sh`** — takes ownership of the volume, relocates every cache onto it by symlink, writes `env.sh`, and creates the dataset and log directories. **It touches the network nowhere**, which matters twice over: a pod's DNS is briefly dead right after boot (below), and every cache has already been moved by the time the last step runs, so a network failure there would discard the whole run.

> ### It does not clone the repo, and `$HOME` lies
>
> **An earlier version cloned the repo as its last step, which was circular** — the script lives *in* the repo, so anything able to run it already has one. Worse, it cloned to a second path, so the checkout being edited and the checkout Isaac ran were different directories. It now reports the checkout it is running from, and whether that checkout is on the volume: one in the container's own filesystem is gone at the next pod start, taking anything edited on the pod with it. Updating the code is a plain `git pull`, by hand, in the checkout you chose.
>
> **The image exports `HOME=/root` while the container runs as uid 1000**, so every `$HOME/...` cache path resolves somewhere this user cannot write. The relocation then fails *partway* — some caches moved, some not — which presents as a half-finished run rather than an error. `bootstrap.sh` checks whether `HOME` is actually writable and, when it is not, takes the home directory from the password database instead, saying so. It also exports the corrected `HOME` into `env.sh`: if Isaac disagrees with us about where home is, it reads caches at a path nothing was relocated to and re-downloads everything while the symlinks sit unused — the failure mode the whole volume layout exists to prevent.
>
> **Docker's embedded DNS (`127.0.0.11`) is sometimes not forwarding yet just after the container boots.** Hostname lookups fail while raw IP connectivity is fine, and it clears itself within seconds — a provider quirk, not an image problem. It is the same condition `preflight.sh` reports as `curl` exit 6. Nothing in `bootstrap.sh` depends on it any more; a `git pull` that fails this way is worth simply retrying, and `getent hosts github.com` tells you whether it is still DNS.
>
> **`/isaac-sim/kit` itself can be root-owned, independently of the volume — and unlike the volume, there is no root shell to fix it with.** §8.3's `chown` instruction covers the *volume*; it says nothing about paths inside the image, and on the 2026-09-16 pod `/isaac-sim/kit` was one of them: `rm` failed with `Permission denied` after the relocation had already copied the cache, and the run died there under `set -e`, with `kit-data` and everything after it un-relocated. Replacing a path with a symlink is a write to its *parent directory*, not to the cache files themselves, so this is a distinct failure from anything file permissions inside `kit/cache` would predict — confirmed on the pod: `/isaac-sim/kit/cache` itself is `ubuntu:ubuntu` and writable, only its parent `/isaac-sim/kit` is not. `bootstrap.sh` probes the nearest existing ancestor of each cache path before touching it and collects every one it cannot write instead of stopping at the first, so everything relocatable still gets relocated.
>
> The obvious next step — `chown` it from a root shell — **does not exist on this image**: both `su` and `sudo` fail from inside the running container (no `sudo` binary at all), and a custom rebuild just to run one `chown` and an `ENTRYPOINT []` was considered and deliberately declined, for the same reason §3.4.1 declined re-provisioning — it trades a known-working setup for new surface (a registry to push to and maintain) to fix something that turns out to be minor. Because it *is* minor: since the cache directory itself is already correctly owned, Isaac's own runtime writes into it work exactly as the vendor intended — only **relocating** it is blocked, so what's actually lost is cross-session persistence of that one cache (`kit/cache`, the largest one persisted; `kit/data` is smaller and newer in Isaac Lab 3.0). It re-populates after every pod restart or recreate instead of surviving one, which costs a slower cold start each time that happens — not on any command within a live session. `bootstrap.sh` reports this as a `NOTE` and still exits `0`: there is nothing to fix and re-run for, so treating it as a failure would just be a check nobody can act on, crying wolf on every single invocation.

> ### Persistent volume layout — do not skip this
>
> Rented pods are ephemeral. Without a network volume, every pod start re-downloads gigabytes of Omniverse assets and recompiles shader caches, costing paid GPU time on every session. The set of paths worth persisting is not a guess — it is the list the vendor's own `docker-compose.yaml` keeps in named volumes:
>
> | Path | What it holds |
> |---|---|
> | `/isaac-sim/kit/cache` | Kit extension cache — the largest, and the one an earlier draft of `bootstrap.sh` missed entirely because it only considered `$HOME` |
> | `/isaac-sim/kit/data` | Kit runtime data — **new in Isaac Lab 3.0**, absent from the 2.3.2 list |
> | `~/.cache/ov` | Omniverse asset cache (the CDN downloads) |
> | `~/.cache/nvidia/GLCache` | compiled GL shaders |
> | `~/.nv/ComputeCache` | compiled CUDA kernels |
> | `~/.cache/pip` | Python packages |
> | `~/.local/share/ov/data` | Omniverse app data |
> | `~/.nvidia-omniverse/logs` | Kit logs |
>
> `bootstrap.sh` relocates each by symlinking its real path at the volume, which works on providers that give you one mount and no control over individual bind mounts. Parent directories are linked where that is a superset. The vendor also persists `/isaac-sim/kit/logs/...` and `~/Documents`; those are outputs rather than caches, so losing them recomputes nothing and they are deliberately left alone.
>
> Isaac Lab 3.0 ships `docker/utils/volume_mounts.py`, which parses that compose file for exactly this list. Once the pod is up, prefer diffing our list against its output over trusting either.
>
> Verify the relocation actually took by restarting the pod once and confirming Isaac does not re-download assets. A cache that silently is not persisting looks exactly like a slow first run.

### 8.4 Operating pattern and cost

Treat the GPU as a batch renderer, not a development environment. Per §1 this is not a preference — there is no alternative on macOS.

- Develop the sampler, spec, driver, metrics and training code **locally against `MockSceneBackend`**. Only rendering needs the GPU.
- Use a small persistent pod (or CPU pod) for code sync, and a large GPU pod launched only for spikes and generation runs.
- Generate datasets to the network volume, then pull them down or train in place — a generated dataset is reused across many encoder training runs, so generation is a one-off cost per configuration.
- **Avoid spot/interruptible instances for long generation runs** unless the generator checkpoints per shard. With per-shard checkpointing, spot becomes attractive and roughly halves cost.

*Cost figures are deliberately omitted from this plan: rates move, and a stale number in a plan of record is worse than no number. Record actual observed rates here at provisioning time. The dominant cost driver is expected to be the `photoreal` preset, scaled by the Spike-3 convergence depth.*

---

## 9. Phased Implementation

| Phase | Deliverable | Contents | Effort |
|---|---|---|---|
| **0** | Pre-flight | NGC account; candidate pod; `nvidia-smi` driver reading; **resolve the Isaac Sim / Isaac Lab / Python versions from that reading and record them in §3** *(done — §3.1, §8.1)*; container pulls and launches headless; one frame rendered to disk *(outstanding)*. | 0.5–1 day |
| **0b** | Pure layers *(parallel, local, no GPU)* | OU sampler, `LatentSpec` + squash, package scaffolding, the §4.2 import guard, pod scripts, tier-0 tests green. **Blocked on nothing — done.** | 2 days |
| **1** | Infrastructure + spikes | Vendor `isaac-lab` image pulled at the resolved tag; network volume mounted at `/idtb` with caches relocated onto it; `infra/bootstrap.sh` handling the image's uid-1000 user against a root-owned volume; repeatable pod launch. **Spikes 1–4 (§7.2) answered and recorded in §3.** | 2–3 days |
| **2** | Scene v1 | `stage_v1_tabletop.usd`: table, Franka, cube, PBR materials, HDRI + area lights, camera rig (2–3 views). Debug-preset renders look right. | 1–2 days |
| **3** | Backend seam + real backend | `SceneBackend` protocol and `MockSceneBackend`, designed against the spike's measurements rather than against documentation; then `IsaacSceneBackend`: handle resolution by name, state writer, read-back assertions confirming every write landed. Collision and visibility diagnostics. Tier-1 contract suite green against Isaac. | 2 days |
| **4** | Determinism gate | Deterministic capture path; the §7.1 acceptance test passing for every preset intended for dataset use, called by `generate.py` itself and not only by the test suite. | 1–2 days |
| **5** | OU generator | Sharded writer storing (x, x′, z, z′, visibility, collision, ρ, seed, intrinsics); per-shard checkpointing. | 2 days |
| **6** | First dataset | ~100k pairs at `standard`, ρ = 0.95 (a starting point, not a finding). Visual audit of a random sample grid. | 0.5–1 day compute |
| **7** | Analysis | LeJEPA/SIGReg training; metrics: R²(h→z), R²(z→h), ‖Q̂ᵀQ̂−I‖_F/√n, ε, δ, bound D + (ε+D)². **First real number.** | 2–3 days |
| **8** | Sweeps | ρ ∈ {0.3 … 0.99}; λ grid; render-realism ablation; gennorm α latent-distribution sweep (converse test); anisotropic-ρ ablation; resolution ablation. | 3–5 days + compute |
| **9** | Scene v2 (room) | USD reference of a room shell around stage v1; relight; re-validate determinism and occlusion statistics; regenerate and re-measure. | 4–6 days |
| **10** | Scene v3 (objects) | Add manipulands one at a time; each adds 2–3 latent dims. Study identifiability vs. *n*, and the *m ≠ n* regime the paper leaves open. | ongoing |

**Critical path to a first defensible result: Phases 0–7, approximately two weeks.** Phases 8–10 are where the scientific contribution lives.

Note that Phase 0b has no dependency on Phase 0 — the pure layers can be built while the GPU question is still open. This is the practical payoff of the seam: the version decision blocks almost nothing.

One deliberate re-ordering against an earlier draft: **the `SceneBackend` protocol and the mock are built in Phase 3, after the spike, not in Phase 0b.** Every Isaac signature in §4.4 came from reading documentation, never from running anything. The risk was never the method *names* — it is granularity and semantics, which only the spike settles. Designing the central abstraction against guesses and then discovering the guesses were wrong is the expensive order.

`docs/PLAN.md` carries the ordered task list for the current milestone, including who runs what; this table stays the plan of record.

---

## 10. Validation and Test Strategy

### 10.1 Pipeline-correctness gates

Each must pass before the next phase is trusted:

- **Sampler:** Cov(z) ≈ I, Cov(z, z′) ≈ ρI, per-dim normality tests pass.
- **Writer:** state read back from the sim matches what was written, to solver tolerance, for every handle. Non-negotiable — see the §6.3 warning.
- **Renderer:** order-independence test of §7.1 passes.
- **Injectivity proxy:** nearest-neighbour check — the fraction of image pairs whose pixel distance is near zero while their latent distance is large should be negligible. A non-trivial fraction means *g* is not injective and there is an occlusion or symmetry problem to fix *before* blaming the encoder.
- **Trivial-baseline check:** a linear probe from raw pixels to *z* should score poorly (confirming the mixing is genuinely nonlinear, analogous to the paper's *R²*(x→z) ≈ 0.73–0.78 column). If raw pixels already predict *z* linearly, the task is too easy to be informative.

### 10.2 Scientific measurements

Mirror the paper's metric set so results are directly comparable to their Tables 1 and 2:

- Bidirectional linear *R²* between *h(z)* and *z*, fitted on a train split and scored on held-out evaluation samples.
- Per-dimension *R²*, to expose anisotropy (their Reacher shoulder-vs-wrist asymmetry).
- Orthogonality error ‖Q̂ᵀQ̂ − I‖_F/√n.
- Approximate-bound verification: compute ε, δ, D = δ/(2ρ(1−ρ)), and check the measured recovery error falls below D + (ε+D)².
- Identifiability conditioned on cube visibility — unique to this setup and probably the most interesting number the project will produce.

### 10.3 Test tiers

Tests accompany every module. The tiering exists because of the platform constraint in §1, not for tidiness.

| Tier | Runs on | Contents |
|---|---|---|
| **0 — pure** | laptop, plain `pytest` | OU sampler statistics, squash monotonicity and injectivity, `LatentSpec` handle bookkeeping, shard writer round-trip, metrics against synthetic ground truth |
| **1 — contract** | laptop (mock) **and** pod (Isaac), one suite parametrized over backend | write → read-back fidelity; render determinism and order-independence (§7.1); output shapes and dtypes; injectivity proxy; diagnostics present and in range |
| **2 — Isaac only** | pod, under Isaac's interpreter | asset loading, joint-name resolution against the real Franka, annotator availability, convergence-to-fixed-point of the chosen preset |

Mechanics:

- Tier 2 and the Isaac half of tier 1 are marked `@pytest.mark.isaac` and deselected by default.
- A **session-scoped fixture** owns the single `SimulationApp` (§4.2); tests must not attempt to create a second.
- **No hosted CI.** The suite is run by hand — locally before a commit, and on the pod for the Isaac tiers. The §10.1 gates therefore live in `src/` and are called by `generate.py`, so a gate cannot be skipped just because nobody ran `pytest`; the tests and the generator call the same function.
- Tier 1 is the payoff of the seam: the §10.1 gates become executable contracts that the mock **must pass** and the Isaac backend **must also pass**. A gate only observed against one implementation is weak evidence.

---

## 11. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Temporal denoiser leaks information between x and x′ | Critical | Determinism acceptance test, run by `generate.py` before every dataset run rather than only in the test suite (§7.1). Aggravated by being the *default* renderer behaviour — must be actively disabled, then measured. |
| Silent state-write failure (object frozen at env origin, `fix_root_link`, kinematic flags) | Critical | Read-back assertion on every sample in development (§6.3). [IsaacSim #251](https://github.com/isaac-sim/IsaacSim/issues/251) was closed by reassignment to the Isaac Lab layer, not by a fix — and we are on a different Isaac Lab major than it was filed against. |
| Provisioned a GPU without RT cores | Critical | Hard rule: RT-core GPUs only; `infra/preflight.sh` hard-fails one. **Closed for the current pod** — RTX 4090, CC 8.9 (§8.1). Re-opens on any re-provisioning. |
| No local runtime → slow, blind iteration on Isaac code | High | `MockSceneBackend` and the tier-0/tier-1 split (§4.3, §10.3); Phase 0b runs in parallel with Phase 0. |
| Occlusion makes g non-injective; results look like encoder failure | High | Multi-view cameras, high oblique placement, per-sample visibility logging, injectivity proxy check. |
| Cube rotational symmetry hides a latent dimension | High | Omit yaw in Stage 1; switch to a visually asymmetric object before introducing orientation latents. |
| Bounded joints break Gaussianity of z | High | Absorbed tanh squash (§5.1). Never clip, never wrap. |
| Driver / Isaac Sim release mismatch | Medium | Resolved to Isaac Sim 6.0.1, on a host below its tested driver, accepted knowingly (§3.4.1) rather than re-provisioned for — two prior attempts to select hosts by driver both failed to land the target branch. Residual: the gap is real (570.195.03 vs. tested 595.58.03) and `preflight.sh` fails on it by default rather than hiding it; the actual check is the §7.2 spike measuring the renderer directly, not the driver number as a proxy for it. |
| Isaac Lab 3.0 beta introduces breaking changes before 3.0 stable | Medium | Accepted knowingly (§3.4). Exposure is bounded by a deliberately small Isaac Lab surface behind the §4.3 seam, with the tier-1 contract suite defining what a migration has to keep working. Pin the tag; do not track `develop`. |
| Rendering throughput makes large datasets infeasible | Medium | Spikes 1, 3, 4 before committing to a preset; `standard` for the main dataset, `photoreal` only for a smaller ablation set. |
| Physics step turns out to be required before rendering | Medium | Spike 2. If confirmed, revisit §5.5 — it changes the validity policy, not just the code. |
| Ephemeral pods re-download assets every session | Medium | Persistent network volume; `infra/bootstrap.sh` relocates the vendor's own cache list onto it by symlink (§8.3). Only a pod restart proves it took — a cache that is not persisting is indistinguishable from a slow first run. |
| Network volume mounted at `/workspace` shadows the image's Isaac Lab install | Medium | Mount at `/idtb`; both pod scripts refuse `/workspace` and `tests/test_infra_scripts.py` pins the refusal. The symptom is a missing `isaaclab.sh`, which reads as a broken image rather than a mount problem. |
| Spot instance preempted mid-generation | Low | Per-shard checkpointing; resume from last completed shard. |
| Isaac Lab API churn breaks the writer | Low | Pin the resolved version; read-back assertions catch silent write failures immediately. |

---

## 12. Immediate Next Actions

Two tracks, and they are independent.

**Track A — resolve the unknowns (needs a GPU):**

1. ~~Launch a candidate RT-core pod, run `nvidia-smi`, and resolve the Isaac Sim / Isaac Lab / Python versions from the driver reading.~~ **Done, then revised** — originally Isaac Sim 6.0.0 / Isaac Lab 3.0.0-beta2 (§3.1); reversed 2026-09-16 to **Isaac Sim 6.0.1 / Isaac Lab 3.0.0-beta2.patch1**, the release the pod was already running, after two recreates failed to land a host on the driver branch 6.0.0 needed. §3.4.1 records why.
2. ~~Create a free NVIDIA NGC account and generate an API key, then `docker login nvcr.io`.~~ **Done** — the pod is on the vendor image; §8.1's second survey finds `/isaac-sim` and `/workspace/isaaclab/isaaclab.sh` in place, with the volume at `/idtb`, owned by 1000 and writable.
3. ~~Recreate the pod, filtering for a specific driver branch.~~ **Dropped, deliberately (§3.4.1).** This pod stays: image **`nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1`**, driver 570.195.03 accepted below the 595.58.03 this release tests at, volume already at `/idtb`. Only recreate for an unrelated reason (e.g. if egress turns out to be unfixable in place, next item) — not to chase the driver again.
4. **Confirm egress, don't just route around it.** The 2026-09-16 survey found zero egress on all four endpoints; a `curl --resolve codeload.github.com:443:<ip>` download worked, meaning it was likely DNS resolution failing rather than the network itself, matching the Docker-embedded-resolver quirk `infra/bootstrap.sh` now documents (§8.3). Re-run `infra/preflight.sh` plainly (no `--resolve`) and see whether it now passes on its own — if it still needs a workaround, `docker login nvcr.io` and Isaac's own first-run asset fetch will hit the same wall, and a per-command `--resolve` will not scale to those.
5. ~~Get the fixed `infra/bootstrap.sh` onto the pod and run it.~~ **Done** — it now finishes clean: every `$HOME` cache relocated, and `/isaac-sim/kit` reported as a `NOTE` (root-owned, unreachable from inside this container, accepted — §8.3) rather than a failure. No custom image; a rebuild for one `chown` + `ENTRYPOINT []` was considered and declined for the same reason §3.4.1 declined re-provisioning.
6. Restart the pod once and confirm Isaac does not re-download assets. This is the only proof the cache relocation actually took.
7. Run a shipped Isaac Lab tutorial headless and look at the PNG, before running any of our code, so "environment broken" and "our blind code broken" stay separable.
8. Run Spikes 1–4 (§7.2). Record the answers in §3 and rewrite §7.3 with real preset definitions. This is also where the driver acceptance in §3.4.1 gets its real answer — a clean determinism result on this host is worth more than the driver number.
9. Only then start on `stage_v1_tabletop.usd`.

**Track B — build what needs no decisions (local, start now):**

1. ~~OU sampler, `LatentSpec`, squash — with tier-0 tests.~~ **Done.** Plus the package scaffolding, the §4.2 import guard as an executable test, and the two pod scripts Track A needs.
2. `spikes/spike_api.py` — one standalone script that meets the whole Isaac API surface in a single boot, checks everything, and never fails fast. Written blind, run by Track A.
3. Then, against what the spike measured: the `SceneBackend` protocol, `MockSceneBackend`, the gates as library code, and the tier-1 contract suite.

> ### Closing note on sequencing
>
> The temptation will be to build the scene first, because it is the visible and satisfying part. Resist it. The determinism gate (§7.1) and the driver/GPU check (§8.2) are the two things that can invalidate weeks of work retroactively, and both can be settled in the first two days.
>
> The second temptation is to wait for the GPU before writing any code. Resist that too. Track B is most of the project and depends on none of the open decisions — which is precisely why the plan is arranged so that the version question blocks almost nothing.
