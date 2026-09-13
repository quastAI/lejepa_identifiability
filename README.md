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

### 3.2 Deferred — recorded here so that de-pinning does not lose the question

| Open question | What decides it | When |
|---|---|---|
| **Isaac Sim release** (5.1 / 6.0 / later) | The NVIDIA driver version actually present on the candidate GPU pool, checked with `nvidia-smi`, against that release's requirements page. Isaac Sim 6.0 is an early developer release; 5.1 is the conservative choice. Decide from evidence, not preference. | Phase 0 |
| **Isaac Lab release** | Follows from the Isaac Sim release. Isaac Lab 3.x targets Isaac Sim 6.x; Isaac Lab 2.3 pairs with Isaac Sim 5.1. | Phase 0, immediately after the above |
| **Python version** | Follows the Isaac Sim release (it has changed between releases). Never pin independently. | Phase 0 |
| **GPU model** | RT-core presence, VRAM headroom for the target scene stage, and actual availability/price at provisioning time. See §8.1 for selection criteria rather than a verdict. | Phase 0 |
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

1. **No Isaac imports at module top level, anywhere in reusable code.** Sim-touching modules import Isaac *inside* functions. A top-level `import isaaclab` in `src/sim/writer.py` would kill `pytest` at collection time on a developer machine — where, per §1, Isaac cannot be installed at all.
2. **Entry points are scripts; libraries are pure.** `src/gen/generate.py` boots the app and *then* imports. `src/latents/*` and `src/eval/*` never touch Isaac.
3. **One app per process.** Isaac-dependent tests run in a single session under Isaac's own interpreter, with a session-scoped fixture owning the app.

### 4.3 The backend seam

Defined in `src/sim/backend.py` — pure Python, zero Isaac imports:

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

1. It gives a local development and CI loop for the OU sampler, the squash, the shard writer, the generation driver, the metrics and the LeJEPA training code — which is most of the project.
2. It gives the correctness gates a **known-good reference**. A determinism gate that has never been observed to pass on something that should pass is not evidence. Running the same contract suite against the mock (must pass) and against Isaac (must also pass) is what makes the Isaac result meaningful.

### 4.4 The Isaac API surface we actually need

Deliberately small. This is the complete list; anything outside it is out of scope.

**Boot** — `src/sim/app.py`, the only module permitted to do this
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

**Render control** — drops below Isaac Lab into `carb`, contained entirely in `src/sim/render.py`
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
identifiability/
├── scenes/
│   ├── stage_v1_tabletop.usd        # table + franka + cube
│   ├── stage_v2_room.usd            # references v1, adds room shell
│   └── materials/                   # PBR material library
├── src/
│   ├── latents/                     # pure python, no Isaac
│   │   ├── spec.py                  # LatentSpec: dims, handles, ranges
│   │   ├── squash.py                # φ: R^n → physical state (absorbed into g)
│   │   └── ou.py                    # OU pair sampler
│   ├── sim/
│   │   ├── backend.py               # SceneBackend protocol — pure python
│   │   ├── mock.py                  # MockSceneBackend — analytic, local
│   │   ├── app.py                   # SimulationApp bootstrap (headless)
│   │   ├── scene.py                 # scene load + handle resolution
│   │   ├── writer.py                # state teleport
│   │   └── render.py                # deterministic capture; the only carb consumer
│   ├── gen/
│   │   └── generate.py              # dataset driver, sharded output
│   └── eval/
│       ├── train_lejepa.py
│       └── metrics.py               # R², ortho err, ε, δ, bound
├── tests/
│   ├── test_ou.py, test_squash.py   # tier 0 — pure
│   ├── contract/                    # tier 1 — parametrized over backend
│   └── isaac/                       # tier 2 — remote only
├── configs/                         # hydra/yaml: ρ, λ, n, render preset
├── docker/
│   └── Dockerfile
└── infra/
    ├── pod.md                       # pod config notes
    └── entrypoint.sh
```

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
# src/latents/ou.py
import torch

def sample_ou_pairs(n: int, batch: int, rho: float,
                    device="cpu", generator=None):
    """z ~ N(0, I_n);  z' = rho*z + sqrt(1-rho^2)*eta,  eta ~ N(0, I_n).

    Single scalar rho across all dims -> isotropic transition, required
    for simultaneous (parallel) identifiability. See paper App. F.
    """
    z = torch.randn(batch, n, device=device, generator=generator)
    eta = torch.randn(batch, n, device=device, generator=generator)
    z_next = rho * z + (1.0 - rho ** 2) ** 0.5 * eta
    return z, z_next
```

Sanity assertions kept in the test suite: `Cov(z) ≈ I`, `Cov(z′) ≈ I`, `Cov(z, z′) ≈ ρI`, and marginal normality per dimension. If any fails, the entire downstream analysis is meaningless.

### 6.2 Latent specification and squash

```python
# src/latents/spec.py
from dataclasses import dataclass
import torch

@dataclass
class Handle:
    kind: str          # "joint" | "root_xy" | "root_pose"
    asset: str         # scene key, e.g. "robot" / "cube"
    name: str | None   # joint name, resolved to an index at bind time
    center: float
    radius: float      # tanh amplitude

class LatentSpec:
    """Registry mapping latent dims -> physical handles.
    Adding an object == appending handles. n grows, nothing else changes."""
    def __init__(self, handles: list[Handle]):
        self.handles = handles

    @property
    def n(self) -> int:
        return len(self.handles)

    def squash(self, z: torch.Tensor) -> torch.Tensor:
        """phi: R^n -> physical values. Monotonic bijection per dim, so
        injectivity of g is preserved. Absorbed into the mixing map."""
        c = torch.tensor([h.center for h in self.handles], device=z.device)
        r = torch.tensor([h.radius for h in self.handles], device=z.device)
        return c + r * torch.tanh(z)
```

Handles carry joint **names**, not indices. Indices are resolved once at bind time via `robot.find_joints(...)`; hardcoded indices are a silent-corruption hazard because they change with asset revisions.

### 6.3 State writer

Sketch, structured to match §4.5. Two points of care: root poses are in **world** frame so the environment origin must be added, and the root pose is a 7-vector whose orientation is a **normalised (w, x, y, z) quaternion** — not something to leave uninitialised while writing only x and y.

```python
# src/sim/writer.py  (sketch — imports Isaac lazily, inside the function)
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
> There is an open report of `write_root_pose_to_sim` leaving objects frozen at the environment origin on Isaac Sim 5.0 despite working on 4.5, with no official resolution ([IsaacSim issue #251](https://github.com/isaac-sim/IsaacSim/issues/251)). A failure of this kind is silent: the dataset generates normally and the cube latents are simply noise. **The read-back assertion in §4.5 is therefore not optional and not a debug aid — it is a correctness gate that runs on every sample during development and on a sampled basis in production runs.**
>
> Assets configured with `fix_root_link` or `kinematic_enabled` can also silently ignore root-pose writes. Same gate catches it.
>
> For the record, `write_joint_state_to_sim` is **not** deprecated as of `main`; it takes `joint_ids`/`env_ids` directly. Deprecations elsewhere in the asset API (`set_external_force_and_torque`, `write_joint_friction_to_sim`) do not affect this project.

### 6.4 Generation loop

```python
# src/gen/generate.py  (sketch — backend is a SceneBackend, real or mock)
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
| **4** | **`TiledCamera` × path tracing.** Does tiled rendering work in the mode chosen by Spike 1? | Worth an order of magnitude in throughput. Note the reported behaviour is the *opposite* of the intuitive guess: tiled render products have been reported black in RTX Real-Time while rendering normally under path tracing ([IsaacSim issue #367](https://github.com/isaac-sim/IsaacSim/issues/367)), with single-camera products fine in both. Test, do not assume. |

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

The GPU model is a deferred decision (§3.2). What is fixed are the criteria:

| Criterion | Rule |
|---|---|
| RT cores | **Mandatory.** A100 and H100 are explicitly unsupported. Verify the allocated GPU is the advertised part, not a substituted datacenter card. |
| VRAM | Enough headroom for the target scene stage. Room-scale (Stage 2/3) wants more than the tabletop scene. |
| Driver | Must satisfy the requirements page *of the Isaac Sim release being pinned*. This is a matching problem between the release and the provider's pool, not a fixed number — check both at provisioning time. |
| Availability / price | Real constraints; an ideal part that is never available is not a choice. |

Practically, this narrows to RTX-class parts (GeForce RTX 40/50 series, L40S, RTX A6000, RTX 6000 Ada, RTX PRO 6000 Blackwell). Pick from availability at Phase 0, record the choice here.

### 8.2 Pre-flight checks (before writing any Isaac-facing code)

1. **Driver version.** Spin up a candidate pod and run `nvidia-smi`. **This reading decides the Isaac Sim release** (§3.2), not the other way round. Getting this backwards is the most common cause of a failed first day.
2. **RT core presence.** Confirm the allocated GPU is what was advertised.
3. **NGC account and API key.** Free; required for `docker login nvcr.io`. Set up before it is needed.
4. **Outbound network.** Isaac Sim streams assets from an Omniverse CDN endpoint on first run. Confirm unrestricted egress.
5. **Headless render smoke test.** Render one frame to disk and inspect it. Do not proceed until an image file exists.

### 8.3 Container and volumes

The image tag and Isaac Lab branch are intentionally left as build arguments, resolved by the Phase-0 driver check:

```dockerfile
# docker/Dockerfile
ARG ISAAC_SIM_TAG          # resolved in Phase 0 from the driver check
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_TAG}

ARG ISAAC_LAB_REF          # the Isaac Lab release matching ISAAC_SIM_TAG

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ACCEPT_EULA=YES

RUN git clone --depth 1 --branch ${ISAAC_LAB_REF} \
        https://github.com/isaac-sim/IsaacLab.git /workspace/IsaacLab
WORKDIR /workspace/IsaacLab
RUN ./isaaclab.sh --install

COPY . /workspace/identifiability
WORKDIR /workspace/identifiability
```

Once Phase 0 resolves both, record the values **here in this document** as well as in the build, so the plan and the build cannot drift.

> ### Persistent volume layout — do not skip this
>
> Rented pods are ephemeral. Without a network volume, every pod start re-downloads gigabytes of Omniverse assets and recompiles shader caches, costing paid GPU time on every session. Mount a network volume and bind the Kit/shader cache, the Omniverse asset cache, the pip cache, generated datasets, and the USD stages into it.
>
> Note that recent Isaac Sim containers run as a **non-root user by default**. Older tutorials' compose files assume root and will produce permission errors on mounted volumes. Set ownership in the entrypoint.

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
| **0** | Pre-flight | NGC account; candidate pod; `nvidia-smi` driver reading; **resolve the Isaac Sim / Isaac Lab / Python versions from that reading and record them in §3**; container pulls and launches headless; one frame rendered to disk. | 0.5–1 day |
| **0b** | Pure layers *(parallel, local, no GPU)* | `SceneBackend` protocol, `MockSceneBackend`, OU sampler, `LatentSpec` + squash, shard writer, tier-0 and tier-1-mock tests green. **Blocked on nothing.** | 2 days |
| **1** | Infrastructure + spikes | Dockerfile with resolved build args; network volume with cache bind mounts; entrypoint handling non-root permissions; repeatable pod launch. **Spikes 1–4 (§7.2) answered and recorded in §3.** | 2–3 days |
| **2** | Scene v1 | `stage_v1_tabletop.usd`: table, Franka, cube, PBR materials, HDRI + area lights, camera rig (2–3 views). Debug-preset renders look right. | 1–2 days |
| **3** | Latent layer, real backend | `IsaacSceneBackend`: handle resolution by name, state writer, read-back assertions confirming every write landed. Collision and visibility diagnostics. Tier-1 contract suite green against Isaac. | 2 days |
| **4** | Determinism gate | Deterministic capture path; the §7.1 acceptance test passing in CI for every preset intended for dataset use. | 1–2 days |
| **5** | OU generator | Sharded writer storing (x, x′, z, z′, visibility, collision, ρ, seed, intrinsics); per-shard checkpointing. | 2 days |
| **6** | First dataset | ~100k pairs at `standard`, ρ = 0.95 (a starting point, not a finding). Visual audit of a random sample grid. | 0.5–1 day compute |
| **7** | Analysis | LeJEPA/SIGReg training; metrics: R²(h→z), R²(z→h), ‖Q̂ᵀQ̂−I‖_F/√n, ε, δ, bound D + (ε+D)². **First real number.** | 2–3 days |
| **8** | Sweeps | ρ ∈ {0.3 … 0.99}; λ grid; render-realism ablation; gennorm α latent-distribution sweep (converse test); anisotropic-ρ ablation; resolution ablation. | 3–5 days + compute |
| **9** | Scene v2 (room) | USD reference of a room shell around stage v1; relight; re-validate determinism and occlusion statistics; regenerate and re-measure. | 4–6 days |
| **10** | Scene v3 (objects) | Add manipulands one at a time; each adds 2–3 latent dims. Study identifiability vs. *n*, and the *m ≠ n* regime the paper leaves open. | ongoing |

**Critical path to a first defensible result: Phases 0–7, approximately two weeks.** Phases 8–10 are where the scientific contribution lives.

Note that Phase 0b has no dependency on Phase 0 — the pure layers and the mock can be built while the GPU question is still open. This is the practical payoff of the seam: the version decision blocks almost nothing.

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
- Tier 1 is the payoff of the seam: the §10.1 gates become executable contracts that the mock **must pass** and the Isaac backend **must also pass**. A gate only observed against one implementation is weak evidence.

---

## 11. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Temporal denoiser leaks information between x and x′ | Critical | Determinism acceptance test as a CI gate (§7.1). Aggravated by being the *default* renderer behaviour — must be actively disabled, then measured. |
| Silent state-write failure (object frozen at env origin, `fix_root_link`, kinematic flags) | Critical | Read-back assertion on every sample in development (§6.3). Known open upstream issue with no fix. |
| Provisioned a GPU without RT cores | Critical | Hard rule: RT-core GPUs only. Verify in Phase 0 before any other work. |
| No local runtime → slow, blind iteration on Isaac code | High | `MockSceneBackend` and the tier-0/tier-1 split (§4.3, §10.3); Phase 0b runs in parallel with Phase 0. |
| Occlusion makes g non-injective; results look like encoder failure | High | Multi-view cameras, high oblique placement, per-sample visibility logging, injectivity proxy check. |
| Cube rotational symmetry hides a latent dimension | High | Omit yaw in Stage 1; switch to a visually asymmetric object before introducing orientation latents. |
| Bounded joints break Gaussianity of z | High | Absorbed tanh squash (§5.1). Never clip, never wrap. |
| Driver / Isaac Sim release mismatch | Medium | Phase 0 driver check *drives* the version choice (§8.2). Record the resolved versions in §3. |
| Rendering throughput makes large datasets infeasible | Medium | Spikes 1, 3, 4 before committing to a preset; `standard` for the main dataset, `photoreal` only for a smaller ablation set. |
| Physics step turns out to be required before rendering | Medium | Spike 2. If confirmed, revisit §5.5 — it changes the validity policy, not just the code. |
| Ephemeral pods re-download assets every session | Medium | Persistent network volume with cache bind mounts (§8.3). |
| Spot instance preempted mid-generation | Low | Per-shard checkpointing; resume from last completed shard. |
| Isaac Lab API churn breaks the writer | Low | Pin the resolved version; read-back assertions catch silent write failures immediately. |

---

## 12. Immediate Next Actions

Two tracks, and they are independent.

**Track A — resolve the unknowns (needs a GPU):**

1. Create a free NVIDIA NGC account and generate an API key.
2. Launch a candidate RT-core pod. Run `nvidia-smi`, record the driver version, and **from that reading choose the Isaac Sim release, the matching Isaac Lab release and the Python version. Record all three in §3.**
3. Pull and launch the container headless. Render a single frame of the default scene to disk and look at it.
4. Provision the network volume and verify the cache bind mounts survive a pod restart.
5. Run Spikes 1–4 (§7.2). Record the answers in §3 and rewrite §7.3 with real preset definitions.
6. Only then start on `stage_v1_tabletop.usd`.

**Track B — build what needs no decisions (local, start now):**

1. `SceneBackend` protocol and `MockSceneBackend`.
2. OU sampler, `LatentSpec`, squash — with tier-0 tests.
3. The tier-1 contract suite, green against the mock.
4. The generation driver and shard writer, exercised end-to-end against the mock.

> ### Closing note on sequencing
>
> The temptation will be to build the scene first, because it is the visible and satisfying part. Resist it. The determinism gate (§7.1) and the driver/GPU check (§8.2) are the two things that can invalidate weeks of work retroactively, and both can be settled in the first two days.
>
> The second temptation is to wait for the GPU before writing any code. Resist that too. Track B is most of the project and depends on none of the open decisions — which is precisely why the plan is arranged so that the version question blocks almost nothing.
