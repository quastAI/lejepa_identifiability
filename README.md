# Isaac Sim Identifiability Testbed

*Implementation plan for a LeJEPA linear-identifiability pipeline — photorealistic tabletop manipulation, OU-sampled positive pairs, RunPod GPU backend. Extensible from single-cube tabletop to full room-scale scenes.*

---

## 1. Executive Summary

The goal is a simulator that produces photorealistic image pairs *(x, x′)* whose ground-truth latents *(z, z′)* are drawn from a known Gaussian process with Ornstein–Uhlenbeck correlation, so that LeJEPA's linear-identifiability claim (Thm. 1–3 of Klindt, LeCun & Balestriero) can be tested on realistic observations rather than on 2D toy mixings or the low-fidelity DMC Reacher.

**Recommendation: Isaac Sim 5.x + Isaac Lab, running headless in a container on RunPod, with an RT-core GPU.** Isaac Sim satisfies every hard requirement: it exposes direct state writes (teleport-then-render, no policy rollout needed), it has an RTX path-tracing renderer for genuine photorealism, and its USD scene graph is built for exactly the incremental scene-growth path you described (tabletop → room → many objects).

> ### ⚠ Single most important infrastructure fact
>
> Isaac Sim's official requirements state that **GPUs without RT Cores — explicitly A100 and H100 — are not supported**. These are the default "big GPU" options on most cloud providers including RunPod. You must select an RTX-class GPU: RTX 4090, RTX 5090, L40S, RTX A6000, or RTX 6000 Ada. Budget and availability planning should start from this constraint, not from raw FLOPs.

Three decisions in this plan matter more than the rest, and all three are about protecting the *mathematical* validity of the experiment rather than about engineering convenience:

1. **Latent parameterisation via an absorbed squashing map.** The theory requires *z* to be exactly Gaussian with unbounded support, but joints and table surfaces are bounded. The fix is to keep *z ~ N(0, I)* genuinely unbounded and push a fixed deterministic squashing function into the mixing map *g*. Since the theory permits *g* to be an arbitrary nonlinear measurable map, this is free — and it eliminates the joint-limit wrapping artefact that degraded the paper's own Reacher results (their Table 2).
2. **Renderer determinism.** The theory assumes *x = g(z)* is a deterministic function. A path tracer with a moving random seed or a temporal denoiser makes *g* stochastic and history-dependent, which silently invalidates the setup. This must be engineered explicitly and verified numerically.
3. **Injectivity of *g* under occlusion and object symmetry.** If the arm hides the cube, or if a symmetric cube looks identical at 0° and 90° yaw, then distinct *z* map to identical *x* and no encoder can recover them. This is the deepest scientific risk in the project and needs design mitigation from day one.

**Effort:** roughly **2 weeks** to a first defensible identifiability number on the single-cube scene, and **4–5 weeks** to a fully extensible room-scale pipeline with a full sweep. Detailed phasing in Section 7.

---

## 2. Requirements Traceability

| Your requirement | How Isaac Sim satisfies it | Residual work / caveat |
|---|---|---|
| Latents of a simple scene (cube pick-and-place, fixed arm) | Franka Panda USD ships with Isaac Sim; cube is a primitive rigid body. Full joint and pose state is readable and writable. | You must *define* what counts as the latent vector — this is a modelling decision, not a default. See Section 4. |
| Extensible: load scenes, add objects later | USD composition (references, sublayers, payloads) is designed for this. ReplicaCAD / room-scale assets and the Omniverse asset library plug in directly. | Latent dimension *n* grows with each object; encoder output dim *m* must track it. The paper flags *m ≠ n* as an open problem (their Sec. 7). |
| Highly realistic rendering from the start | RTX path tracing with physically based materials, HDRI domes, area lights, DLSS. This is the strongest renderer of any robotics sim. | Path tracing costs 10–100× more per frame than rasterisation. Determinism must be forced. See Section 6. |
| OU sampling capability | `write_joint_state_to_sim` and `write_root_pose_to_sim` allow arbitrary state teleport without physics rollout. Render immediately after. | Need collision/validity policy for teleported states, and isotropic ρ across all latent dimensions. |
| RunPod GPU backend | Official NGC container `nvcr.io/nvidia/isaac-sim` runs headless on Linux; RunPod supports custom images and persistent network volumes. | RT-core GPU mandatory; host driver version must match; asset cache must be persisted or every pod start re-downloads gigabytes. |
| Cost | Free for internal and commercial R&D under NVIDIA's licence FAQ. Source is Apache 2.0. | Paid NVIDIA AI Enterprise licence only if you *redistribute* Isaac Sim or deliver it as a service to third parties. Publishing research and datasets does not trigger this. |

---

## 3. Architecture

### 3.1 Component stack

```
┌──────────────────────────────────────────────────────────────┐
│  Analysis layer  (any GPU / local)                           │
│   LeJEPA encoder training · SIGReg · R² · orthogonality      │
│   ε, δ, bound D + (ε+D)²  ·  planning eval                   │
└──────────────────────────▲───────────────────────────────────┘
                           │  HDF5 / WebDataset shards
                           │  (x, x′, z, z′, ρ, metadata)
┌──────────────────────────┴───────────────────────────────────┐
│  Dataset generation  (RunPod, RT-core GPU)                   │
│   ┌────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│   │ OU sampler │──▶│ Latent→State │──▶│ State writer     │   │
│   │  z, z′     │   │  map  φ      │   │ (teleport)       │   │
│   └────────────┘   └──────────────┘   └────────┬─────────┘   │
│                                                ▼             │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ Isaac Sim / Isaac Lab  ·  RTX path-traced camera     │   │
│   └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┴───────────────────────────────────┐
│  Scene layer  (USD)                                          │
│   stage_v1: table + Franka + cube                            │
│   stage_v2: + room shell (ReplicaCAD / Omniverse assets)     │
│   stage_v3: + N distractor & manipuland objects              │
└──────────────────────────────────────────────────────────────┘
```

The critical architectural property is that **the scene layer is swappable without touching the sampler**. The latent-to-state map *φ* is a registry of named handles (an articulation's joint vector, a rigid body's root pose) with per-handle squashing ranges. Adding an object to the scene means appending entries to that registry and incrementing *n*; the OU sampler, the writer and the renderer are unchanged.

### 3.2 Repository layout

```
identifiability/
├── scenes/
│   ├── stage_v1_tabletop.usd        # table + franka + cube
│   ├── stage_v2_room.usd            # references v1, adds room shell
│   └── materials/                   # PBR material library
├── src/
│   ├── latents/
│   │   ├── spec.py                  # LatentSpec: dims, handles, ranges
│   │   ├── squash.py                # φ: R^n → physical state (absorbed into g)
│   │   └── ou.py                    # OU pair sampler
│   ├── sim/
│   │   ├── app.py                   # SimulationApp bootstrap (headless)
│   │   ├── scene.py                 # scene load + handle resolution
│   │   ├── writer.py                # state teleport
│   │   └── render.py                # deterministic capture
│   ├── gen/
│   │   └── generate.py              # dataset driver, sharded output
│   └── eval/
│       ├── train_lejepa.py
│       └── metrics.py               # R², ortho err, ε, δ, bound
├── configs/                         # hydra/yaml: ρ, λ, n, render preset
├── docker/
│   └── Dockerfile                   # FROM nvcr.io/nvidia/isaac-sim:5.x
└── runpod/
    ├── template.md                  # pod config notes
    └── entrypoint.sh
```

---

## 4. Latent Design — The Scientific Core

This section is where the experiment is won or lost. Everything downstream is engineering.

### 4.1 The bounded-support problem and its clean solution

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

### 4.2 Stage-1 latent specification (n = 7)

| Index | Semantic factor | Physical handle | Squash φᵢ |
|---|---|---|---|
| 0–3 | Arm configuration (4 principal DoF) | Franka `panda_joint1,2,4,6` | cᵢ + rᵢ·tanh(zᵢ), rᵢ = 0.85·halfrange |
| 4 | Gripper aperture | `panda_finger_joint1/2` | 0.02·(1 + tanh(z₄))/2 |
| 5–6 | Cube position on table | cube root pose x, y | x₀ + 0.18·tanh(z₅), y₀ + 0.18·tanh(z₆) |

Cube *z*-height is held fixed at the table surface plus half the cube edge — it is a deterministic function of the other coordinates, not a free latent. Remaining Franka joints (3, 5, 7) are held at a fixed nominal pose in Stage 1 so that *n* stays small and the arm configuration is uniquely determined by the four active joints.

> ### ⚠ Do not include cube yaw in Stage 1
>
> A cube has 90° rotational symmetry about its vertical axis. Yaw values of 0 and π/2 render to *identical pixels*, making *g* non-injective and that latent dimension formally unrecoverable. Either (a) omit yaw, (b) restrict it to a range narrower than the symmetry period via the squash, or (c) replace the cube with a visually asymmetric object (a textured block, a mug, a toy) before adding yaw. Option (c) is the right long-term answer and should be the Stage-2 change.

### 4.3 The occlusion problem

This is the most serious threat to a clean result and it is intrinsic to manipulation scenes, not to Isaac Sim.

When the arm passes in front of the cube from the camera's viewpoint, the cube's position is no longer observable. Two different latent vectors — same arm pose, different hidden cube position — produce the same image. *g* is then not injective, *h = f ∘ g* cannot be a bijection, and Theorem 1's conclusion is unreachable in principle for those regions of latent space. The encoder will look like it is failing when in fact the data-generating process is degenerate.

Mitigations, in order of preference:

1. **Multi-view observation.** Render 2–3 cameras at well-separated viewpoints and concatenate (or stack as channels). Occlusion from all views simultaneously is rare. This is the cleanest fix and costs proportionally more render time.
2. **Camera placement.** A high, oblique, near-top-down view makes arm-over-cube occlusion much less frequent than an eye-level view. Cheap and worth doing regardless.
3. **Spatial separation in Stage 1.** Restrict the cube's squashed region to a table area the arm rarely sweeps over. Scientifically a bit of a dodge, but useful as a control condition to isolate occlusion as the cause of any measured gap.
4. **Measure it.** Render a segmentation pass alongside RGB, compute the fraction of cube pixels visible, and log it per sample. You can then report identifiability conditioned on visibility — which turns a confound into a finding.

I would run (1) + (2) as the default configuration and (4) always, because per-sample visibility is cheap to record and enormously clarifying when a number comes out low.

### 4.4 Isotropy of the transition

Appendix F of the paper makes a point that is easy to miss and expensive to get wrong: **the simultaneous (non-sequential) optimisation used by LeJEPA requires isotropic transitions**. If different latent dimensions have different autocorrelations ρ_α, the eigenvalue ordering interleaves, and the encoder recovers the *second* Hermite component of a slow latent instead of the first component of a fast one. Their formal condition is max_α K_α < 2 min_β K_β.

Because we sample *z* directly rather than rolling out a policy, this is trivially satisfiable: **use one scalar ρ for all dimensions**. Resist any temptation to give the cube a different correlation from the arm. Conversely, an *anisotropic* ρ sweep is a cheap and valuable ablation — it should reproduce the eigenvalue-interleaving failure and would be a genuine contribution beyond the paper's own experiments.

### 4.5 Validity policy for teleported states

Teleporting can produce arm-cube interpenetration. Note carefully that this does *not* break the theory: *g* only needs to be deterministic and injective, and an interpenetrating configuration is still a well-defined deterministic render. It looks physically wrong but is mathematically admissible.

Three options, and I recommend the first:

- **Accept interpenetration, log it.** Maximally faithful to the theory, zero sampling bias. Record a collision flag per sample so you can ablate later. Some reviewers will find the images odd; the flag lets you answer them with data.
- **Rejection sampling.** Discard colliding pairs. *This biases the latent distribution away from Gaussian* and therefore partially undermines Theorem 1's premise. Avoid unless the collision rate is tiny.
- **One settling step.** Write the state, step physics once, then render. This makes *g* depend on the physics solver and introduces a non-injective many-to-one map (different pre-settle states settle to the same post-settle state). Worst option for identifiability; useful only if you need physically plausible imagery for a figure.

---

## 5. OU Sampling Implementation

### 5.1 The sampler

Direct transcription of Eq. (1) of the paper. Trivial code, but note the batching and the single shared ρ.

```python
# src/latents/ou.py
import torch

def sample_ou_pairs(n: int, batch: int, rho: float,
                    device="cuda", generator=None):
    """z ~ N(0, I_n);  z' = rho*z + sqrt(1-rho^2)*eta,  eta ~ N(0, I_n).

    Single scalar rho across all dims -> isotropic transition, required
    for simultaneous (parallel) identifiability. See paper App. F.
    """
    z = torch.randn(batch, n, device=device, generator=generator)
    eta = torch.randn(batch, n, device=device, generator=generator)
    z_next = rho * z + (1.0 - rho ** 2) ** 0.5 * eta
    return z, z_next
```

Sanity assertions worth keeping in the test suite: `Cov(z) ≈ I`, `Cov(z′) ≈ I`, `Cov(z, z′) ≈ ρI`, and marginal normality per dimension (Shapiro–Wilk or a KS test). If any fails, the entire downstream analysis is meaningless.

### 5.2 Latent specification and squash

```python
# src/latents/spec.py
from dataclasses import dataclass
import torch

@dataclass
class Handle:
    kind: str          # "joint" | "root_xy" | "root_pose"
    asset: str         # scene key, e.g. "robot" / "cube"
    index: int | None  # joint index, if applicable
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

### 5.3 State writer

The Isaac Lab API verified against current documentation. Two points of care: root states are in *world* frame so the environment origin must be added, and internal buffers must be cleared after writing.

```python
# src/sim/writer.py
import torch

def write_latent_state(scene, spec, phi_vals: torch.Tensor):
    """Teleport the scene to the physical state encoded by phi_vals.
    phi_vals: [B, n] already squashed. No physics stepping."""
    robot = scene["robot"]
    cube  = scene["cube"]

    # --- arm joints -------------------------------------------------
    joint_pos = robot.data.default_joint_pos.clone()
    for i, h in enumerate(spec.handles):
        if h.kind == "joint":
            joint_pos[:, h.index] = phi_vals[:, i]
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # --- cube root pose ---------------------------------------------
    root = cube.data.default_root_state.clone()      # [B, 13]
    for i, h in enumerate(spec.handles):
        if h.kind == "root_xy":
            root[:, h.index] = phi_vals[:, i]        # index 0=x, 1=y
    root[:, 0:3] += scene.env_origins                # world frame offset
    cube.write_root_pose_to_sim(root[:, :7])
    cube.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]))

    # --- clear internal buffers/caches -------------------------------
    scene.reset()
```

> ### ⚠ API version sensitivity
>
> Recent Isaac Lab releases are migrating `write_joint_state_to_sim` toward fused index/mask variants (`write_joint_state_to_sim_index`, `write_joint_state_to_sim_mask`), with the old name deprecated. Pin your Isaac Lab version in the Dockerfile and check the CHANGELOG before upgrading. Also note that assets configured with `fix_root_link` or `kinematic_enabled` can silently ignore root-pose writes — a known and frequently reported gotcha. Verify with an assertion that the read-back pose matches what you wrote.

### 5.4 Generation loop

```python
# src/gen/generate.py  (sketch)
for shard in range(n_shards):
    z, z_next = sample_ou_pairs(spec.n, batch, rho, generator=g)

    write_latent_state(scene, spec, spec.squash(z))
    x  = render_deterministic(camera, sim)

    write_latent_state(scene, spec, spec.squash(z_next))
    x2 = render_deterministic(camera, sim)

    store(shard, x=x, x_next=x2, z=z, z_next=z_next,
          visibility=seg_visibility(camera),   # occlusion diagnostic
          collision=collision_flag(scene),     # validity diagnostic
          rho=rho, seed=seed)
```

---

## 6. Rendering: Photorealism vs. Determinism

### 6.1 The determinism requirement

> ### ⚠⚠ This will silently corrupt your results if ignored
>
> Theorem 1 assumes *x = g(z)* is a deterministic function of *z*. Two failure modes in a path-traced renderer break this:
>
> **(a) Sampling noise.** Monte Carlo path tracing with a free-running seed gives a different image for the same state on every call. The encoder then sees *x = g(z) + noise*, which shifts the setup into a different (noisy-observation) regime not covered by the theorem.
>
> **(b) Temporal accumulation.** DLSS ray reconstruction and RTX denoisers accumulate across frames. This makes the render of *x′* depend on the previously rendered *x* — a direct, artificial correlation between the two members of every positive pair. An encoder can exploit this leakage to score well on alignment without learning anything about the world. This is the single most dangerous artefact in the whole pipeline.

Required mitigations:

- Fix the render seed per sample (deterministic function of the sample index), or use enough samples-per-pixel that residual MC noise is below quantisation.
- Disable temporal denoising / DLSS-RR, or force an accumulation reset between every capture. Verify by rendering the same state twice, in different orders, and asserting bitwise or near-bitwise equality.
- Render *x* and *x′* with independent accumulation buffers, never back-to-back within one accumulating sequence.

**Determinism acceptance test (run before generating any dataset):**

```
render(z_a) -> A1 ;  render(z_b) -> B1
render(z_b) -> B2 ;  render(z_a) -> A2
assert mean_abs_diff(A1, A2) < 1e-3   # order-independence
assert mean_abs_diff(B1, B2) < 1e-3   # no temporal leakage
```

If this fails, nothing else in the project is worth running. Treat it as a hard gate in CI.

### 6.2 Render presets

| Preset | Mode | Speed (indicative) | Use |
|---|---|---|---|
| `debug` | RTX Real-Time, no DLSS | fast | Scene authoring, camera placement, pipeline smoke tests |
| `standard` | RTX Real-Time, high quality, fixed seed | moderate | Main dataset generation; good realism at tractable cost |
| `photoreal` | RTX Path Tracing, fixed SPP, denoiser off or spatial-only | slow (order 1–10 fps) | Headline figures; a smaller high-fidelity dataset for the realism ablation |

A genuinely interesting experiment falls out of having both presets: **does linear identifiability degrade as rendering realism increases?** Same latents, same ρ, same encoder, three render presets. If *R²* drops with realism, that is a real and publishable finding about the gap between the theory's idealisation and realistic observation. Cost is one extra dataset generation run.

> ### ⚠ Verify before committing to path tracing at scale
>
> Isaac Lab's `TiledCamera` (the fast multi-environment renderer) and full path tracing may not be compatible — tiled rendering targets the rasterised/real-time RTX path. Confirm early whether `photoreal` requires the standard `Camera` sensor with a single environment, because that changes throughput by an order of magnitude and therefore your GPU-hour budget. Treat this as a Phase-1 spike.

### 6.3 Scene and lighting setup

- **Lighting:** HDRI dome plus one or two area lights. Fixed across the dataset — lighting variation is *not* a latent unless you deliberately make it one.
- **Materials:** proper PBR (roughness, metallic, normal maps) on table, arm and cube. Flat diffuse materials will make the "photorealistic" claim indefensible.
- **Camera:** fixed extrinsics and intrinsics, high oblique angle. Resolution 128×128 or 224×224 — a resolution ablation is cheap and worth having.
- **Deliberate non-latents:** anything you do not put in the latent vector must be held exactly constant, or it becomes uncontrolled nuisance variation that the encoder must marginalise. Fix random seeds for any procedural material or placement.

---

## 7. RunPod Infrastructure

### 7.1 GPU selection

| GPU | RT cores | VRAM | Verdict for this project |
|---|---|---|---|
| A100 / H100 | **No** | 40–80 GB | **Unsupported.** Officially excluded by NVIDIA. Do not provision, regardless of price or availability. |
| RTX 4090 | Yes | 24 GB | Best price/performance. Ample for single-scene generation. **Default choice.** |
| L40S | Yes | 48 GB | Datacenter Ada part; more VRAM headroom for room-scale scenes. Best choice for Stage 2/3. |
| RTX A6000 | Yes | 48 GB | Ampere; works, slower RT than Ada. Good fallback on availability. |
| RTX 6000 Ada / 5090 | Yes | 32–48 GB | Excellent, usually pricier or scarcer on RunPod. |

Isaac Sim's published minimum is an RTX 4080 with 16 GB VRAM (recent versions; older docs said RTX 3070/8 GB), recommended RTX 5080, ideal RTX PRO 6000 Blackwell. 32 GB system RAM minimum, 64 GB preferred. Plan on **RTX 4090 for Stage 1, L40S for Stage 2+**.

### 7.2 Pre-flight checks (do these before writing any code)

1. **Driver version.** Isaac Sim 5.1 expects Linux driver 580.65.06 or compatible. Spin up a candidate RunPod instance and run `nvidia-smi`. If the host driver is older, you must either choose a different RunPod GPU pool or pin an older Isaac Sim release. *This is the most common cause of a failed first day.*
2. **RT core presence.** Confirm the allocated GPU is the one advertised, not a substituted datacenter part.
3. **NGC account and API key.** Free; required for `docker login nvcr.io`. Set up before you need it.
4. **Outbound network.** Isaac Sim streams assets from an Omniverse CloudFront endpoint on first run. Confirm the pod has unrestricted egress.
5. **Headless render smoke test.** Render one frame to disk and inspect it. Do not proceed until an image file exists.

### 7.3 Container and volumes

```dockerfile
# docker/Dockerfile
FROM nvcr.io/nvidia/isaac-sim:5.1.0

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ACCEPT_EULA=YES

# Isaac Lab (pin the version)
RUN git clone --depth 1 --branch v2.x https://github.com/isaac-sim/IsaacLab.git /workspace/IsaacLab
WORKDIR /workspace/IsaacLab
RUN ./isaaclab.sh --install

COPY . /workspace/identifiability
WORKDIR /workspace/identifiability
RUN /isaac-sim/python.sh -m pip install -r requirements.txt
```

> ### Persistent volume layout — do not skip this
>
> RunPod pods are ephemeral. Without a network volume, every pod start re-downloads gigabytes of Omniverse assets and recompiles shader caches, costing 10–20 minutes of paid GPU time per session. Mount a **200–500 GB network volume** and bind these paths into it:
>
> ```
> /isaac-sim/kit/cache          → shader / kit cache
> /root/.cache/ov               → Omniverse asset cache
> /root/.cache/pip              → pip cache
> /workspace/data               → generated datasets
> /workspace/scenes             → your USD stages
> ```
>
> Note that since Isaac Sim 5.1 containers run as a **non-root user by default**. Older tutorials' compose files assume root and will produce permission errors on mounted volumes. Set ownership in the entrypoint.

### 7.4 Operating pattern and cost

Treat RunPod as a batch renderer, not a development environment:

- Develop the sampler, spec, and analysis code **locally with the sim mocked out**. Only the render step needs the GPU.
- Use a small persistent pod (or CPU pod) for code sync and a large GPU pod launched only for generation runs.
- Generate datasets to the network volume, then pull them down (or train in place) — a generated dataset is reused across many encoder training runs, so generation is a one-off cost per configuration.
- **Avoid spot/interruptible instances for long generation runs** unless your generator checkpoints per shard. With per-shard checkpointing, spot becomes attractive and roughly halves cost.

*Indicative RunPod pricing at time of writing is roughly $0.3–0.7/hr for RTX 4090 and $0.8–1.2/hr for L40S, with network volumes around $0.05–0.10/GB/month. Verify current rates — these move. A realistic first-month budget for development plus several dataset generations and sweeps is in the low hundreds of dollars, dominated by the `photoreal` preset runs if you use them heavily.*

---

## 8. Phased Implementation

| Phase | Deliverable | Contents | Effort |
|---|---|---|---|
| **0** | Pre-flight | NGC account; RunPod GPU availability + driver version check; container pulls and launches headless; one frame rendered to disk. | 0.5–1 day |
| **1** | Infrastructure | Dockerfile pinned; network volume with cache bind mounts; entrypoint handling non-root permissions; repeatable pod launch. Spike: path tracing vs. TiledCamera compatibility. | 1–2 days |
| **2** | Scene v1 | `stage_v1_tabletop.usd`: table, Franka, cube, PBR materials, HDRI + area lights, camera rig (2–3 views). Debug-preset renders look right. | 1–2 days |
| **3** | Latent layer | `LatentSpec`, squash map, state writer. Read-back assertions confirming every write landed. Collision and visibility diagnostics. | 2 days |
| **4** | Determinism gate | Deterministic capture path; the order-independence acceptance test in §6.1 passing in CI for all three presets. | 1–2 days |
| **5** | OU generator | Sampler with statistical assertions; sharded HDF5/WebDataset writer storing (x, x′, z, z′, visibility, collision, ρ, seed); per-shard checkpointing. | 2 days |
| **6** | First dataset | ~100k pairs at `standard` preset, ρ = 0.95. Visual audit of a random sample grid. | 0.5–1 day compute |
| **7** | Analysis | LeJEPA/SIGReg training on the dataset; metrics: R²(h→z), R²(z→h), ‖Q̂ᵀQ̂−I‖_F/√n, ε, δ, bound D + (ε+D)². **First real number.** | 2–3 days |
| **8** | Sweeps | ρ ∈ {0.3 … 0.99}; λ grid; render-realism ablation; gennorm α latent-distribution sweep (converse test); anisotropic-ρ ablation. | 3–5 days + compute |
| **9** | Scene v2 (room) | USD reference of a room shell around stage v1; relight; re-validate determinism and occlusion statistics; regenerate and re-measure. | 4–6 days |
| **10** | Scene v3 (objects) | Add manipulands one at a time; each adds 2–3 latent dims. Study identifiability vs. *n*, and the *m ≠ n* regime the paper leaves open. | ongoing |

**Critical path to a first defensible result: Phases 0–7, approximately two weeks.** Phases 8–10 are where the scientific contribution lives.

---

## 9. Validation Protocol

### 9.1 Pipeline-correctness gates

Each must pass before the next phase is trusted:

- **Sampler:** Cov(z) ≈ I, Cov(z, z′) ≈ ρI, per-dim normality tests pass.
- **Writer:** state read back from the sim matches what was written, to solver tolerance, for every handle.
- **Renderer:** order-independence test of §6.1 passes.
- **Injectivity proxy:** nearest-neighbour check — the fraction of image pairs whose pixel distance is near zero while their latent distance is large should be negligible. A non-trivial fraction means *g* is not injective and you have an occlusion or symmetry problem to fix *before* blaming the encoder.
- **Trivial-baseline check:** a linear probe from raw pixels to *z* should score poorly (confirming the mixing is genuinely nonlinear, analogous to the paper's *R²*(x→z) ≈ 0.73–0.78 column). If raw pixels already predict *z* linearly, the task is too easy to be informative.

### 9.2 Scientific measurements

Mirror the paper's metric set so results are directly comparable to their Tables 1 and 2:

- Bidirectional linear *R²* between *h(z)* and *z*, fitted on a train split and scored on held-out evaluation samples.
- Per-dimension *R²*, to expose anisotropy (their Reacher shoulder-vs-wrist asymmetry).
- Orthogonality error ‖Q̂ᵀQ̂ − I‖_F/√n.
- Approximate-bound verification: compute ε, δ, D = δ/(2ρ(1−ρ)), and check the measured recovery error falls below D + (ε+D)².
- Identifiability conditioned on cube visibility — unique to this setup and, in my view, the most interesting number you will produce.

### 9.3 Experiments that go beyond the paper

These are the reasons to build on Isaac rather than reuse their Reacher:

1. **Realism vs. identifiability.** Does photorealistic rendering degrade linear identifiability relative to simple rendering, holding latents fixed? Nobody has measured this.
2. **Occlusion vs. identifiability.** With per-sample visibility logged, quantify how non-injectivity of *g* erodes recovery. Directly relevant to every real manipulation system.
3. **Scaling in object count.** Track *R²* as objects are added and *n* grows in a *realistic* scene, rather than in a synthetic RealNVP mixing.
4. **The m ≠ n regime.** The paper explicitly names this as an open problem with consequences for JEPA design. A scene where you can add and remove objects at will is the natural instrument to study it.
5. **Anisotropic ρ.** Deliberately violate isotropy and confirm the eigenvalue-interleaving prediction of their Appendix F.

---

## 10. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Provisioned an A100/H100 and Isaac Sim will not render | Critical | Hard rule: RT-core GPUs only. Verify in Phase 0 before any other work. |
| Temporal denoiser leaks information between x and x′ | Critical | Determinism acceptance test as a CI gate; disable DLSS-RR / reset accumulation between captures. |
| Occlusion makes g non-injective; results look like encoder failure | High | Multi-view cameras, high oblique placement, per-sample visibility logging, injectivity proxy check. |
| Cube rotational symmetry hides a latent dimension | High | Omit yaw in Stage 1; switch to a visually asymmetric object before introducing orientation latents. |
| Bounded joints break Gaussianity of z | High | Absorbed tanh squash (§4.1). Never clip, never wrap. |
| Host driver too old for the chosen Isaac Sim version | Medium | Phase 0 driver check; pin Isaac Sim version to match the available RunPod pool. |
| Path tracing throughput makes large datasets infeasible | Medium | Phase-1 spike on TiledCamera compatibility; use `standard` for the main dataset and `photoreal` only for a smaller ablation set. |
| Ephemeral pods re-download assets every session | Medium | Persistent network volume with cache bind mounts (§7.3). |
| Spot instance preempted mid-generation | Low | Per-shard checkpointing; resume from last completed shard. |
| Isaac Lab API churn breaks the writer | Low | Pin version in Dockerfile; read-back assertions catch silent write failures immediately. |

---

## 11. Immediate Next Actions

1. Create a free NVIDIA NGC account and generate an API key.
2. Launch a candidate RunPod RTX 4090 pod. Run `nvidia-smi` and record the driver version. Confirm it meets the requirement for the Isaac Sim release you intend to pin.
3. Pull and launch `nvcr.io/nvidia/isaac-sim` headless. Render a single frame of the default scene to disk and look at it.
4. Provision the network volume and verify the cache bind mounts survive a pod restart.
5. Run the path-tracing / TiledCamera compatibility spike — this determines your throughput budget and should be answered before scene authoring begins.
6. Only then start on `stage_v1_tabletop.usd`.

> ### Closing note on sequencing
>
> The temptation will be to build the scene first, because it is the visible and satisfying part. Resist it. The determinism gate (§6.1) and the driver/GPU check (§7.2) are the two things that can invalidate weeks of work retroactively, and both can be settled in the first two days. Everything else is recoverable.