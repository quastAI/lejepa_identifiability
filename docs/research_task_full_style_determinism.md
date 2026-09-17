# Research task: non-deterministic rendering of material/light/camera writes in Isaac Sim (RTX PathTracing)

## TL;DR

On a fixed scene state, rendering **twice in a row with nothing changed** gives a
small (~1.7–2.6 mean-abs-diff per pixel channel, out of 255) but **exactly
reproducible** difference — for material, light, and camera-pose writes only. A
pure geometry transform write on the same scene, same pipeline, same everything
else, gives **exactly 0.0** difference. Four targeted fixes were each tried and
cleanly falsified. We need a diagnosis of the mechanism and either a fix or a
confirmed dead end, so we know whether to keep pursuing this render path at all.

This came out of a research project building an Isaac Sim data-generation
pipeline for testing LeJEPA's linear-identifiability claim. The core requirement
is that rendering a given simulator state must be a **deterministic** function —
the same state always produces bit-identical pixels, regardless of render order
or history. That's a hard requirement, not a nice-to-have: any residual
frame-to-frame correlation is exactly the kind of shortcut a contrastive/
self-supervised encoder can exploit to "cheat" without learning anything about
the underlying latent state, which would invalidate the experiment. Physics-state
writes (joint angles, rigid-body poses) already passed this bar cleanly in an
earlier spike. This task is about a *different* set of write paths that don't.

## Environment (exact, as measured on the affected host)

- **Isaac Sim**: `6.0.1-rc.7+release.42383.32955d8d.gl`
- **Isaac Lab**: image tag `nvcr.io/nvidia/isaac-lab:3.0.0-beta2-post1` (= `3.0.0-beta2.patch1`); the `isaaclab` Python package itself reports `__version__ == "6.1.16"` (this differs from the product-level tag above — both are reported as measured, not reconciled, and may just be separate version schemes)
- **Python**: 3.12.13 (Isaac's bundled interpreter)
- **torch**: `2.10.0+cu128`, CUDA runtime `12.8`
- **GPU**: NVIDIA GeForce RTX 4090, 24 GB, compute capability 8.9
- **Driver**: `595.91.07` — on the 595 branch, which [IsaacSim #537](https://github.com/isaac-sim/IsaacSim/issues/537) reports as sometimes breaking CUDA detection relative to the 580 branch (this driver is otherwise functioning: CUDA is visible, physics and rendering both run)
- **OS**: Ubuntu 24.04.3 LTS, kernel 6.8.0-138-generic
- **Platform**: a rented GPU pod (RunPod), headless, no local display

Printed on every single boot, unprompted, and never investigated as a lead:

```
[Warning] [rtx.hydra] RenderDelegate : Warning /rtx/hydra/readTransformsFromFabricInRenderDelegate
and geometry streaming are enabled together but this can cause issues with dynamic objects not
streaming correctly due to problems in the transform update.
```

This is a real warning from Isaac's own logs, on literally every run, and its text
("dynamic objects not streaming correctly due to problems in the transform
update") is suspiciously on-topic. It has not been investigated. **This is
probably the single best lead to start from.**

## The scene and capture pipeline

No `InteractiveScene`, no multi-env templating — every prim sits at a fixed
absolute path, and the `Camera` sensor is a standalone object, not
scene-managed (this was a deliberate simplification for an earlier, unrelated
reason and is very unlikely to be the cause, but is stated for completeness):

```python
GROUND_PATH  = "/World/Ground"
LIGHT_PATH   = "/World/Light"
CUBE_PATH    = "/World/Cube"
CAMERA_PATH  = "/World/Camera"

sim = SimulationContext(SimulationCfg(
    dt=1.0/60.0, device="cuda:0",
    render=RenderCfg(antialiasing_mode="Off", enable_dl_denoiser=False),
))

sim_utils.GroundPlaneCfg().func(GROUND_PATH, ...)

light_cfg = sim_utils.DistantLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0))
light_cfg.func(LIGHT_PATH, light_cfg)   # no `angle` override -- USD default (near-zero, ~0.53deg)

cube_cfg = sim_utils.CuboidCfg(
    size=(0.06, 0.06, 0.06),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(r, g, b)),  # HSV-derived
)
cube_cfg.func(CUBE_PATH, cube_cfg, translation=(0.45, 0.0, 0.03))

camera_cfg = CameraCfg(
    prim_path=CAMERA_PATH, height=128, width=128, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 40.0)),
    offset=CameraCfg.OffsetCfg(pos=(1.0, 1.0, 0.9), convention="world"),
)
camera = Camera(camera_cfg)
# aimed via camera.set_world_poses_from_view(eye=(1.0,1.0,0.9), target=(0.45,0.0,0.1))
```

**Render preset** (applied via `carb.settings.get_settings().set(...)`, each
key read back and confirmed accepted):

```
/rtx/rendermode                          = "PathTracing"
/rtx/pathtracing/spp                     = 1
/rtx/pathtracing/totalSpp                = 64
/rtx/pathtracing/optixDenoiser/enabled   = 0
```

**Capture sequence** (`depth` tested at both 1 and 8 — no difference in the
result either way):

```python
def capture(_state):
    sim.forward()                                    # flush USD/Fabric, no time advance
    camera.update(dt=0.0, force_recompute=True)
    for _ in range(depth):
        sim.render()
    camera.update(dt=0.0, force_recompute=True)
    return camera.data.output["rgb"]                  # uint8 [1, 128, 128, 3]
```

Known, permanent, **unrelated** quirk of this build, mentioned so it isn't
mistaken for the mechanism under investigation: `camera.data.output[...]`
returns the same underlying tensor across calls (confirmed on every render mode
and preset tried, including the ones that render cleanly) — every capture is
`.clone()`d immediately, which the code above already does correctly.

## The determinism test

Standard A/B/B/A protocol — write state A, render; write state B, render twice
back-to-back; write state A again, render:

```python
a1_live = capture(state_a); a1 = a1_live.clone()
b1 = capture(state_b).clone()
b2 = capture(state_b).clone()
a2 = capture(state_a).clone()

order_independent_mad = mean_abs_diff(a1, a2)   # same state, rendered far apart
back_to_back_mad      = mean_abs_diff(b1, b2)   # same state, rendered immediately consecutively
```

`mean_abs_diff` widens to float32 before subtracting (uint8 wraps on
subtraction) and averages the absolute per-channel difference over the whole
128×128×3 frame.

## Results

| Write path | Mechanism | `order_independent_mad` | `back_to_back_mad` |
|---|---|---|---|
| **`cube.size`** (geometry, control) | `UsdGeom.Xformable` — find-or-add a `TypeScale` xform op, `.Set(Gf.Vec3f(...))` | `0.0` (bitwise) | `0.0` (bitwise) |
| `cube.hue` (material) | `UsdShade.Shader.GetInput("diffuseColor").Set(Gf.Vec3f(...))` | `0.0` | `1.877` |
| `light.intensity` | `UsdLux.LightAPI.GetIntensityAttr().Set(float)` | `0.0` | `1.746` |
| `light.warmth` | `UsdLux.LightAPI.GetEnableColorTemperatureAttr().Set(True)` + `GetColorTemperatureAttr().Set(float)` | `0.0` | `1.864` |
| `light.azimuth_elevation` (light rotation) | `UsdGeom.Xformable` — find-or-add a `TypeRotateXYZ` op, `.Set(Gf.Vec3f(...))` (Euler angles from `Gf.Rotation(...).Decompose(...)`) | `0.0` | `1.861` |
| `cam.jitter` (camera re-aim, per capture) | Isaac Lab `Camera.set_world_poses_from_view(eye, target)`, called before every capture | `0.0` | `1.853` |
| `exposure` (post-process) | carb settings only (`/rtx/post/tonemap/exposure`, `/rtx/post/tonemap/filmIso`, `/rtx/post/histogram/enabled`) | not directly tested this way | not gated, but `repeat_bitwise_equal: false` on a candidate-vs-repeat comparison |

**The numbers in the `back_to_back_mad` column reproduce exactly (to 3+ decimal
places) across separate, independent process launches with identical code and
arguments.** This was checked specifically because it looked like it could be
driver-level flakiness; it is not — it's a fully deterministic function of
something in this pipeline.

Note `cube.size` uses the *same general mechanism* as `light.azimuth_elevation`
(a raw `UsdGeom.Xformable` op, found-or-added, then `.Set()`) — one is a scale
op on the cube prim, the other a rotate op on the light prim — yet one is
perfectly clean and the other isn't. So "which USD API" doesn't cleanly predict
the outcome; "which prim" (or something about that prim's role in the scene)
seems to matter more.

## Where the difference actually is (not a uniform shift)

Using `torch.load` on saved `.pt` tensors, the pixel-level location of the
`cube.hue` back-to-back difference was localized:

```
mean_abs_diff:                2.498
std_abs_diff:                 9.56      # >> mean: most pixels identical, a few pixels jump hard
max_abs_diff:                 143.0
fraction_pixels_changed_gt_1: 0.147     # ~15% of the 128x128x3 frame
max_diff_at_index:            [batch=0, row=7, col=73, channel=0 (R)]
value at that pixel:          0 in one capture, 143 in the other
```

An ASCII brightness map of the frame, and a `#`/`.` map of exactly which pixels
changed, showed the changed-pixel region **visually overlapping the cube's own
silhouette** (the object being rendered), not the ground-plane/background
horizon and not a shadow boundary. So this is instability in **how the varied
object's own surface renders**, not a global exposure/accumulation shift and not
an edge-antialiasing artifact at some unrelated boundary.

One nuance worth flagging rather than treating as settled: in a *separate*,
later run of the same a1/b1/b2/a2 sequence for `light.intensity` and
`cam.jitter` (executed after many other checks had already run in the same
process), **both** `order_independent` and `back_to_back` showed the same
nonzero magnitude at the same pixel — i.e., the clean 0.0 for
`order_independent` isn't a fixed property of that comparison type, it may
depend on how much prior render history exists for whichever value is being
compared. This wasn't isolated further.

## Four hypotheses tried, each cleanly falsified

| # | Change | Result |
|---|---|---|
| 1 | `sim.render()` calls per capture: 1 → 8 (testing whether `totalSpp` accumulation just needs more external calls to converge) | **No effect at all** — identical numbers at depth 1 and depth 8 |
| 2 | Discard-first "warm-up" render: call the full capture sequence twice per value, keep only the second (testing whether a shader/material rebuild needs one extra full render cycle to settle) | **Made it worse**: regressed the previously-perfect `cube.size` (`0.0`→`~1.87`) without fixing any of the affected knobs (numbers moved slightly, e.g. `cube.hue` `1.877`→`1.838`, but stayed nonzero) |
| 3 | Apply the render preset (the four carb settings above) exactly once at scene-build time, vs. redundantly re-applying it before every single check (testing whether *repeated* preset application, not the individual writes, was the trigger) | **No effect at all** — identical numbers to the per-check-reapplication baseline |
| 4 | Widen the `DistantLight`'s angular size from its USD default (near-zero, knife-edge shadow) to `angle=3.0` degrees (testing whether an unstable hard-shadow edge, under `antialiasing_mode="Off"` + `spp=1`, was the cause) | **Made three checks measurably worse** (`cube.hue` `1.877`→`2.302`, `light.intensity` `1.746`→`2.61`, `light.warmth` `1.864`→`2.239`) **while two stayed bit-for-bit unchanged** (`light.azimuth_elevation` and `cam.jitter` both stayed at exactly `1.861` / `1.853` across every one of these four experiments) |

That last split — three checks moving, two completely inert, across a change
that should plausibly affect all of them if it were about shadow softness at
all — is itself a real clue: whatever the mechanism is, it doesn't uniformly
apply to "everything that isn't `cube.size`."

## A separate, likely-unrelated finding (context, not the main question)

The already-verified physics-write spike (joint/rigid-body pose only, no
material/light/camera writes) showed an *intermittent* failure of its own
sensitivity check at `--num-envs 1` and `--num-envs 32` — noise floor between
two captures of the *same untouched state* exploding to ~73–77 (vs. the
previously verified `0.0`). Critically, this **did not reproduce**: an
identical repeat run at `--num-envs 1` passed cleanly. That's flakiness, in
contrast to everything above, which is exactly reproducible. Plausibly related
to the driver risk noted above (`IsaacSim #537`), but not confirmed, and
probably a separate phenomenon from the main question here — included so it
isn't rediscovered as if new.

## Research questions

1. What is `/rtx/hydra/readTransformsFromFabricInRenderDelegate` actually doing, and is the logged warning ("geometry streaming... can cause issues with dynamic objects not streaming correctly due to problems in the transform update") directly implicated here? Is there a known fix, a setting to disable one side of it, or a documented interaction with non-Fabric-tracked prim edits (materials, lights, camera pose) specifically?
2. Physics-state writes go through Isaac Lab's tensorized, Fabric-backed write API (`write_joint_state_to_sim` / `write_root_state_to_sim`, flushed via `scene.write_data_to_sim()`) and render cleanly. Every write path in this task instead edits USD directly (`pxr.UsdGeom` / `UsdShade` / `UsdLux`) or goes through Isaac Lab's `Camera.set_world_poses_from_view`. Is there a known synchronization gap between direct USD edits (or this specific Camera API) and Hydra's render-delegate seeing them fully, that `sim.forward()` + `camera.update(force_recompute=True)` does not close?
3. Why would a `TypeScale` xform op on one prim (the cube) be clean while a mechanically similar `TypeRotateXYZ` op on a different prim (the light) is not? Is this about which prim (e.g., does light rotation trigger a shadow-map/BVH rebuild that a mesh transform doesn't), or about something else entirely?
4. Is there a known Isaac Sim / Kit / RTX GitHub issue (isaac-sim/IsaacSim, isaac-sim/IsaacLab, or upstream omniverse-kit) matching "small, exactly reproducible per-pixel difference between two renders of an identical scene state, under PathTracing with the denoiser disabled, when a material/light/camera attribute was written between an earlier render and this one"?
5. Is there a stronger/different synchronization call that should be tried — e.g. an explicit `omni.usd` stage-update call, `omni.kit.app.get_app().update()`, waiting a fixed number of frames after a *specific kind* of USD edit before the first "real" render, or forcing a BVH/acceleration-structure rebuild explicitly rather than implicitly?
6. Is this simply a known, permanent limitation of this exact preset/build (i.e., "don't expect bitwise determinism for non-Fabric USD writes under PathTracing with `spp=1`, full stop") — and if so, what is the standard workaround serious Isaac Sim pipelines use (a different preset, more `spp` and a tolerance instead of bitwise, per-sample process isolation, or something else)?

## What a useful answer looks like

Any of the following would move this forward:
- A specific diagnosis (a named Isaac/Kit/RTX/USD mechanism) with a suggested code change or setting to test next, framed as a hypothesis we can falsify the same way the four above were — not a confident claim without a next experiment attached.
- A pointer to a matching upstream issue/discussion, even if unresolved upstream — confirms this isn't something we're uniquely doing wrong.
- A confirmed "this is a known, permanent limitation of PathTracing + `spp=1` + non-Fabric writes on this Isaac Sim version" verdict, ideally with the standard workaround other Isaac Sim projects use for this class of problem.

We can run new experiments on the actual GPU pod (RunPod, headless, terminal-only
access) but cannot do so as part of this research turn — this document exists so
that whoever picks it up can reason from evidence already gathered rather than
needing pod access to make progress.
