# Research task: a light's rotation write has *zero* pixel effect under two specific Isaac Sim RTX configs

## TL;DR

Rotating a `UsdLux.DistantLight` between two very different orientations
(azimuth/elevation `0°/35°` vs `120°/55°`) produces a **real, substantial**
image difference (~1.2–1.9 mean-abs-diff per pixel channel, out of 255) under
every RTX PathTracing config we've tried **except two**, where it produces
**exactly `0.0` — bitwise-identical frames, not just "hard to see"**. Every
other light/material/camera attribute on the same prim (intensity, colour
temperature, camera jitter) renders correctly and distinctly under those same
two configs. The effect is isolated to this one attribute, on this one prim,
under two specific renderer configurations. We need a diagnosis of the
mechanism.

## Environment

- Isaac Sim `6.0.1`, Isaac Lab `3.0.0-beta2-post1` (`isaaclab.__version__` reports `6.1.16`)
- Confirmed backend, from the boot log: `Registered backend 'physx' for factory Renderer` / `Using renderer: IsaacRtxRenderer` — i.e. **PhysX + IsaacRtxRenderer**, not the Newton + OVRTX backend some upstream issues are filed against.
- RTX 4090, driver 570.211.01, headless RunPod container.

## The write

```python
from pxr import Gf, UsdGeom

def write_light_direction(light_prim, azimuth_rad, elevation_rad):
    direction = azel_to_direction(azimuth_rad, elevation_rad)  # unit vector
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), Gf.Vec3d(*direction))
    euler_deg = rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
    op = light_prim's existing or newly-added UsdGeom.XformOp.TypeRotateXYZ
    op.Set(Gf.Vec3f(euler_deg[0], euler_deg[1], euler_deg[2]))
```

The light is a `DistantLightCfg`-spawned prim, kinematic-free (no physics, no
Fabric publication — only a separate kinematic rigid-body cube in the same
scene is Fabric-tracked). Capture sequence: `sim.forward()` → `camera.update()`
→ N × `sim.render()` → `camera.update()` → read `camera.data.output["rgb"]`.
Zero `sim.step()` calls anywhere (by design — the pipeline never advances
physics time).

## Results (mean-abs-diff between the two orientations, base carb config is `/rtx/rendermode=PathTracing, spp=1, totalSpp=64, optixDenoiser=Off`)

| Extra config on top of the base PathTracing settings | mad between the two light orientations |
|---|---|
| none (baseline) | **1.861** |
| `/rtx/pathtracing/cached/enabled=False` + 4 related cache/AA keys | **1.261** (reduced, still real) |
| `/rtx/rendermode=RealTimePathTracing` + `/rtx/rtpt/cached/enabled=False` + `/rtx/rtpt/lightcache/cached/enabled=False` | **0.058** (small, still real) |
| **`/rtx/resetPtAccumOnAnimTimeChange=True`** | **exactly `0.0`** |
| **`/rtx/rendermode=Minimal`** | **exactly `0.0`** |
| `resetPtAccumOnAnimTimeChange=True` **+** `/rtx/hydra/readTransformsFromFabricInRenderDelegate=False` | still exactly `0.0` (ruled out: this is not a Fabric-vs-USD transform-source issue) |

In both zero-effect configs, **every other attribute on the same light prim
still works**: intensity and colour-temperature changes render correctly and
pass a strict bitwise-determinism gate in the same run. Only the rotation
(`UsdGeom.XformOp.TypeRotateXYZ`) is affected.

**Also ruled out:** destroying and recreating the light prim from scratch
immediately before each rotation write (instead of mutating the existing
prim's xform op in place) — testing whether this was a stale per-light
acceleration structure or shadow cache keyed to prim identity. Result:
**still exactly `0.0`**, under `/rtx/resetPtAccumOnAnimTimeChange=True`. A
brand-new prim, rotated before its first-ever render, is indistinguishable
from the same prim at a different rotation. This points away from anything
tied to the prim's *history* and toward the render config simply not
sampling the light's transform for shading at all — a baked/default
direction, a position-only light model, or a per-scene-topology cache that a
same-type prim swap doesn't invalidate.

## Research questions

1. Is there a known Isaac Sim / Kit / RTX mechanism by which a light's
   **transform** specifically (as opposed to its intensity/colour shader
   inputs) is cached, baked at first use, or only re-evaluated on some event
   *other than* a per-frame Hydra sync — under either `/rtx/rendermode=Minimal`
   or with `/rtx/resetPtAccumOnAnimTimeChange=True` set?
2. Given a *brand-new* light prim (destroyed and recreated immediately before
   its first render) shows the identical zero effect, this is not a
   history-dependent or identity-keyed cache. Is there a known code path where
   `Minimal` mode or `resetPtAccumOnAnimTimeChange` shading ignores a
   `UsdLux.DistantLight`'s authored transform entirely — e.g. sampling only a
   default/world-up direction, or a per-scene-topology light setup that a
   same-type prim swap doesn't invalidate?
3. Does Hydra/RTX treat a rotation (`TypeRotateXYZ`) op differently from a
   scale op or a shader-parameter change, specifically for `UsdLux` light
   prims, under these two render configs?
4. Any known matching upstream issue (isaac-sim/IsaacSim, isaac-sim/IsaacLab,
   or omniverse-kit)?

## What a useful answer looks like

A named mechanism with a specific carb key, USD API call, or renderer setting
to test next — framed as a falsifiable next experiment, the same discipline as
the four configs in the results table above. A confirmed "this is a known,
permanent limitation of light-transform updates under this mode" verdict,
with or without a workaround, is also useful and moves this forward.
