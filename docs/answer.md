The single most important finding

Your depth result (hypothesis 1) isn't a mystery, and it reframes everything else. The ovrtx camera docs state that in ovrtx, a render step accumulates towards the omni:rtx:pt:samplesPerPixel limit before returning output, so application code usually captures Path Tracing output with a single step rather than the multi-frame warm-up loop used for Real-Time Path-Tracing convergence, and that PathTracing mode does not need the same warm-up loop for path-tracing convergence that Real-Time Path-Tracing uses. 
github
github

So extra sim.render() calls are a no-op by design, and both frames you compare are nominally fully accumulated. That kills the obvious hypothesis ("not enough samples yet") and forces the conclusion that the variance lives in persistent renderer state that survives between render steps, not in sample count. Everything below follows from that.

A second, cheaper problem sits underneath it. Isaac Sim 6.x sets render mode and path-tracing parameters per RenderProduct as USD attributes: the render mode is set per RenderProduct using the omni:rtx:rendermode attribute, with omni:rtx:pt:samplesPerPixel, omni:rtx:pt:samplesPerIteration, omni:rtx:pt:adaptiveSampling:enabled, omni:rtx:pt:radianceCache:enabled and so on as per-product attributes. Meanwhile the render-settings schema extension describes a sync layer for changes to deprecated carb-global settings with an ignoreSync switch. Your four carb keys are the global/deprecated path. Reading a carb key back confirms carb stored it, not that your camera's render product used it. That also quietly explains hypothesis 3: re-applying a preset that never mattered changes nothing. 
github
nvidia

Relatedly, antialiasing_mode="Off" is applied by Isaac Lab through rep.settings.set_render_rtx_realtime(...), a Real-Time-mode call. Under PathTracing, AA is the jittered pixel filter: /rtx/pathtracing/aa/op defaults to Triangle (1) and /rtx/pathtracing/aa/filterRadius to 1.0. You almost certainly have sub-pixel jitter on while believing it's off. 
nvidia

Primary diagnosis: the path tracer's caches, not the sample count

Three cross-frame caches are on by default and you never touched any of them: /rtx/pathtracing/cached/enabled (Bool True) enables caching path tracing results for improved performance at the cost of some accuracy, and /rtx/pathtracing/lightcache/cached/enabled (Bool True) enables many-light sampling. Adaptive sampling is ambiguous across the two stacks: the carb doc gives /rtx/pathtracing/adaptiveSampling/enabled as Bool False, but the per-product ovrtx default for omni:rtx:pt:adaptiveSampling:enabled is true. You need to check which one is live. 
nvidia
nvidia

And the accumulation restart is change-detection driven: when the totalSpp sample count is reached the rendering stops until a scene or setting change is detected, restarting the rendering process. That is precisely the shape of asymmetry you measured. Geometry/transform edits invalidate caches and acceleration structures; material, light-attribute and camera-pose edits need not. 
nvidia

This model predicts every result in your table:

Deterministic across process launches: same history produces the same cache contents produces the same numbers.
Localized to the cube's silhouette, not the horizon, not a shadow edge: that's where cached indirect radiance is being sampled.
Hypothesis 2 is the smoking gun. A "needs more convergence" theory predicts a warm-up render helps. A cache-state theory predicts it hurts, because you hand the cube a warm cache too. Your cube.size control regressed 0.0 → 1.87. That's a clean prediction hit and it's hard to explain otherwise.
Your flagged nuance (order_independent going nonzero in a later run, after lots of prior checks in the same process) is exactly what history-dependent cache state looks like.

Upstream corroboration, all of it recent:

IsaacSim #822: a RectLight projector pattern only resolves when the camera is stationary under RTPT's radiance cache, with contrast climbing over about a second of simulated time — "That rising trend, rather than a fixed blur, is what points at an accumulation/convergence effect." 
GitHub
RTX 110.1's fixed list includes "Sensor RTX tests produce a different result after octree-based radiance cache" and "In RTX Interactive mode, sample accumulation stops prematurely the first time Adaptive Sampling is enabled, requiring a disable/re-enable cycle to accumulate the full 512 samples". 110.1 postdates Isaac Sim 6.0.1's bundled RTX, so both are plausibly live on your build. The second one is a history-dependent accumulation stop, which is your phenomenon almost verbatim. 
Nvidia
Nvidia
Your warning lead: right clue, wrong target

readTransformsFromFabricInRenderDelegate makes OmniHydra bypass UsdImaging for transforms: the OmniHydra scene delegate will query Fabric for changes to these special attributes when discovering transform data in the Hydra render index, enabling faster updates of prim transforms than USD and UsdImaging conventionally allow. Critically, NVIDIA also states that only OmniHydra XForm attributes in Fabric will drive the rendered prims — any USD-style xformOps in Fabric are ignored by OmniHydra. 
nvidia
nvidia

That answers question 3 directly, and it isn't about "which USD API" or "shadow-map rebuild". The cube is a kinematic rigid body, so Isaac writes its transform into Fabric; the light is not in Fabric. Your TypeScale op and your TypeRotateXYZ op are the same API taking two completely different routes to the renderer. Which route, not which prim, is the variable.

It also raises a serious possibility about your control: cube.size may be clean because it's a no-op. If Fabric's transform overrides your USD scale op, both "states" render identically and 0.0 is meaningless. Your doc never reports the A-vs-B magnitude for cube.size, only the same-state comparisons. Check that first; it's five minutes and it could invalidate your only clean row.

Question 2, and the version-matched issue you should read

sim.forward() + camera.update(force_recompute=True) does not close the gap. IsaacLab #6609, reproduced on your exact version, reports that RenderContext publishes renderer scene state at most once for a physics-step count, and an environment reset can change asset state and call forward() without advancing that count, fixed by calling RenderContext.reset_scene_state_cadence(), with reset_transform_cadence() public in patch1. Your capture loop never advances the physics step count, so you sit squarely in that regime. That issue also confirms the accumulation is explicit, resettable state: native Renderer.reset() restarts RTPT accumulation and makes immediate beauty frames intentionally cold. 
[Bug Report] Environment reset can skip renderer scene-state publication · Issue #6609 · isaac-sim/IsaacLab +2

Four experiments, in the order I'd run them

A. Audit what's actually configured (10 min, may invalidate your whole preset). Dump every omni:rtx:* attribute on the camera's RenderProduct prim and compare against your four carb keys. Also confirm cube.size A-vs-B is nonzero. Falsifies: "my preset is live."

B. Turn off every cache and every stochastic filter. /rtx/pathtracing/cached/enabled=False, /rtx/pathtracing/lightcache/cached/enabled=False, /rtx/pathtracing/adaptiveSampling/enabled=False, /rtx/pathtracing/fireflyFilter/enabled=False, /rtx/pathtracing/aa/op=0, /rtx/pathtracing/aa/filterRadius=0.0, set via the per-product attributes if A shows carb isn't authoritative. Prediction: the five dirty knobs go to 0.0 or drop by an order of magnitude. If they don't move at all, the cache hypothesis is dead.

C. Force a cold accumulation every frame. /rtx/resetPtAccumOnAnimTimeChange (Bool False) restarts the Path-Tracer accumulation every time the MDL animation time changes. Set it True. Combine with RenderContext.reset_transform_cadence() before each capture, and try omni.kit.app.get_app().update() in place of sim.render(). 
nvidia

D. Abandon PathTracing. Minimal completely disables all indirect light transport and uses only the first distant light source found in the scene, with hard shadows only, and is positioned for training-in-the-loop and high-throughput workflows where low latency is more important than full light transport. No Monte Carlo state, no caches, no accumulation. Your material, light and camera knobs all still change the image. For a 128×128 cube on a ground plane you lose very little. 
Nvidia
Isaac Sim

Verdict on question 6

I found no document or issue stating that PT + spp=1 + non-Fabric writes is permanently non-deterministic. What the documentation does establish is that bitwise determinism is not a designed property of this mode: it's progressive, its restart is change-detection driven, and it ships with three history-carrying caches enabled. Nothing guarantees what you need, and there's no known switch labelled "be deterministic."

The ecosystem's answer is to not fight it. Replicator's documented guidance for your two exact symptom classes is subframes, not more renders: if randomized materials are not loaded on time for synthetic data generation, the rt_subframes must be set to be at least 2, and if ghosting artifacts are observed, especially for scenes with moving objects or significant changes in lighting conditions, increase the rt_subframes value. If bitwise is non-negotiable, the real options are Minimal mode or per-sample process isolation. 
nvidia
nvidia

One note on the underlying research requirement. You're right that a render-order-dependent residual is a genuine shortcut risk, but bitwise equality is stronger than what you need. The operative property is that the residual carries no information about the latent. A cheap decisive test: render each state twice in shuffled order and check whether a linear probe on b1 − b2 predicts the varied factor above chance. If experiment B doesn't rescue PathTracing, that's a defensible fallback criterion rather than giving up the render path.