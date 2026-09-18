"""Render control (README §4.4): raw `carb` settings, contained here because
Isaac Lab's own `RenderCfg` exposes no path-tracing SPP or accumulation
control.
"""

from __future__ import annotations

from typing import Any

#: README §7.3's `standard` -- measured bitwise-deterministic for `base`
#: (§7.2 Spike 1) and for `cube.size`/`cube.hue`/`light.intensity`/
#: `light.warmth`/`cam.jitter`/`table.albedo` (§7.5 Spike 5). The fifth key is
#: what fixed everything except `light.azimuth_elevation`/`table.roughness`,
#: which stay blocked regardless of this preset (README §11).
STANDARD: dict[str, Any] = {
    "/rtx/rendermode": "PathTracing",
    "/rtx/pathtracing/spp": 1,
    "/rtx/pathtracing/totalSpp": 64,
    "/rtx/pathtracing/optixDenoiser/enabled": 0,
    "/rtx/resetPtAccumOnAnimTimeChange": True,
}

#: README §7.5's `exposure_lever` check: the primary of three accepted
#: candidates, chosen because it is the direct tonemap exposure control
#: rather than a film-speed proxy (`filmIso`) or an unrelated toggle
#: (`histogram/enabled`) -- one continuous handle needs one lever.
EXPOSURE_KEY = "/rtx/post/tonemap/exposure"


def apply_standard_preset() -> dict[str, Any]:
    """Set every `standard` key and read each back.

    carb happily creates unknown keys, so read-back equality -- not the
    absence of an exception -- is the only evidence a given key exists on
    this build (README §7.2/§7.5).
    """
    import carb

    settings = carb.settings.get_settings()
    applied: dict[str, Any] = {}
    for key, value in STANDARD.items():
        settings.set(key, value)
        applied[key] = {"wanted": value, "read_back": settings.get(key)}
    return applied


def write_exposure(value: float) -> dict[str, Any]:
    """`exposure` (README §5.2.3) -- the one confirmed-accepted lever."""
    import carb

    settings = carb.settings.get_settings()
    settings.set(EXPOSURE_KEY, float(value))
    return {EXPOSURE_KEY: {"wanted": float(value), "read_back": settings.get(EXPOSURE_KEY)}}


def read_exposure() -> float:
    import carb

    return float(carb.settings.get_settings().get(EXPOSURE_KEY))
