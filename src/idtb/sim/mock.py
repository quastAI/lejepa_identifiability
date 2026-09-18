"""``MockSceneBackend`` -- an analytic, numpy-only renderer (README §4.3).

Not a convenience. It does two jobs: a local dev/test loop for everything
above the backend seam, and a **known-good reference** for the §10.1 gates --
a determinism gate never observed to pass on something that should pass is
not evidence.

Deterministic and injective *by construction*: every pixel is a closed-form,
anti-aliased function of the physical state, no RNG anywhere. Hard edges
would make two nearby latents render identically and fail the injectivity
proxy on the one backend where it must pass, so every shape is blended over
a soft edge rather than a hard cutoff.
"""

from __future__ import annotations

import colorsys
from collections.abc import Mapping

import numpy as np
import torch

from idtb.latents import LatentSpec
from idtb.sim.backend import HandleInfo, UnsupportedRoleError, WritePath

Tensor = torch.Tensor

#: Every role this backend knows how to render, and which write path it
#: takes (README §6.3) -- mirrored here even though the mock's own "writes"
#: are all trivially exact, so the tier-1 suite can exercise the same
#: three-path dispatch against both backends. A role outside this table is
#: refused at bind time, never silently ignored.
_ROLE_WRITE_PATHS: dict[str, WritePath] = {
    "arm.j0": "joint",
    "arm.j1": "joint",
    "arm.j2": "joint",
    "arm.j3": "joint",
    "gripper.aperture": "joint",
    "cube.x": "root_state",
    "cube.y": "root_state",
    "cube.size": "attribute",
    "cube.hue": "attribute",
    "light.intensity": "attribute",
    "light.warmth": "attribute",
    "light.azimuth": "attribute",
    "light.elevation": "attribute",
    "cam.jitter.x": "attribute",
    "cam.jitter.y": "attribute",
    "table.roughness": "attribute",
    "table.albedo": "attribute",
    "exposure": "attribute",
}

#: Ids stable across every run of this backend (README Phase 4) -- Isaac's own
#: annotator ids are not, which would silently turn the visibility column into
#: noise.
SEG_BACKGROUND = 0
SEG_ARM = 1
SEG_CUBE = 2


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _alpha_from_sdf(distance: np.ndarray, edge: float) -> np.ndarray:
    """1 well inside the shape, 0 well outside, smooth across ``edge`` pixels."""
    return _smoothstep(0.5 - distance / edge)


def _bilinear_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Sub-pixel image translation -- the camera-jitter style knobs (README §5.2.3)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    src_x = np.clip(xx - dx, 0.0, w - 1.0)
    src_y = np.clip(yy - dy, 0.0, h - 1.0)
    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (src_x - x0)[..., None]
    fy = (src_y - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


class MockSceneBackend:
    """Numpy-only analytic scene: one cube, one arm marker, global style tone.

    ``resolution`` is ``(H, W)``; frames come back as if from Isaac's own
    camera sensor -- ``uint8``, ``[B, H, W, 3]`` -- except never aliased
    (README §7.2's aliasing finding is a real Isaac fact this backend does
    not reproduce; the mock is the known-good reference, not a faithful bug
    replica).
    """

    def __init__(self, resolution: tuple[int, int] = (64, 64)) -> None:
        self.resolution = resolution
        self._spec: LatentSpec | None = None
        self._phi: Tensor | None = None
        self._last_diagnostics: dict[str, Tensor] | None = None

    def bind(self, spec: LatentSpec) -> None:
        unsupported = [role for role in spec.roles if role not in _ROLE_WRITE_PATHS]
        if unsupported:
            raise UnsupportedRoleError(
                f"MockSceneBackend cannot write role(s) {unsupported!r}; "
                f"known roles: {sorted(_ROLE_WRITE_PATHS)}"
            )
        self._spec = spec
        self._phi = None
        self._last_diagnostics = None

    def handles(self) -> Mapping[str, HandleInfo]:
        spec = self._require_bound()
        return {
            role: HandleInfo(role, _ROLE_WRITE_PATHS[role])
            for role in spec.roles
        }

    def write_state(self, phi: Tensor) -> None:
        spec = self._require_bound()
        if phi.ndim != 2 or phi.shape[1] != spec.n:
            raise ValueError(f"expected phi of shape [B, {spec.n}], got {tuple(phi.shape)}")
        self._phi = phi.detach().clone()

    def read_state(self) -> Tensor:
        """Exact by construction -- the mock has no solver tolerance to lose to."""
        if self._phi is None:
            raise RuntimeError("read_state() before any write_state()")
        return self._phi.clone()

    def render(self, sample_idx: int) -> dict[str, dict[str, Tensor]]:
        del sample_idx  # a pure function of state; sample_idx carries no meaning here
        spec = self._require_bound()
        if self._phi is None:
            raise RuntimeError("render() before any write_state()")
        batch = self._phi.shape[0]
        h, w = self.resolution
        rgb = np.empty((batch, h, w, 3), dtype=np.uint8)
        seg = np.empty((batch, h, w), dtype=np.int64)
        visibility = np.empty(batch, dtype=np.float32)
        collision = np.empty(batch, dtype=np.float32)
        for b in range(batch):
            u = {role: self._normalized(role, spec, b) for role in _ROLE_WRITE_PATHS}
            frame, seg_b, diag = self._render_one(u)
            rgb[b], seg[b] = frame, seg_b
            visibility[b] = diag["visibility"]
            collision[b] = diag["collision"]
        self._last_diagnostics = {
            "visibility": torch.from_numpy(visibility),
            "collision": torch.from_numpy(collision),
        }
        return {
            "cam0": {
                "rgb": torch.from_numpy(rgb),
                "seg": torch.from_numpy(seg),
            }
        }

    def diagnostics(self) -> dict[str, Tensor]:
        if self._last_diagnostics is None:
            raise RuntimeError("diagnostics() before any render()")
        return dict(self._last_diagnostics)

    def close(self) -> None:
        pass

    # -- internals ----------------------------------------------------------

    def _require_bound(self) -> LatentSpec:
        if self._spec is None:
            raise RuntimeError("bind() must be called before use")
        return self._spec

    def _normalized(self, role: str, spec: LatentSpec, batch_row: int) -> float:
        """``phi`` normalised to ``(-1, 1)`` via the handle's own center/radius.

        A role absent from the *bound* (possibly narrowed) spec is held at its
        handle centre -- ``u = 0`` -- matching README §5.2: "inactive handles
        are held at their handle centre, never removed."
        """
        if role not in spec.roles or self._phi is None:
            return 0.0
        idx = spec.index(role)
        handle = spec.handles[idx]
        value = float(self._phi[batch_row, idx].item())
        return (value - handle.center) / handle.radius

    def _render_one(
        self, u: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        h, w = self.resolution
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ny = yy / (h - 1) * 2.0 - 1.0
        nx = xx / (w - 1) * 2.0 - 1.0
        edge = 3.0 / min(h, w)  # anti-aliasing band width, in normalised units

        # -- style: global tone (README §5.2.3, §7.4) ------------------------
        intensity_gain = 0.6 + 0.35 * u["light.intensity"]  # kept off both clip ends
        exposure_gain = 1.0 + 0.25 * u["exposure"]
        albedo = 0.5 + 0.3 * u["table.albedo"]
        warmth = u["light.warmth"]
        roughness = u["table.roughness"]

        canvas = np.full((h, w, 3), albedo, dtype=np.float32) * intensity_gain * exposure_gain
        canvas[..., 0] += 0.08 * warmth
        canvas[..., 2] -= 0.08 * warmth

        # deterministic low-frequency pattern, never random -- injectivity must
        # never depend on an RNG seed
        grid = 0.05 * roughness * (np.sin(nx * 6.0) * np.cos(ny * 6.0))
        canvas += grid[..., None]

        # directional "shadow" gradient from light azimuth/elevation
        azimuth = u["light.azimuth"] * np.pi
        elevation = 0.5 * (u["light.elevation"] + 1.0)
        theta = np.arctan2(ny, nx)
        canvas += (0.12 * elevation * np.cos(theta - azimuth))[..., None]

        # per-capture camera jitter: shift the whole frame, sub-pixel
        dx = u["cam.jitter.x"] * 0.06 * w
        dy = u["cam.jitter.y"] * 0.06 * h
        canvas = _bilinear_shift(canvas, dx, dy)

        seg = np.full((h, w), SEG_BACKGROUND, dtype=np.int64)

        # -- base: cube position, drawn first so the arm can occlude it -----
        cube_x = 0.6 * u["cube.x"]
        cube_y = 0.6 * u["cube.y"]
        cube_half = 0.06 + 0.05 * (0.5 * (u["cube.size"] + 1.0))
        hue = 0.5 * (u["cube.hue"] + 1.0) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        cube_sdf = np.maximum(np.abs(nx - cube_x), np.abs(ny - cube_y)) - cube_half
        cube_alpha = _alpha_from_sdf(cube_sdf, edge)
        canvas = canvas * (1 - cube_alpha[..., None]) + np.array([r, g, b]) * cube_alpha[..., None]
        cube_mask = cube_alpha > 0.5
        seg[cube_mask] = SEG_CUBE
        cube_pixels = int(cube_mask.sum())

        # -- base: arm marker, a deterministic nonlinear projection of the
        # four joints + gripper aperture, drawn on top -----------------------
        arm_x = 0.4 * np.tanh(u["arm.j0"] + 0.5 * u["arm.j2"])
        arm_y = 0.4 * np.tanh(u["arm.j1"] + 0.5 * u["arm.j3"])
        arm_radius = 0.05 + 0.04 * (0.5 * (u["gripper.aperture"] + 1.0))
        arm_sdf = np.hypot(nx - arm_x, ny - arm_y) - arm_radius
        arm_alpha = _alpha_from_sdf(arm_sdf, edge)
        arm_color = np.array([0.2, 0.45, 0.9])
        canvas = canvas * (1 - arm_alpha[..., None]) + arm_color * arm_alpha[..., None]
        arm_mask = arm_alpha > 0.5
        seg[arm_mask] = SEG_ARM

        occluded = int((cube_mask & arm_mask).sum())
        visibility = 1.0 - occluded / cube_pixels if cube_pixels else 1.0
        center_distance = float(np.hypot(arm_x - cube_x, arm_y - cube_y))
        collision = 1.0 if center_distance < (arm_radius + cube_half) * 0.6 else 0.0

        rgb = np.clip(canvas, 0.0, 1.0)
        rgb_u8 = np.round(rgb * 255.0).astype(np.uint8)
        return rgb_u8, seg, {"visibility": visibility, "collision": collision}
