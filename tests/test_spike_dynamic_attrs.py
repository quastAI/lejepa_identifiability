"""The Spike 5 pure layer, exercised on the dev machine where Isaac cannot run.

``spikes/spike_dynamic_attrs.py`` is written blind against README §5.2/§7.5 and
only ever runs on the pod, so the parts that can be tested here are tested here:
the generic "resolve, don't guess" executor and the small pieces of pure math
(hue conversion, azimuth/elevation, the measured-aperture formula) that the
Isaac-layer writers build on. Everything else -- the actual USD/carb calls --
needs a live stage and is exercised only on the pod (docs/PLAN.md Phase 3b).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SPIKE = Path(__file__).resolve().parent.parent / "spikes" / "spike_dynamic_attrs.py"


def _load():
    """Import the spike by path -- ``spikes/`` is a script directory, not a package."""
    spec = importlib.util.spec_from_file_location("spike_dynamic_attrs", SPIKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


spike = _load()


def test_spike_imports_with_the_isaac_roots_blocked():
    """Same discipline as spike_api.py and README §4.2: every Isaac/USD import
    lives inside a function, after ``SimulationApp`` exists. A module-level one
    here would fail only on the pod, one container restart per violation.
    """
    script = f"""
import importlib.util, sys

BLOCKED = {{"isaacsim", "isaaclab", "omni", "carb", "pxr"}}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in BLOCKED:
            raise ImportError(f"module-level Isaac import of {{name!r}}")
        return None


sys.meta_path.insert(0, Blocker())

try:  # negative control: a blocker nobody has seen fire is not a blocker
    import carb  # noqa: F401
except ImportError:
    pass
else:
    raise SystemExit("blocker inactive")

spec = importlib.util.spec_from_file_location("spike_dynamic_attrs", {str(SPIKE)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# --- try_candidates: the generic "resolve, don't guess" executor --------------


def test_try_candidates_returns_the_first_that_does_not_raise():
    name, value = spike.try_candidates(
        [
            ("broken", lambda: (_ for _ in ()).throw(RuntimeError("nope"))),
            ("works", lambda: 42),
            ("never_reached", lambda: 1 / 0),
        ]
    )
    assert name == "works"
    assert value == 42


def test_try_candidates_raises_with_every_error_when_all_fail():
    with pytest.raises(spike.CheckFailed) as excinfo:
        spike.try_candidates(
            [
                ("broken", lambda: (_ for _ in ()).throw(RuntimeError("a"))),
                ("also_broken", lambda: (_ for _ in ()).throw(ValueError("b"))),
            ]
        )
    message = str(excinfo.value)
    assert "broken: RuntimeError" in message
    assert "also_broken: ValueError" in message


# --- hue_to_rgb ----------------------------------------------------------------


def test_hue_to_rgb_matches_known_primary_colours():
    r, g, b = spike.hue_to_rgb(0.0, saturation=1.0, value=1.0)
    assert (r, g, b) == pytest.approx((1.0, 0.0, 0.0))

    r, g, b = spike.hue_to_rgb(1.0 / 3.0, saturation=1.0, value=1.0)
    assert (r, g, b) == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_hue_to_rgb_wraps_the_circle():
    assert spike.hue_to_rgb(0.0) == pytest.approx(spike.hue_to_rgb(1.0))
    assert spike.hue_to_rgb(0.2) == pytest.approx(spike.hue_to_rgb(1.2))


# --- azel_to_direction -----------------------------------------------------------


def test_azel_to_direction_is_a_unit_vector():
    import math

    for az, el in [(0.0, 0.0), (1.3, 0.4), (-2.0, -0.6)]:
        x, y, z = spike.azel_to_direction(az, el)
        assert math.sqrt(x * x + y * y + z * z) == pytest.approx(1.0)


def test_azel_to_direction_zero_elevation_points_in_the_xy_plane():
    x, y, z = spike.azel_to_direction(0.0, 0.0)
    assert (x, y, z) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_azel_to_direction_straight_up_at_the_pole():
    import math

    x, y, z = spike.azel_to_direction(1.0, math.pi / 2.0)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)


# --- cube_size_radius_from_aperture ---------------------------------------------


def test_cube_size_radius_from_aperture_matches_the_measured_finger_travel():
    """README §5.2.2: the upper bound is a *measured* task constraint, not a
    guess -- derived from the finger aperture Spike 1 measured, ``[0.0, 0.04]`` m.
    """
    radius = spike.cube_size_radius_from_aperture(0.04, margin=1.0)
    assert radius == pytest.approx(0.08)
    assert spike.cube_size_radius_from_aperture(0.04, margin=0.5) == pytest.approx(0.04)


def test_cube_size_radius_from_aperture_rejects_a_margin_outside_zero_one():
    with pytest.raises(ValueError, match="margin"):
        spike.cube_size_radius_from_aperture(0.04, margin=0.0)
    with pytest.raises(ValueError, match="margin"):
        spike.cube_size_radius_from_aperture(0.04, margin=1.5)


# --- carb_value_matches: tolerant of carb's float32 round-trip -----------------


def test_carb_value_matches_a_float32_rounded_readback():
    """The exact bug this guards against: carb stores 1.6 as float32 and hands
    back 1.600000023841858, which a strict == would wrongly call rejected.
    """
    assert spike.carb_value_matches(1.6, 1.600000023841858)
    assert spike.carb_value_matches(200.0, 200.00000762939453)


def test_carb_value_matches_rejects_a_genuinely_different_float():
    assert not spike.carb_value_matches(1.6, 1.8)


def test_carb_value_matches_compares_bools_and_ints_exactly():
    assert spike.carb_value_matches(False, False)
    assert not spike.carb_value_matches(False, True)
    assert spike.carb_value_matches(64, 64)
    assert not spike.carb_value_matches(64, 63)


# --- diff_summary: localizing a mad number to an actual pixel region -----------


def test_diff_summary_finds_the_max_diff_pixel():
    a = torch.zeros(1, 4, 4, 3, dtype=torch.uint8)
    b = a.clone()
    b[0, 2, 3, 1] = 200  # one pixel, one channel, way off
    summary = spike.diff_summary(a, b)
    assert summary["max_abs_diff"] == pytest.approx(200.0)
    assert summary["value_a_at_max_diff"] == pytest.approx(0.0)
    assert summary["value_b_at_max_diff"] == pytest.approx(200.0)
    # index is (batch, row, col, channel) -- the exact pixel that moved.
    assert summary["max_diff_at_index"] == [0, 2, 3, 1]


def test_diff_summary_identical_frames_are_zero_everywhere():
    a = torch.full((1, 4, 4, 3), 42, dtype=torch.uint8)
    summary = spike.diff_summary(a, a.clone())
    assert summary["mean_abs_diff"] == pytest.approx(0.0)
    assert summary["max_abs_diff"] == pytest.approx(0.0)
    assert summary["fraction_pixels_changed_gt_1"] == pytest.approx(0.0)


def test_diff_summary_distinguishes_uniform_shift_from_localized_spike():
    """The whole point: two frames with the same mean_abs_diff can look
    completely different -- a uniform +2 everywhere vs. one blown-out pixel --
    and fraction_pixels_changed_gt_1 is what tells them apart.
    """
    base = torch.zeros(1, 4, 4, 3, dtype=torch.uint8)
    uniform = torch.full((1, 4, 4, 3), 2, dtype=torch.uint8)
    localized = base.clone()
    localized[0, 0, 0, 0] = 96  # 96 / 48 pixels ~= 2 mean, concentrated in one spot

    uniform_summary = spike.diff_summary(base, uniform)
    localized_summary = spike.diff_summary(base, localized)
    assert uniform_summary["mean_abs_diff"] == pytest.approx(localized_summary["mean_abs_diff"])
    assert uniform_summary["fraction_pixels_changed_gt_1"] == pytest.approx(1.0)
    assert localized_summary["fraction_pixels_changed_gt_1"] == pytest.approx(1.0 / 48.0)


# --- reused pure layer, sanity that the load actually worked --------------------


def test_reuses_spike_api_pure_layer_rather_than_redefining_it():
    """The whole point of loading spike_api.py by path is not duplicating its
    detectors -- this pins that the names actually resolve to the same objects.
    """
    assert spike.CheckFailed is spike.spike_api.CheckFailed
    assert spike.determinism_report is spike.spike_api.determinism_report
    assert spike.sensitivity_report is spike.spike_api.sensitivity_report
    assert spike.Report is spike.spike_api.Report
