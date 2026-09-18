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


# --- dump_render_product_rtx_attributes: experiment A's diagnostic filter ------


class _FakeAttr:
    def __init__(self, name, value, *, raises=False):
        self._name = name
        self._value = value
        self._raises = raises

    def GetName(self):
        return self._name

    def Get(self):
        if self._raises:
            raise RuntimeError("boom")
        return self._value


class _FakePrim:
    def __init__(self, attrs):
        self._attrs = attrs

    def GetAttributes(self):
        return self._attrs


def test_dump_render_product_rtx_attributes_filters_by_namespace():
    prim = _FakePrim(
        [
            _FakeAttr("omni:rtx:rendermode", "PathTracing"),
            _FakeAttr("omni:rtx:pt:samplesPerPixel", 1),
            _FakeAttr("xformOp:translate", (0.0, 0.0, 0.0)),
        ]
    )
    values = spike.dump_render_product_rtx_attributes(prim)
    assert values == {"omni:rtx:rendermode": "PathTracing", "omni:rtx:pt:samplesPerPixel": 1}


def test_dump_render_product_rtx_attributes_records_unreadable_attrs_without_raising():
    prim = _FakePrim([_FakeAttr("omni:rtx:broken", None, raises=True)])
    values = spike.dump_render_product_rtx_attributes(prim)
    assert "omni:rtx:broken" in values
    assert "unreadable" in values["omni:rtx:broken"]


def test_dump_render_product_rtx_attributes_empty_prim_is_empty_dict():
    assert spike.dump_render_product_rtx_attributes(_FakePrim([])) == {}


# --- Round 2 preset registry (docs/PLAN.md Phase 3c) ----------------------------


def test_extra_presets_include_a_none_noop_default():
    """`none` must be a real, empty no-op -- an unflagged run has to reproduce
    exactly the Round-1 configuration that produced README §7.5's verdict.
    """
    assert spike.EXTRA_PRESETS["none"] == {}


def test_extra_presets_render_mode_overrides_are_distinct_strings():
    """`realtime_rtpt_caches_off` and `minimal` each pick a different
    `/rtx/rendermode` -- if these ever collided, experiment D's two candidates
    would silently test the same thing twice.
    """
    modes = {
        name: preset["/rtx/rendermode"]
        for name, preset in spike.EXTRA_PRESETS.items()
        if "/rtx/rendermode" in preset
    }
    assert len(set(modes.values())) == len(modes)


def test_reset_accum_on_time_change_is_a_single_bool_key():
    assert spike.RESET_ACCUM_ON_TIME_CHANGE == {"/rtx/resetPtAccumOnAnimTimeChange": True}


def test_disable_fabric_transform_sync_is_a_single_bool_key():
    assert spike.DISABLE_FABRIC_TRANSFORM_SYNC == {
        "/rtx/hydra/readTransformsFromFabricInRenderDelegate": False
    }


# --- resolve_cadence_reset: IsaacLab#6609's cadence-invalidation lookup ---------


class _FakeRenderContext:
    def __init__(self, **methods):
        for name, fn in methods.items():
            setattr(self, name, fn)


class _FakeSim:
    def __init__(self, render_context=None):
        self.render_context = render_context


def test_resolve_cadence_reset_prefers_reset_transform_cadence():
    calls = []
    sim = _FakeSim(
        _FakeRenderContext(
            reset_transform_cadence=lambda: calls.append("transform"),
            reset_scene_state_cadence=lambda: calls.append("scene_state"),
        )
    )
    name, fn = spike.resolve_cadence_reset(sim)
    assert name == "sim.render_context.reset_transform_cadence()"
    fn()
    assert calls == ["transform"]


def test_resolve_cadence_reset_falls_back_to_reset_scene_state_cadence():
    sim = _FakeSim(_FakeRenderContext(reset_scene_state_cadence=lambda: None))
    name, fn = spike.resolve_cadence_reset(sim)
    assert name == "sim.render_context.reset_scene_state_cadence()"
    assert fn is not None


def test_resolve_cadence_reset_reports_missing_render_context():
    name, fn = spike.resolve_cadence_reset(_FakeSim(render_context=None))
    assert fn is None
    assert "render_context not found" in name


def test_resolve_cadence_reset_reports_render_context_without_either_method():
    name, fn = spike.resolve_cadence_reset(_FakeSim(_FakeRenderContext()))
    assert fn is None
    assert "neither reset_transform_cadence nor reset_scene_state_cadence" in name


# --- resolve_render_tick: the capture loop's render-tick lever -------------------


def test_resolve_render_tick_defaults_to_sim_render():
    sim = _FakeSim()
    sim.render = lambda: "rendered"
    fn, desc = spike.resolve_render_tick("sim_render", sim)
    assert fn is sim.render
    assert desc == "sim.render()"


# --- get_or_create_shader_input: table.roughness's fallback-create path --------


class _FakeShaderInput:
    def __init__(self, value=None):
        self.value = value

    def Set(self, value):
        self.value = value


class _FakeShader:
    """Duck-typed ``UsdShade.Shader`` stand-in -- only ``GetInput``/``CreateInput``
    matter to :func:`get_or_create_shader_input`, same technique as this file's
    own :func:`resolve_bound_shader` test fakes below."""

    def __init__(self, declared=None):
        self._inputs = dict(declared or {})
        self.created: list[tuple[str, object]] = []

    def GetInput(self, name):
        return self._inputs.get(name)

    def CreateInput(self, name, sdf_type):
        self.created.append((name, sdf_type))
        new_input = _FakeShaderInput()
        self._inputs[name] = new_input
        return new_input


def test_get_or_create_shader_input_reuses_an_already_declared_input():
    existing = _FakeShaderInput(0.5)
    shader = _FakeShader(declared={"roughness": existing})

    result = spike.get_or_create_shader_input(shader, "roughness", "Float")

    assert result is existing
    assert shader.created == []


def test_get_or_create_shader_input_creates_a_missing_input():
    shader = _FakeShader()

    result = spike.get_or_create_shader_input(shader, "roughness", "Float")

    assert shader.created == [("roughness", "Float")]
    result.Set(0.9)
    assert shader.GetInput("roughness").value == 0.9


# --- reused pure layer, sanity that the load actually worked --------------------


def test_reuses_spike_api_pure_layer_rather_than_redefining_it():
    """The whole point of loading spike_api.py by path is not duplicating its
    detectors -- this pins that the names actually resolve to the same objects.
    """
    assert spike.CheckFailed is spike.spike_api.CheckFailed
    assert spike.determinism_report is spike.spike_api.determinism_report
    assert spike.sensitivity_report is spike.spike_api.sensitivity_report
    assert spike.Report is spike.spike_api.Report
