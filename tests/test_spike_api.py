"""The spike's pure layer, exercised on the dev machine where Isaac cannot run.

``spikes/spike_api.py`` is written blind against README §4.4 and only ever runs on
the pod, so the parts that can be tested here are tested here: the detectors, and
the machinery that keeps one failing check from taking the other thirteen down
with it.

Most of this file is **negative controls**. A determinism gate that has only ever
been seen to pass is not a gate -- it is a detector nobody has watched detect
anything (README §10.1). Each fake below is a specific way the real pipeline
could be broken, and the test asserts the corresponding report *notices*:

``StaleCapture``
    renders the same frame whatever the state. Perfectly deterministic, perfectly
    order-independent, and completely worthless -- the failure the spike ranks
    sensitivity first to catch.
``AliasedCapture``
    hands back a view onto one reused buffer, so two captures compare equal
    because they are the same memory.
``LeakyCapture``
    a temporal denoiser: each frame is nudged toward the previous one, which is
    exactly the x -> x' correlation §7.1 calls the most dangerous artefact.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SPIKE = Path(__file__).resolve().parent.parent / "spikes" / "spike_api.py"


def _load():
    """Import the spike by path -- ``spikes/`` is a script directory, not a package.

    Registered in ``sys.modules`` before execution because ``@dataclass`` resolves
    the module's own annotations through it, and ``from __future__ import
    annotations`` makes every one of them a string.
    """
    spec = importlib.util.spec_from_file_location("spike_api", SPIKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


spike = _load()


def frame(value: int, *, size: int = 8) -> torch.Tensor:
    return torch.full((1, size, size, 3), value, dtype=torch.uint8)


class DeterministicCapture:
    """The pipeline we want: one frame per state, byte-identical on repeat."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state: int) -> torch.Tensor:
        self.calls += 1
        return frame(state)


class StaleCapture:
    """Renders the same frame no matter the state."""

    def __call__(self, state: int) -> torch.Tensor:
        return frame(17)


class NoisyCapture:
    """Free-running MC sampling: a different image every call (§7.1 failure (a))."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state: int) -> torch.Tensor:
        self.calls += 1
        return frame((state + 7 * self.calls) % 256)


class SubtleLeakCapture:
    """A denoiser leak of exactly one grey level, regardless of the two states.

    Separate from :class:`LeakyCapture`, whose leak scales with the distance
    between the states -- which cannot be small *and* leave the states
    distinguishable at the same time.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, state: int) -> torch.Tensor:
        self.calls += 1
        return frame(state + (self.calls % 2))


class LeakyCapture:
    """Temporal accumulation: each frame is pulled toward the previous one."""

    def __init__(self) -> None:
        self.previous: int | None = None

    def __call__(self, state: int) -> torch.Tensor:
        value = state if self.previous is None else (state + self.previous) // 2
        self.previous = state
        return frame(value)


class AliasedCapture:
    """One buffer, overwritten in place -- every capture is the same memory."""

    def __init__(self) -> None:
        self.buffer = torch.zeros((1, 8, 8, 3), dtype=torch.uint8)

    def __call__(self, state: int) -> torch.Tensor:
        self.buffer.fill_(state)
        return self.buffer


def test_spike_imports_with_the_isaac_roots_blocked():
    """The spike runs under Isaac's interpreter but must import without it.

    Same discipline as README §4.2 and ``tests/test_import_guard.py``: every Isaac
    import lives inside a function, after ``SimulationApp`` exists. A module-level
    one here would fail only on the pod, one container restart per violation.
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

spec = importlib.util.spec_from_file_location("spike_api", {str(SPIKE)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_uint8_differences_do_not_wrap():
    """``a - b`` on uint8 wraps, turning the largest possible difference into 1.

    Every threshold in the spike would then be measured against a number ~255x
    too small, and a wildly non-deterministic renderer would read as perfect.
    """
    assert spike.mean_abs_diff(frame(255), frame(0)) == pytest.approx(255.0)
    assert spike.max_abs_diff(frame(255), frame(0)) == pytest.approx(255.0)


def test_determinism_report_passes_a_clean_pipeline():
    report = spike.determinism_report(DeterministicCapture(), 10, 200, tol=0.0)
    assert report["deterministic"]
    assert report["order_independent_bitwise"]
    assert report["back_to_back_bitwise"]
    assert not report["buffers_aliased"]
    assert report["states_distinguishable"]


def test_determinism_report_catches_a_stale_pipeline():
    """The one that matters: a frozen renderer passes every *other* signal here."""
    report = spike.determinism_report(StaleCapture(), 10, 200, tol=0.0)
    # It is flawlessly "deterministic" on the narrow reading...
    assert report["order_independent_mad"] == 0.0
    assert report["back_to_back_mad"] == 0.0
    # ...and still must not be reported as usable.
    assert not report["states_distinguishable"]
    assert not report["deterministic"]


def test_determinism_report_catches_sampling_noise():
    report = spike.determinism_report(NoisyCapture(), 10, 200, tol=0.0)
    assert not report["deterministic"]
    assert report["order_independent_mad"] > 0.0


def test_determinism_report_catches_a_temporal_leak():
    """The x -> x' correlation an encoder can exploit without learning anything."""
    report = spike.determinism_report(LeakyCapture(), 40, 200, tol=0.0)
    assert not report["deterministic"]
    assert report["back_to_back_mad"] > 0.0
    assert not report["back_to_back_bitwise"]


def test_determinism_report_catches_an_aliased_buffer():
    """Equal because it is the same memory, not because the render is stable."""
    report = spike.determinism_report(AliasedCapture(), 10, 200, tol=0.0)
    assert report["buffers_aliased"]
    assert not report["deterministic"]


def test_a_threshold_would_hide_the_leak_that_bitwise_catches():
    """Why §7.1 insists acceptance is bitwise rather than a tolerance.

    A leak of a fraction of one grey level passes any tolerance someone would
    plausibly write down, while being precisely the structure the encoder is
    trained to find.
    """
    subtle = spike.determinism_report(SubtleLeakCapture(), 50, 200, tol=1.0)
    assert subtle["deterministic"], "this leak is below a 1/255 tolerance"
    assert subtle["back_to_back_mad"] <= 1.0
    assert not subtle["back_to_back_bitwise"], "and bitwise still sees it"


def test_convergence_report_finds_the_smallest_settled_depth():
    def capture_at_depth(depth: int) -> torch.Tensor:
        # Settles exactly at 8 and never moves again.
        return frame(100 if depth >= 8 else 100 - 8 // depth)

    report = spike.convergence_report(capture_at_depth, [1, 2, 4, 8, 16, 32], tol=0.0)
    assert report["smallest_converged_depth"] == 8
    assert report["changed_across_depths"]
    assert report["curve"][0]["depth"] == 1


def test_convergence_report_flags_a_curve_that_never_moved():
    """Everything "converged" at depth 1 is a stale pipeline, not a cheap one."""
    report = spike.convergence_report(lambda depth: frame(42), [1, 2, 4], tol=0.0)
    assert report["smallest_converged_depth"] == 1
    assert not report["changed_across_depths"]


def test_convergence_report_needs_more_than_one_depth():
    with pytest.raises(ValueError, match="at least two depths"):
        spike.convergence_report(lambda depth: frame(1), [4], tol=0.0)


def test_sensitivity_report_measures_response_against_the_noise_floor():
    captures = {0: frame(10), 1: frame(90), 2: frame(11)}
    report = spike.sensitivity_report(
        lambda state: captures[state],
        0,
        {"arm_only": 1, "cube_only": 2},
        noise_floor=0.5,
        min_ratio=10.0,
    )
    assert report["responses"]["arm_only"]["responsive"]
    # 1 grey level against a 0.5 floor is only 2x -- under the 10x bar, so this
    # is a factor the pipeline is very nearly blind to.
    assert not report["responses"]["cube_only"]["responsive"]
    assert not report["all_responsive"]


def test_sensitivity_report_catches_a_pipeline_blind_to_the_state():
    stale = StaleCapture()
    report = spike.sensitivity_report(stale, 0, {"arm_only": 1, "cube_only": 2}, noise_floor=0.0)
    assert not report["all_responsive"]
    assert report["responses"]["arm_only"]["mad_vs_base"] == 0.0


def test_frame_hash_is_content_addressed():
    assert spike.frame_hash(frame(7)) == spike.frame_hash(frame(7))
    assert spike.frame_hash(frame(7)) != spike.frame_hash(frame(8))


def test_report_never_lets_one_check_stop_the_rest():
    """One boot, one full report: the whole reason the spike is shaped this way."""
    report = spike.Report(verbose=False)

    def exploding():
        raise RuntimeError("kit fell over")

    def failing():
        raise spike.CheckFailed("answered, and the answer is no")

    def skipping():
        raise spike.CheckSkipped("nothing to compare against yet")

    report.run("first", lambda: {"ok": True})
    report.run("exploded", exploding)
    report.run("failed", failing)
    report.run("skipped", skipping)
    report.run("last", lambda: {"still_ran": True})

    assert [r.status for r in report.results] == ["PASS", "FAIL", "FAIL", "SKIP", "PASS"]
    assert report.counts() == {"PASS": 2, "FAIL": 2, "SKIP": 1}
    assert report.facts_of("last") == {"still_ran": True}
    # An unexpected exception has to arrive with evidence attached, or the pod
    # session is spent guessing rather than fixing.
    exploded = next(r for r in report.results if r.name == "exploded")
    assert "kit fell over" in exploded.note
    assert "RuntimeError" in exploded.note


def test_report_serialises_for_facts_json():
    report = spike.Report(verbose=False)
    report.run("measured", lambda: {"value": 3})
    payload = report.to_dict()
    assert payload["summary"]["PASS"] == 1
    assert payload["checks"][0]["facts"] == {"value": 3}
    assert "measured" in report.table()
