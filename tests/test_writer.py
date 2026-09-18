"""``idtb.sim.writer``'s pure layer -- everything above the "Isaac layer"
banner, same split as the spikes (README §7.2/§7.5)."""

from __future__ import annotations

import pytest
import torch

from idtb.latents import Handle, LatentSpec
from idtb.sim.writer import handle_infos, resolve_cube_edge_m, split_camera_jitter

SPEC = LatentSpec(
    (
        Handle("cube.x", 0.0, 0.3, group="base"),
        Handle("cube.size", 0.05, 0.02, group="full"),
        Handle("cam.jitter.x", 0.0, 0.5, group="style"),
    )
)

BASE_SPEC = LatentSpec((Handle("cube.x", 0.0, 0.3, group="base"),))


def test_resolve_cube_edge_m_reads_the_active_dimension():
    phi = SPEC.squash(torch.zeros(1, SPEC.n))
    phi[0, SPEC.index("cube.size")] = 0.09
    assert resolve_cube_edge_m(SPEC, phi, default_edge_m=0.06) == pytest.approx(0.09)


def test_resolve_cube_edge_m_falls_back_when_size_is_not_active():
    phi = BASE_SPEC.squash(torch.zeros(1, BASE_SPEC.n))
    assert resolve_cube_edge_m(BASE_SPEC, phi, default_edge_m=0.06) == 0.06


def test_split_camera_jitter_reads_both_axes():
    phi = SPEC.squash(torch.zeros(1, SPEC.n))
    phi[0, SPEC.index("cam.jitter.x")] = 0.05
    dx, dy = split_camera_jitter(SPEC, phi)
    assert dx == pytest.approx(0.05)
    assert dy == 0.0


def test_split_camera_jitter_defaults_to_zero_when_absent():
    phi = BASE_SPEC.squash(torch.zeros(1, BASE_SPEC.n))
    assert split_camera_jitter(BASE_SPEC, phi) == (0.0, 0.0)


def test_handle_infos_reports_the_write_path_per_role():
    info = handle_infos(SPEC)
    assert info["cube.x"].write_path == "root_state"
    assert info["cube.size"].write_path == "attribute"
    assert info["cam.jitter.x"].write_path == "attribute"
