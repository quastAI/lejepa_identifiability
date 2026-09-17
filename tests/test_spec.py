"""Tier 0: handle bookkeeping and the squash (README §5.1, §5.2)."""

import pytest
import torch

from idtb.latents import Handle, LatentSpec

SPEC = LatentSpec((Handle("arm.j0", 0.0, 1.5), Handle("cube.x", 0.4, 0.1)))

MIXED_SPEC = LatentSpec(
    (
        Handle("arm.j0", 0.0, 1.5, group="base"),
        Handle("cube.x", 0.4, 0.1, group="base"),
        Handle("cube.size", 0.05, 0.02, group="full"),
        Handle("light.intensity", 500.0, 200.0, group="style"),
        Handle("cam.jitter.x", 0.0, 0.01, group="style"),
    )
)


def test_registry_bookkeeping():
    assert SPEC.n == 2
    assert SPEC.roles == ("arm.j0", "cube.x")
    assert SPEC.index("cube.x") == 1


def test_from_limits_keeps_a_fraction_of_the_measured_half_range():
    handle = Handle.from_limits("arm.j1", lo=-1.0, hi=3.0, fraction=0.5)
    assert handle.center == 1.0
    assert handle.radius == 1.0


@pytest.mark.parametrize(
    "build",
    [
        lambda: Handle("bad", 0.0, 0.0),
        lambda: Handle.from_limits("bad", 1.0, 1.0),
        lambda: LatentSpec((Handle("dup", 0.0, 1.0), Handle("dup", 1.0, 1.0))),
    ],
)
def test_rejects_degenerate_specs(build):
    with pytest.raises(ValueError):
        build()


def test_squash_never_leaves_the_physical_range():
    """Every sampled state is physically valid by construction, for any z."""
    z = torch.linspace(-50, 50, 64).unsqueeze(1).expand(-1, SPEC.n).contiguous()
    phi = SPEC.squash(z)
    for dim, handle in enumerate(SPEC.handles):
        assert (phi[:, dim] - handle.center).abs().max() <= handle.radius


def test_squash_stays_strictly_inside_away_from_saturation():
    """Strict interiority holds until float32 tanh rounds to 1 -- see saturation test."""
    z = torch.linspace(-4, 4, 64).unsqueeze(1).expand(-1, SPEC.n).contiguous()
    phi = SPEC.squash(z)
    for dim, handle in enumerate(SPEC.handles):
        assert (phi[:, dim] - handle.center).abs().max() < handle.radius


def test_squash_is_strictly_monotonic_per_dimension():
    """Monotonicity is what keeps phi injective, hence g injective (README §5.1)."""
    z = torch.linspace(-4, 4, 500).unsqueeze(1).expand(-1, SPEC.n).contiguous()
    assert (SPEC.squash(z).diff(dim=0) > 0).all()


def test_squash_rejects_a_latent_of_the_wrong_width():
    with pytest.raises(ValueError):
        SPEC.squash(torch.zeros(3, SPEC.n + 1))


def test_saturation_flags_exactly_where_float32_injectivity_dies():
    spec = LatentSpec((Handle("arm.j0", 0.0, 1.0),))
    atol = 1e-6  # a plausible physical read-back tolerance
    z = torch.tensor([[0.0], [3.0], [9.0], [12.0]])
    phi = spec.squash(z)

    assert spec.saturated(phi, atol=atol).squeeze(1).tolist() == [False, False, True, True]
    # ...and the flagged pair really has collapsed: distinct latents, one state.
    assert phi[2] == phi[3]
    assert phi[0] != phi[1]


# --- Latent groups (README §5.2, §5.4.1) --------------------------------------


def test_group_defaults_to_base():
    assert Handle("arm.j0", 0.0, 1.5).group == "base"


def test_rejects_unknown_group():
    with pytest.raises(ValueError):
        Handle("bad", 0.0, 1.0, group="nuisance")
    with pytest.raises(ValueError):
        Handle.from_limits("bad", -1.0, 1.0, group="nuisance")


def test_from_limits_passes_group_through():
    handle = Handle.from_limits("cube.size", lo=0.0, hi=0.04, group="full")
    assert handle.group == "full"


def test_out_of_order_spec_raises():
    """base, then full, then style -- the ordering that makes dims("base") a prefix."""
    with pytest.raises(ValueError):
        LatentSpec(
            (
                Handle("light.intensity", 500.0, 200.0, group="style"),
                Handle("arm.j0", 0.0, 1.5, group="base"),
            )
        )


def test_dims_are_cumulative_for_base_and_full_but_not_style():
    assert MIXED_SPEC.dims("base") == (0, 1)
    assert MIXED_SPEC.dims("full") == (0, 1, 2)
    assert MIXED_SPEC.dims("style") == (3, 4)
    # base is a *prefix* of full, by construction of the ordering contract.
    assert MIXED_SPEC.dims("full")[: len(MIXED_SPEC.dims("base"))] == MIXED_SPEC.dims("base")


def test_dims_rejects_unknown_group():
    with pytest.raises(ValueError):
        MIXED_SPEC.dims("nuisance")


def test_group_of_dim_matches_handle_order():
    assert MIXED_SPEC.group_of_dim == ("base", "base", "full", "style", "style")


def test_subset_squash_agrees_with_full_spec_on_those_dims():
    z = torch.randn(8, MIXED_SPEC.n)
    full_phi = MIXED_SPEC.squash(z)
    for group in ("base", "full", "style"):
        dims = MIXED_SPEC.dims(group)
        subset = MIXED_SPEC.subset(group)
        assert torch.equal(subset.squash(z[:, dims]), full_phi[:, dims])


def test_rho_vector_puts_rho_task_on_base_and_full_rho_style_on_style():
    rho_vec = MIXED_SPEC.rho_vector(0.95, rho_style=0.1)
    assert rho_vec.tolist() == pytest.approx([0.95, 0.95, 0.95, 0.1, 0.1])


def test_rho_vector_defaults_style_to_zero():
    rho_vec = MIXED_SPEC.rho_vector(0.95)
    assert rho_vec.tolist() == pytest.approx([0.95, 0.95, 0.95, 0.0, 0.0])
