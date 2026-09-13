"""Pure latent layer: OU pair sampling and the latent -> physical-state map."""

from idtb.latents.ou import sample_ou_pairs
from idtb.latents.spec import Handle, LatentSpec

__all__ = ["Handle", "LatentSpec", "sample_ou_pairs"]
