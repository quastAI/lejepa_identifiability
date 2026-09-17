"""Pure latent layer: OU pair sampling and the latent -> physical-state map."""

from idtb.latents.ou import sample_ou_pairs
from idtb.latents.spec import GROUP_ORDER, Handle, LatentSpec

__all__ = ["GROUP_ORDER", "Handle", "LatentSpec", "sample_ou_pairs"]
