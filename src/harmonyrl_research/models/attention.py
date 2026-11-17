"""Transformer blocks with rotary attention over times in 8th notes.

Rotary embeddings are applied here rather than taken from a library, which assumes positions
0, 1, 2, ... Here an event sits at its start time, and cross-attention compares events and frames
on one time axis.
"""

import torch
from torch import nn
from torch.nn import functional

ROTARY_BASE = 10000.0


def rotate(x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
	"""Rotary embedding of `x` (batch, heads, length, dim) at `times` (batch, length)."""
	half = x.shape[-1] // 2
	steps = torch.arange(half, device=x.device, dtype=torch.float32) / half
	angles = times.float()[:, None, :, None] * ROTARY_BASE**-steps
	cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
	first, second = x[..., :half], x[..., half:]

	return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class Attention(nn.Module):
	"""Multi-head attention from `x` to `context` with rotary time positions."""

	def __init__(self, dim: int, heads: int, dropout: float) -> None:
		super().__init__()

		if dim % (2 * heads):
			raise ValueError(f"dim {dim} must split into {heads} heads of even size")

		self.heads = heads
		self.dropout = dropout
		self.query = nn.Linear(dim, dim)
		self.key_value = nn.Linear(dim, 2 * dim)
		self.out = nn.Linear(dim, dim)

	def forward(
		self,
		x: torch.Tensor,
		times: torch.Tensor,
		context: torch.Tensor,
		context_times: torch.Tensor,
		mask: torch.Tensor,
	) -> torch.Tensor:
		"""`mask` (batch, length, context lenth) is True where `x` may attend to `context`."""
		batch, length, dim = x.shape
		query = self.query(x).view(batch, length, self.heads, -1).transpose(1, 2)
		key, value = (
			self.key_value(context)
			.view(batch, context.shape[1], 2, self.heads, -1)
			.permute(2, 0, 3, 1, 4)
		)
		y = functional.scaled_dot_product_attention(
			rotate(query, times),
			rotate(key, context_times),
			value,
			attn_mask=mask[:, None],
			dropout_p=self.dropout if self.training else 0.0,
		)

		return self.out(y.transpose(1, 2).reshape(batch, length, dim))


class Block(nn.Module):
	"""Pre-norm Transformer layer: self-attention, optional cross-attention, feed-forward."""

	def __init__(self, dim: int, heads: int, ff_mult: int, dropout: float, cross: bool) -> None:
		super().__init__()

		self.self_norm = nn.LayerNorm(dim)
		self.self_attention = Attention(dim, heads, dropout)

		self.cross_norm = nn.LayerNorm(dim) if cross else None
		self.cross_attention = Attention(dim, heads, dropout) if cross else None

		self.ff_norm = nn.LayerNorm(dim)
		self.ff = nn.Sequential(
			nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim)
		)

		self.drop = nn.Dropout(dropout)

	def forward(
		self,
		x: torch.Tensor,
		times: torch.Tensor,
		mask: torch.Tensor,
		context: torch.Tensor | None = None,
		context_times: torch.Tensor | None = None,
		context_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		"""Masks are True where attention is allowed. Context arguments go with cross-attention."""
		h = self.self_norm(x)
		x = x + self.drop(self.self_attention(h, times, h, times, mask))
		if self.cross_attention is not None and self.cross_norm is not None:
			if context is None or context_times is None or context_mask is None:
				raise ValueError("a cross-attention block needs context, its times, and a mask")

			h = self.cross_norm(x)
			x = x + self.drop(self.cross_attention(h, times, context, context_times, context_mask))

		return x + self.drop(self.ff(self.ff_norm(x)))
