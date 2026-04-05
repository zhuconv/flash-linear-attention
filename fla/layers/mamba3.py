# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang
# Mamba-3 layer implementation for FLA framework.
# Adapted from state-spaces/mamba (Dao AI Lab, Goombalab).

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import get_layer_cache, update_layer_cache
from fla.modules.layernorm_gated import RMSNormGated

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
        from mamba_ssm.ops.triton.angle_cumsum import angle_dt
    except ImportError:
        mamba3_siso_combined, angle_dt = None, None
    try:
        from mamba_ssm.ops.cute.mamba3.mamba3_step_fn import mamba3_step_fn
    except ImportError:
        mamba3_step_fn = None
    is_fast_path_available = mamba3_siso_combined is not None

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class Mamba3(nn.Module):
    """Mamba-3 SISO layer compatible with the FLA framework.

    Unlike Mamba-2, Mamba-3:
    - Has no conv1d; B/C are projected directly and RMSNorm'd
    - Projects A and dt from input (not fixed A_log)
    - Uses RoPE via angle states
    - Has a 'trap' parameter for trapezoidal integration
    - Uses the mamba3_siso_combined Triton kernel
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        state_size: int = 128,
        expand: int = 2,
        head_dim: int = 64,
        n_groups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        use_bias: bool = False,
        norm_eps: float = 1e-5,
        layer_idx: int | None = None,
        # absorbed kwargs for compatibility
        num_heads: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.expand = expand
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.norm_eps = norm_eps

        self.mimo_rank = mimo_rank if is_mimo else 1
        if is_mimo:
            raise NotImplementedError("MIMO mode requires tilelang; only SISO is supported in FLA.")

        self.d_inner = int(self.expand * self.hidden_size)
        assert self.d_inner % self.head_dim == 0
        self.num_heads = self.d_inner // self.head_dim
        self.num_bc_heads = n_groups
        self.n_groups = n_groups

        # RoPE setup
        assert rope_fraction in (0.5, 1.0)
        self.rope_fraction = rope_fraction
        self.split_tensor_size = int(state_size * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Input projection: [z, x, B, C, dd_dt, dd_A, trap, angles]
        d_in_proj = (
            2 * self.d_inner
            + 2 * self.state_size * self.num_bc_heads * self.mimo_rank
            + 3 * self.num_heads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.hidden_size, d_in_proj, bias=use_bias)

        # dt bias (learnable, initialized via inverse softplus)
        _dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias)
        self.dt_bias._no_weight_decay = True
        self.dt_bias._no_reinit = True

        # B and C biases
        self.B_bias = nn.Parameter(torch.ones(self.num_heads, self.mimo_rank, self.state_size))
        self.C_bias = nn.Parameter(torch.ones(self.num_heads, self.mimo_rank, self.state_size))

        # B and C normalization
        self.B_norm = RMSNormGated(self.state_size, eps=norm_eps)
        self.C_norm = RMSNormGated(self.state_size, eps=norm_eps)

        # D skip parameter
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        # Optional output projection norm
        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner, eps=norm_eps, norm_before_gate=True,
                group_size=self.head_dim,
            )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.hidden_size, bias=use_bias)

    def _prefill_forward(
        self,
        hidden_states: torch.Tensor,
        use_cache: bool = False,
        input_states: tuple | None = None,
    ):
        """Prefill path: process full sequence via mamba3_siso_combined kernel."""
        batch, seqlen, _ = hidden_states.shape

        # Input projection and split
        zxBCdtAtrap = self.in_proj(hidden_states)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner, self.d_inner,
                self.state_size * self.num_bc_heads * self.mimo_rank,
                self.state_size * self.num_bc_heads * self.mimo_rank,
                self.num_heads, self.num_heads, self.num_heads,
                self.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.head_dim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.head_dim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        # Compute ADT and DT
        _A = -F.softplus(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # Angles: expand to all heads
        angles = angles.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)

        # Normalize B and C
        B = self.B_norm(B)
        C = self.C_norm(C)

        # Call SISO kernel with optional input states from cache
        y = mamba3_siso_combined(
            Q=C.squeeze(2),
            K=B.squeeze(2),
            V=x,
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=self.C_bias.squeeze(1),
            K_bias=self.B_bias.squeeze(1),
            Angles=angles,
            D=self.D,
            Z=z if not self.is_outproj_norm else None,
            chunk_size=self.chunk_size,
            Input_States=input_states,
            return_final_states=use_cache,
        )

        final_states = None
        if use_cache:
            y, last_angle, last_ssm, last_k, last_v, *_ = y
            # Store as tuple for FLA cache compatibility (offload/prefetch)
            final_states = (last_angle, last_ssm, last_k, last_v)

        y = rearrange(y, "b l h p -> b l (h p)")

        if self.is_outproj_norm:
            z_flat = rearrange(z, "b l h p -> b l (h p)")
            y = self.norm(y, z_flat)

        out = self.out_proj(y.to(hidden_states.dtype))
        return out, final_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if mamba3_siso_combined is None:
            raise ImportError(
                "Mamba3 requires `mamba_ssm` with Triton SISO kernels. "
                "Install via: pip install mamba_ssm (with triton>=3.5.0)"
            )

        last_state = get_layer_cache(self, past_key_values)

        # Reconstruct input_states tuple from cache if available
        input_states = None
        if last_state is not None:
            rec = last_state.get('recurrent_state')
            if rec is not None and isinstance(rec, tuple) and len(rec) == 4:
                input_states = rec

        # Apply attention mask only on prefill (not decode), matching Mamba2 behavior
        if last_state is None and attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        output, final_states = self._prefill_forward(hidden_states, use_cache, input_states)

        # Update cache: store states as tuple (compatible with FLA offload/prefetch)
        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=final_states,
            offset=hidden_states.shape[1],
        )

        return output, None, past_key_values
