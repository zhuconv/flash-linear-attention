# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang
# Mamba-3 configuration adapted for FLA framework.

import warnings

from transformers.configuration_utils import PretrainedConfig


class Mamba3Config(PretrainedConfig):
    """
    Configuration for Mamba-3 (ICLR 2026) model in the FLA framework.

    Mamba-3 replaces Mamba-2's conv1d + SSD with a RoPE-augmented SSM using
    explicit Q/K/V structure and per-token A/dt projections from input.

    Args:
        head_dim: Dimension per SSM head (default 64).
        vocab_size: Vocabulary size (default 32000).
        hidden_size: Model dimension (default 2048).
        state_size: SSM state dimension N (default 128).
        num_hidden_layers: Number of layers (default 48).
        expand: Expansion factor for inner dimension (default 2).
        n_groups: Number of B/C groups (default 1).
        rope_fraction: Fraction of state_size used for RoPE, 0.5 or 1.0 (default 0.5).
        chunk_size: Chunk size for SISO kernel (default 64).
        is_mimo: Use MIMO mode (default False, SISO is simpler and faster).
        mimo_rank: MIMO rank if is_mimo=True (default 4).
        dt_min: Minimum dt for initialization (default 0.001).
        dt_max: Maximum dt for initialization (default 0.1).
        dt_init_floor: Minimum clamping for dt init (default 1e-4).
        A_floor: Minimum magnitude for A (default 1e-4).
        is_outproj_norm: Apply RMSNorm after output projection (default False).
        use_bias: Use bias in in_proj/out_proj (default False).
        rmsnorm: Use RMSNorm (default True).
        residual_in_fp32: Keep residuals in fp32 (default True).
        rescale_prenorm_residual: GPT-2 style weight scaling (default True).
        use_cache: Enable cache for inference (default True).
        fuse_norm: Use fused Triton norm kernels (default True).
        fuse_cross_entropy: Use fused cross-entropy (default True).
    """

    model_type = "mamba3"

    def __init__(
        self,
        head_dim: int = 64,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        state_size: int = 128,
        num_hidden_layers: int = 48,
        norm_eps: float = 1e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        expand: int = 2,
        n_groups: int = 1,
        rope_fraction: float = 0.5,
        chunk_size: int = 64,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        use_bias: bool = False,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rmsnorm: bool = True,
        residual_in_fp32: bool = True,
        rescale_prenorm_residual: bool = True,
        use_cache: bool = True,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.expand = expand
        self.head_dim = head_dim
        self.num_heads = int(self.expand * self.hidden_size / self.head_dim)
        self.n_groups = n_groups
        self.rope_fraction = rope_fraction
        self.chunk_size = chunk_size
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init_floor = dt_init_floor
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.use_bias = use_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rmsnorm = rmsnorm
        self.residual_in_fp32 = residual_in_fp32
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.use_cache = use_cache
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.tie_word_embeddings = tie_word_embeddings

        assert rope_fraction in (0.5, 1.0), "rope_fraction must be 0.5 or 1.0"
        if dt_min <= 0 or dt_max < dt_min:
            raise ValueError("`dt_min` and `dt_max` must satisfy 0 < dt_min <= dt_max.")
        if dt_init_floor <= 0:
            raise ValueError("`dt_init_floor` must be > 0.")
        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError("`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot both be True.")
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` may reduce precision. "
                "Disable if you observe loss divergence.",
            )

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
