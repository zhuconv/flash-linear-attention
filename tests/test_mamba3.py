#!/usr/bin/env python3
"""Tests for FLA Mamba3 implementation.

Tests:
1. Numerical correctness: FLA vs native mamba_ssm output match
2. Speed comparison: FLA vs native forward+backward timing
3. 1k-step training loss: FLA vs native should converge similarly
"""

import sys
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Skip if no GPU ----
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ---------------------------------------------------------------------------
# Helper: create matched FLA and native models with identical weights
# ---------------------------------------------------------------------------
def make_models(d_model=768, n_layer=4, vocab_size=1024, dtype=torch.bfloat16, device="cuda"):
    """Create FLA Mamba3ForCausalLM and native MambaLMHeadModel with shared weights."""
    from fla.models.mamba3 import Mamba3Config, Mamba3ForCausalLM

    fla_config = Mamba3Config(
        hidden_size=d_model,
        num_hidden_layers=n_layer,
        vocab_size=vocab_size,
        state_size=64,
        head_dim=64,
        expand=2,
        n_groups=1,
        rope_fraction=0.5,
        chunk_size=64,
        fuse_cross_entropy=False,
        fuse_norm=False,
    )
    fla_model = Mamba3ForCausalLM(fla_config).to(device=device, dtype=dtype)

    # Native model
    sys.path.insert(0, "/work/11012/jiajunzhu2002/mamba3")
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    native_config = MambaConfig(
        d_model=d_model,
        n_layer=n_layer,
        d_intermediate=0,
        vocab_size=vocab_size,
        ssm_cfg=dict(
            layer="Mamba3",
            d_state=64,
            expand=2,
            headdim=64,
            rope_fraction=0.5,
            is_mimo=False,
            chunk_size=64,
        ),
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=1,
        tie_embeddings=True,
    )
    native_model = MambaLMHeadModel(native_config, device=device, dtype=dtype)

    return fla_model, native_model


def copy_weights_fla_to_native(fla_model, native_model):
    """Copy FLA weights to native model for numerical comparison.

    The weight names differ between the two frameworks; this maps them.
    """
    fla_sd = fla_model.state_dict()
    native_sd = native_model.state_dict()

    # Embedding
    native_sd["backbone.embedding.weight"].copy_(fla_sd["backbone.embeddings.weight"])

    # LM head (may be tied)
    if "lm_head.weight" in native_sd:
        native_sd["lm_head.weight"].copy_(fla_sd["lm_head.weight"])

    # Final norm
    native_sd["backbone.norm_f.weight"].copy_(fla_sd["backbone.norm_f.weight"])

    # Per-layer weights
    for i in range(len(native_model.backbone.layers)):
        fla_prefix = f"backbone.layers.{i}"
        nat_prefix = f"backbone.layers.{i}"

        # Block norm
        native_sd[f"{nat_prefix}.norm.weight"].copy_(fla_sd[f"{fla_prefix}.norm.weight"])

        # Mixer weights
        mixer_keys = [
            "mixer.in_proj.weight",
            "mixer.out_proj.weight",
            "mixer.dt_bias",
            "mixer.B_bias",
            "mixer.C_bias",
            "mixer.B_norm.weight",
            "mixer.C_norm.weight",
            "mixer.D",
        ]
        for k in mixer_keys:
            fla_key = f"{fla_prefix}.{k}"
            nat_key = f"{nat_prefix}.{k}"
            if fla_key in fla_sd and nat_key in native_sd:
                native_sd[nat_key].copy_(fla_sd[fla_key])

    native_model.load_state_dict(native_sd)


# ---------------------------------------------------------------------------
# Test 1: Numerical correctness
# ---------------------------------------------------------------------------
class TestNumericalCorrectness:

    def test_forward_output_shape(self):
        """FLA Mamba3 produces correct output shapes."""
        from fla.models.mamba3 import Mamba3Config, Mamba3ForCausalLM

        config = Mamba3Config(
            hidden_size=256, num_hidden_layers=2, vocab_size=512,
            state_size=32, head_dim=32, expand=2, chunk_size=32,
            fuse_cross_entropy=False, fuse_norm=False,
        )
        model = Mamba3ForCausalLM(config).to("cuda", dtype=torch.bfloat16)

        input_ids = torch.randint(0, 512, (2, 64), device="cuda")
        output = model(input_ids)

        assert output.logits.shape == (2, 64, 512)

    def test_forward_backward(self):
        """FLA Mamba3 forward+backward runs without errors."""
        from fla.models.mamba3 import Mamba3Config, Mamba3ForCausalLM

        config = Mamba3Config(
            hidden_size=256, num_hidden_layers=2, vocab_size=512,
            state_size=32, head_dim=32, expand=2, chunk_size=32,
            fuse_cross_entropy=False, fuse_norm=False,
        )
        model = Mamba3ForCausalLM(config).to("cuda", dtype=torch.bfloat16)
        model.train()

        input_ids = torch.randint(0, 512, (2, 64), device="cuda")
        labels = torch.randint(0, 512, (2, 64), device="cuda")
        output = model(input_ids, labels=labels)

        assert output.loss is not None
        output.loss.backward()

        # Check gradients exist
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_fla_vs_native_logits_close(self):
        """FLA and native Mamba3 produce similar logits with shared weights."""
        fla_model, native_model = make_models(d_model=256, n_layer=2, vocab_size=512)
        copy_weights_fla_to_native(fla_model, native_model)

        fla_model.eval()
        native_model.eval()

        input_ids = torch.randint(0, 512, (2, 64), device="cuda")

        with torch.no_grad():
            fla_logits = fla_model(input_ids).logits
            native_logits = native_model(input_ids).logits

        # Allow some tolerance due to bf16 and different norm implementations
        rel_error = (fla_logits.float() - native_logits.float()).abs().mean() / native_logits.float().abs().mean()
        print(f"Relative error FLA vs native: {rel_error:.6f}")
        assert rel_error < 0.05, f"Relative error too high: {rel_error}"


# ---------------------------------------------------------------------------
# Test 2: Speed comparison
# ---------------------------------------------------------------------------
class TestSpeed:

    def test_forward_speed(self):
        """Benchmark FLA Mamba3 forward pass."""
        from fla.models.mamba3 import Mamba3Config, Mamba3ForCausalLM

        config = Mamba3Config(
            hidden_size=768, num_hidden_layers=12, vocab_size=32000,
            state_size=64, head_dim=64, expand=2, chunk_size=64,
            fuse_cross_entropy=False, fuse_norm=False,
        )
        model = Mamba3ForCausalLM(config).to("cuda", dtype=torch.bfloat16)
        model.eval()

        input_ids = torch.randint(0, 32000, (4, 1024), device="cuda")

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(input_ids)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        n_iters = 10
        with torch.no_grad():
            for _ in range(n_iters):
                model(input_ids)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        ms_per_iter = elapsed / n_iters * 1000
        tokens_per_sec = (4 * 1024 * n_iters) / elapsed
        print(f"FLA Mamba3 forward: {ms_per_iter:.1f} ms/iter, {tokens_per_sec:.0f} tok/s")

    def test_forward_backward_speed(self):
        """Benchmark FLA Mamba3 forward+backward."""
        from fla.models.mamba3 import Mamba3Config, Mamba3ForCausalLM

        config = Mamba3Config(
            hidden_size=768, num_hidden_layers=12, vocab_size=32000,
            state_size=64, head_dim=64, expand=2, chunk_size=64,
            fuse_cross_entropy=False, fuse_norm=False,
        )
        model = Mamba3ForCausalLM(config).to("cuda", dtype=torch.bfloat16)
        model.train()

        input_ids = torch.randint(0, 32000, (4, 1024), device="cuda")
        labels = torch.randint(0, 32000, (4, 1024), device="cuda")

        # Warmup
        for _ in range(3):
            loss = model(input_ids, labels=labels).loss
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        n_iters = 10
        for _ in range(n_iters):
            loss = model(input_ids, labels=labels).loss
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()
        elapsed = time.time() - start

        ms_per_iter = elapsed / n_iters * 1000
        print(f"FLA Mamba3 fwd+bwd: {ms_per_iter:.1f} ms/iter")


# ---------------------------------------------------------------------------
# Test 3: 1k-step training loss comparison
# ---------------------------------------------------------------------------
class TestTrainingLoss:

    @pytest.mark.slow
    def test_1k_steps_loss_close(self):
        """FLA and native Mamba3 achieve similar loss after 1k steps on random data."""
        torch.manual_seed(42)
        d_model, n_layer, vocab_size = 256, 4, 512
        seq_len, batch_size, max_steps = 128, 4, 1000
        lr = 6e-4

        # Create both models with same init
        fla_model, native_model = make_models(d_model, n_layer, vocab_size)
        copy_weights_fla_to_native(fla_model, native_model)

        fla_model.train()
        native_model.train()

        fla_opt = torch.optim.AdamW(fla_model.parameters(), lr=lr)
        native_opt = torch.optim.AdamW(native_model.parameters(), lr=lr)

        # Generate fixed random data
        torch.manual_seed(123)
        all_data = torch.randint(0, vocab_size, (max_steps, batch_size, seq_len + 1), device="cuda")

        fla_losses, native_losses = [], []
        for step in range(max_steps):
            data = all_data[step]
            input_ids = data[:, :-1]
            labels = data[:, 1:]

            # FLA forward
            fla_out = fla_model(input_ids)
            fla_loss = F.cross_entropy(
                fla_out.logits.float().reshape(-1, vocab_size),
                labels.reshape(-1),
            )
            fla_loss.backward()
            torch.nn.utils.clip_grad_norm_(fla_model.parameters(), 1.0)
            fla_opt.step()
            fla_opt.zero_grad()
            fla_losses.append(fla_loss.item())

            # Native forward
            native_out = native_model(input_ids)
            native_loss = F.cross_entropy(
                native_out.logits.float().reshape(-1, vocab_size),
                labels.reshape(-1),
            )
            native_loss.backward()
            torch.nn.utils.clip_grad_norm_(native_model.parameters(), 1.0)
            native_opt.step()
            native_opt.zero_grad()
            native_losses.append(native_loss.item())

            if step % 100 == 0:
                print(f"step {step}: fla={fla_losses[-1]:.4f} native={native_losses[-1]:.4f}")

        # Compare final losses (should be very close)
        fla_final = sum(fla_losses[-50:]) / 50
        native_final = sum(native_losses[-50:]) / 50
        diff = abs(fla_final - native_final)
        print(f"\nFinal avg-50 loss: FLA={fla_final:.4f}, Native={native_final:.4f}, diff={diff:.4f}")

        assert diff < 0.1, f"Loss difference too large: {diff:.4f} (FLA={fla_final:.4f}, Native={native_final:.4f})"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick smoke test when run directly
    print("=== Shape test ===")
    TestNumericalCorrectness().test_forward_output_shape()
    print("PASS")

    print("\n=== Forward+backward test ===")
    TestNumericalCorrectness().test_forward_backward()
    print("PASS")

    print("\n=== Speed test (forward) ===")
    TestSpeed().test_forward_speed()

    print("\n=== Speed test (fwd+bwd) ===")
    TestSpeed().test_forward_backward_speed()

    print("\n=== Numerical correctness vs native ===")
    TestNumericalCorrectness().test_fla_vs_native_logits_close()
    print("PASS")
