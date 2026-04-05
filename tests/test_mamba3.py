#!/usr/bin/env python3
"""Tests for FLA Mamba3 implementation.

Tests:
1. Numerical correctness: FLA vs native mamba_ssm output match
2. Speed comparison: FLA forward+backward timing
3. Cache / use_cache behavior
4. 1k-step training loss: FLA vs native should converge similarly
"""

import os
import sys
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Skip if no GPU ----
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# Path to native mamba_ssm (optional, for cross-validation tests)
MAMBA_SSM_PATH = os.environ.get("MAMBA_SSM_PATH", "/work/11012/jiajunzhu2002/mamba3")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _has_native_mamba3():
    """Check if native mamba_ssm with Mamba3 is importable."""
    try:
        if MAMBA_SSM_PATH not in sys.path:
            sys.path.insert(0, MAMBA_SSM_PATH)
        from mamba_ssm.modules.mamba3 import Mamba3  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


def make_fla_model(d_model=256, n_layer=2, vocab_size=512, dtype=torch.bfloat16, device="cuda"):
    from fla.models.mamba3 import Mamba3Config, Mamba3ForCausalLM

    config = Mamba3Config(
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
    return Mamba3ForCausalLM(config).to(device=device, dtype=dtype)


def make_native_model(d_model=256, n_layer=2, vocab_size=512, dtype=torch.bfloat16, device="cuda"):
    if MAMBA_SSM_PATH not in sys.path:
        sys.path.insert(0, MAMBA_SSM_PATH)
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    config = MambaConfig(
        d_model=d_model, n_layer=n_layer, d_intermediate=0, vocab_size=vocab_size,
        ssm_cfg=dict(layer="Mamba3", d_state=64, expand=2, headdim=64,
                     rope_fraction=0.5, is_mimo=False, chunk_size=64),
        rms_norm=True, residual_in_fp32=True, fused_add_norm=False,
        pad_vocab_size_multiple=1, tie_embeddings=False,
    )
    return MambaLMHeadModel(config, device=device, dtype=dtype)


def copy_weights_native_to_fla(native_model, fla_model):
    nat_sd = native_model.state_dict()
    fla_sd = fla_model.state_dict()
    for fla_key in fla_sd:
        nat_key = fla_key.replace("embeddings.", "embedding.")
        if nat_key in nat_sd:
            fla_sd[fla_key].copy_(nat_sd[nat_key])
        elif fla_key == "lm_head.weight" and "backbone.embedding.weight" in nat_sd:
            fla_sd[fla_key].copy_(nat_sd["backbone.embedding.weight"])
    fla_model.load_state_dict(fla_sd)


# ---------------------------------------------------------------------------
# Test 1: Basic correctness
# ---------------------------------------------------------------------------
class TestBasicCorrectness:

    def test_forward_output_shape(self):
        model = make_fla_model()
        input_ids = torch.randint(0, 512, (2, 64), device="cuda")
        output = model(input_ids)
        assert output.logits.shape == (2, 64, 512)

    def test_forward_backward(self):
        model = make_fla_model()
        model.train()
        input_ids = torch.randint(0, 512, (2, 64), device="cuda")
        labels = torch.randint(0, 512, (2, 64), device="cuda")
        output = model(input_ids, labels=labels)
        assert output.loss is not None
        output.loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_use_cache_returns_states(self):
        """use_cache=True should populate past_key_values."""
        model = make_fla_model()
        model.eval()
        input_ids = torch.randint(0, 512, (1, 32), device="cuda")
        with torch.no_grad():
            output = model(input_ids, use_cache=True)
        assert output.past_key_values is not None

    def test_attention_mask(self):
        """attention_mask should not crash and should affect output."""
        model = make_fla_model()
        model.eval()
        input_ids = torch.randint(0, 512, (2, 32), device="cuda")
        mask = torch.ones(2, 32, device="cuda")
        mask[1, 16:] = 0  # mask second half of second sequence
        with torch.no_grad():
            out_masked = model(input_ids, attention_mask=mask)
            out_plain = model(input_ids)
        # Outputs should differ due to masking
        assert not torch.allclose(out_masked.logits, out_plain.logits)


# ---------------------------------------------------------------------------
# Test 2: Numerical correctness vs native
# ---------------------------------------------------------------------------
class TestNumericalCorrectness:

    @pytest.mark.skipif(not _has_native_mamba3(), reason="native mamba_ssm not available")
    def test_fla_vs_native_logits_close(self):
        fla_model = make_fla_model()
        native_model = make_native_model()
        copy_weights_native_to_fla(native_model, fla_model)

        fla_model.eval()
        native_model.eval()

        torch.manual_seed(42)
        input_ids = torch.randint(0, 512, (2, 64), device="cuda")

        with torch.no_grad():
            fla_logits = fla_model(input_ids).logits
            native_logits = native_model(input_ids).logits

        rel_error = (fla_logits.float() - native_logits.float()).abs().mean() / native_logits.float().abs().mean()
        print(f"Relative error FLA vs native: {rel_error:.6f}")
        assert rel_error < 0.1, f"Relative error too high: {rel_error}"


# ---------------------------------------------------------------------------
# Test 3: Speed benchmark
# ---------------------------------------------------------------------------
class TestSpeed:

    def test_forward_speed(self):
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

        for _ in range(3):
            loss = model(input_ids, labels=labels).loss
            loss.backward()
            model.zero_grad()
        torch.cuda.synchronize()

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
# Test 4: 1k-step training loss comparison
# ---------------------------------------------------------------------------
class TestTrainingLoss:

    @pytest.mark.slow
    @pytest.mark.skipif(not _has_native_mamba3(), reason="native mamba_ssm not available")
    def test_1k_steps_loss_close(self):
        torch.manual_seed(42)
        d_model, n_layer, vocab_size = 256, 4, 512
        seq_len, batch_size, max_steps = 128, 4, 1000
        lr = 6e-4

        fla_model = make_fla_model(d_model, n_layer, vocab_size)
        native_model = make_native_model(d_model, n_layer, vocab_size)
        copy_weights_native_to_fla(native_model, fla_model)

        fla_model.train()
        native_model.train()

        fla_opt = torch.optim.AdamW(fla_model.parameters(), lr=lr)
        native_opt = torch.optim.AdamW(native_model.parameters(), lr=lr)

        torch.manual_seed(123)
        all_data = torch.randint(0, vocab_size, (max_steps, batch_size, seq_len + 1), device="cuda")

        fla_losses, native_losses = [], []
        for step in range(max_steps):
            data = all_data[step]
            input_ids = data[:, :-1]
            labels = data[:, 1:]

            fla_out = fla_model(input_ids)
            fla_loss = F.cross_entropy(fla_out.logits.float().reshape(-1, vocab_size), labels.reshape(-1))
            fla_loss.backward()
            torch.nn.utils.clip_grad_norm_(fla_model.parameters(), 1.0)
            fla_opt.step()
            fla_opt.zero_grad()
            fla_losses.append(fla_loss.item())

            native_out = native_model(input_ids)
            native_loss = F.cross_entropy(native_out.logits.float().reshape(-1, vocab_size), labels.reshape(-1))
            native_loss.backward()
            torch.nn.utils.clip_grad_norm_(native_model.parameters(), 1.0)
            native_opt.step()
            native_opt.zero_grad()
            native_losses.append(native_loss.item())

            if step % 100 == 0:
                print(f"step {step}: fla={fla_losses[-1]:.4f} native={native_losses[-1]:.4f}")

        fla_final = sum(fla_losses[-50:]) / 50
        native_final = sum(native_losses[-50:]) / 50
        diff = abs(fla_final - native_final)
        print(f"\nFinal avg-50 loss: FLA={fla_final:.4f}, Native={native_final:.4f}, diff={diff:.4f}")
        assert diff < 0.1, f"Loss difference too large: {diff:.4f}"


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Shape test ===")
    TestBasicCorrectness().test_forward_output_shape()
    print("PASS")

    print("\n=== Forward+backward test ===")
    TestBasicCorrectness().test_forward_backward()
    print("PASS")

    print("\n=== Cache test ===")
    TestBasicCorrectness().test_use_cache_returns_states()
    print("PASS")

    print("\n=== Attention mask test ===")
    TestBasicCorrectness().test_attention_mask()
    print("PASS")

    if _has_native_mamba3():
        print("\n=== Numerical correctness vs native ===")
        TestNumericalCorrectness().test_fla_vs_native_logits_close()
        print("PASS")

    print("\n=== Speed test (forward) ===")
    TestSpeed().test_forward_speed()

    print("\n=== Speed test (fwd+bwd) ===")
    TestSpeed().test_forward_backward_speed()
