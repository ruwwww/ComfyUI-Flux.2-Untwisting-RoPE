import torch
from unittest.mock import MagicMock
from flux_untwist.config import anima_axes_dim_from_head_dim
from flux_untwist.patches import build_frequency_scale_vector, anima_untwist_self_attn_patch
from flux_untwist.utils import _looks_like_anima


def test_anima_axes_dim_from_head_dim():
    # standard head_dim = 128
    axes = anima_axes_dim_from_head_dim(128)
    assert axes == (44, 42, 42)
    assert sum(axes) == 128

    # smaller head_dim = 64
    axes_small = anima_axes_dim_from_head_dim(64)
    assert axes_small == (24, 20, 20)
    assert sum(axes_small) == 64


def test_anima_rope_frequency_scale_vector():
    # With head_dim = 128, axes_dim = (44, 42, 42)
    # The first axis (temporal, dim=44) should receive low_scale
    # The second and third axes (spatial, dim=42 each) should be scaled with interpolation
    v = build_frequency_scale_vector(
        head_dim=128,
        axes_dim=(44, 42, 42),
        high_scale=0.25,
        low_scale=1.5,
        beta=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert v.shape == (128,)
    # first 44 elements are temporal, should be all low_scale (1.5)
    assert torch.allclose(v[0:44], torch.full((44,), 1.5))
    # spatial axes should start at high_scale (0.25) and go to low_scale (1.5)
    assert torch.allclose(v[44:46], torch.tensor([0.25, 0.25]))
    assert torch.allclose(v[84:86], torch.tensor([1.5, 1.5]))
    assert torch.allclose(v[86:88], torch.tensor([0.25, 0.25]))
    assert torch.allclose(v[126:128], torch.tensor([1.5, 1.5]))


def test_looks_like_anima():
    mock_model = MagicMock()
    mock_model.blocks = [1, 2, 3]
    mock_model.x_embedder = MagicMock()
    mock_model.pos_embedder = MagicMock()
    mock_model.t_embedder = MagicMock()
    mock_model.params = MagicMock()

    assert _looks_like_anima(mock_model) is True

    # Missing blocks
    mock_model_bad = MagicMock()
    del mock_model_bad.blocks
    assert _looks_like_anima(mock_model_bad) is False


def test_anima_untwist_self_attn_patch():
    # Setup mock attention
    mock_self = MagicMock()
    mock_self.is_selfattn = True
    mock_self.head_dim = 128
    mock_self.n_heads = 16
    mock_self._block_index = 5
    mock_self.k_proj = MagicMock(side_effect=lambda x: torch.ones(x.shape[0], x.shape[1], 16 * 128))
    mock_self.v_proj = MagicMock(side_effect=lambda x: torch.ones(x.shape[0], x.shape[1], 16 * 128))
    mock_self.q_proj = MagicMock(side_effect=lambda x: torch.ones(x.shape[0], x.shape[1], 16 * 128))
    mock_self.k_norm = MagicMock(side_effect=lambda x: x)
    mock_self.v_norm = MagicMock(side_effect=lambda x: x)
    mock_self.q_norm = MagicMock(side_effect=lambda x: x)

    # Mock original compute_qkv
    q = torch.ones(1, 100, 16, 128)
    k = torch.ones(1, 100, 16, 128)
    v = torch.ones(1, 100, 16, 128)
    original_compute_qkv = MagicMock(return_value=(q, k, v))

    # Mock stack frame to inject transformer_options
    import sys
    transformer_options = {
        "anima_untwist_rope": {
            "enabled": True,
            "ref_ranges": [(60, 100)],
            "target_range": (0, 60),
            "high_scale": 0.25,
            "low_scale": 1.5,
            "beta": 2.0,
            "axes_dim": (44, 42, 42),
            "start_block": 0,
            "end_block": 10,
            "qk_adain_strength": 0.0,
        }
    }

    # Helper function to run call where transformer_options is in locals
    def run_patched_call():
        return anima_untwist_self_attn_patch(
            original_compute_qkv, mock_self, x=torch.ones(1, 100, 2048), context=None, rope_emb=None
        )

    q_out, k_out, v_out = run_patched_call()

    # Original combined sequence was 100 tokens (60 target + 40 reference).
    assert k_out.shape == (1, 100, 16, 128)
    # target range [0, 60] should be unchanged (all 1s)
    assert torch.allclose(k_out[:, :60, :, :], torch.ones(1, 60, 16, 128))
    # ref range [60:100] should be scaled
    scale_vec = build_frequency_scale_vector(
        128, (44, 42, 42), 0.25, 1.5, 2.0, torch.device("cpu"), torch.float32
    ).view(1, 1, 1, -1)
    expected_ref = torch.ones(1, 40, 16, 128) * scale_vec
    assert torch.allclose(k_out[:, 60:100, :, :], expected_ref)

