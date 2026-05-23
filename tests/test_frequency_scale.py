import torch

from flux_untwist.patches import build_frequency_scale_vector, _reference_ranges_from_options


def test_frequency_scale_vector_endpoints_for_two_spatial_axes():
    v = build_frequency_scale_vector(
        head_dim=8,
        axes_dim=[4, 4],
        high_scale=0.25,
        low_scale=1.5,
        beta=2.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert v.shape == (8,)
    assert torch.allclose(v[0:2], torch.tensor([0.25, 0.25]))
    assert torch.allclose(v[2:4], torch.tensor([1.5, 1.5]))
    assert torch.allclose(v[4:6], torch.tensor([0.25, 0.25]))
    assert torch.allclose(v[6:8], torch.tensor([1.5, 1.5]))


def test_three_axis_first_axis_uses_low_scale():
    v = build_frequency_scale_vector(12, [4, 4, 4], 0.2, 1.4, 2.0, torch.device("cpu"), torch.float32)
    assert torch.allclose(v[0:4], torch.full((4,), 1.4))


def test_reference_ranges_are_tail_of_image_slice():
    target_range, refs = _reference_ranges_from_options(
        180,
        {
            "img_slice": [64, 180],
            "reference_image_num_tokens": [16, 20],
        },
    )
    assert target_range == (64, 144)
    assert refs == [(144, 160), (160, 180)]
