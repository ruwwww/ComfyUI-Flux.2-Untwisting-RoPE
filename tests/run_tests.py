import os
import sys
import traceback

# Add repository root and ComfyUI root to python path so imports resolve correctly
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

comfy_root = os.path.dirname(os.path.dirname(repo_root))
if comfy_root not in sys.path:
    sys.path.insert(0, comfy_root)

# Mock comfy_aimdo to allow testing without platform-specific binary dependencies
from unittest.mock import MagicMock
comfy_aimdo_mock = MagicMock()
comfy_aimdo_mock.__path__ = []
sys.modules['comfy_aimdo'] = comfy_aimdo_mock
sys.modules['comfy_aimdo.host_buffer'] = MagicMock()
sys.modules['comfy_aimdo.torch'] = MagicMock()
sys.modules['comfy_aimdo.model_vbar'] = MagicMock()

# Add tests directory to python path
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)


if __name__ == "__main__":
    tests_passed = 0
    tests_failed = 0

    try:
        from test_frequency_scale import (
            test_frequency_scale_vector_endpoints_for_two_spatial_axes,
            test_three_axis_first_axis_uses_low_scale,
            test_reference_ranges_are_tail_of_image_slice,
        )
        from test_anima_rope import (
            test_anima_axes_dim_from_head_dim,
            test_anima_rope_frequency_scale_vector,
            test_looks_like_anima,
            test_anima_untwist_self_attn_patch,
        )

        test_funcs = [
            test_frequency_scale_vector_endpoints_for_two_spatial_axes,
            test_three_axis_first_axis_uses_low_scale,
            test_reference_ranges_are_tail_of_image_slice,
            test_anima_axes_dim_from_head_dim,
            test_anima_rope_frequency_scale_vector,
            test_looks_like_anima,
            test_anima_untwist_self_attn_patch,
        ]

        print("Running tests...")
        for func in test_funcs:
            try:
                print(f"Running {func.__name__}...", end="")
                func()
                print(" PASSED")
                tests_passed += 1
            except Exception as e:
                print(" FAILED")
                traceback.print_exc()
                tests_failed += 1

    except Exception as e:
        print("Failed to import or run tests:")
        traceback.print_exc()
        sys.exit(1)

    print(f"\nTest Summary: {tests_passed} passed, {tests_failed} failed.")
    if tests_failed > 0:
        sys.exit(1)
