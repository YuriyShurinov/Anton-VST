"""Generate dummy ONNX models for testing.
Each model takes magnitude input and outputs two masks (noise, reverb).
Input: [1, num_frames, num_bins] float32
Output noise_mask: [1, num_bins] float32
Output reverb_mask: [1, num_bins] float32
The dummy model simply outputs 0.5 for all bins (identity-ish).
"""
import numpy as np
import os

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("pip install onnx numpy")
    exit(1)

FFT_SIZES = [128, 256, 512, 1024, 2048]
NUM_FRAMES = 4  # temporal context

os.makedirs("models", exist_ok=True)

for fft_size in FFT_SIZES:
    num_bins = fft_size // 2 + 1

    # Input
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, NUM_FRAMES, num_bins])

    # Output nodes -- just return constants
    noise_out = helper.make_tensor_value_info("noise_mask", TensorProto.FLOAT, [1, num_bins])
    reverb_out = helper.make_tensor_value_info("reverb_mask", TensorProto.FLOAT, [1, num_bins])

    # Identity-like: take mean over frames axis, then sigmoid-ish clamp
    reduce_node = helper.make_node("ReduceMean", ["input"], ["reduced"], axes=[1], keepdims=1)
    reshape_node = helper.make_node("Reshape", ["reduced", "shape"], ["reshaped"])
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [1, num_bins])
    sigmoid_noise = helper.make_node("Sigmoid", ["reshaped"], ["noise_mask"])
    sigmoid_reverb = helper.make_node("Sigmoid", ["reshaped"], ["reverb_mask"])

    graph = helper.make_graph(
        [reduce_node, reshape_node, sigmoid_noise, sigmoid_reverb],
        f"denoiser_{fft_size}",
        [X],
        [noise_out, reverb_out],
        [shape_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = f"models/denoiser_{fft_size}.onnx"
    onnx.save(model, path)
    print(f"Saved {path} (bins={num_bins})")
