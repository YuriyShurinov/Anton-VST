#!/usr/bin/env python3
"""
Create an NSNet2-architecture ONNX model for DeFeedback Pro.

Architecture: 3x GRU(hidden=400) + FC(161) + Sigmoid
Input:  [batch, frames, 161] - log-power spectrum (161 = 320/2 + 1 bins at 16kHz)
Output: [batch, frames, 161] - suppression gain mask (0..1)

The model has randomly initialized weights. For production use, train on
the Microsoft DNS Challenge dataset or use pre-trained NSNet2 weights.
"""

import os
import numpy as np

def create_nsnet2_model(output_path: str, hidden_size: int = 400, input_size: int = 161):
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        print("ERROR: 'onnx' package required. Install with: pip install onnx")
        return False

    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ['batch', 'frames', input_size])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ['batch', 'frames', input_size])

    rng = np.random.RandomState(42)  # Reproducible weights

    def make_gru_weights(name, in_sz, h_sz):
        scale_W = np.sqrt(2.0 / (in_sz + h_sz))
        scale_R = np.sqrt(2.0 / (h_sz + h_sz))
        W = numpy_helper.from_array(
            rng.randn(1, 3*h_sz, in_sz).astype(np.float32) * np.float32(scale_W), f'{name}_W')
        R = numpy_helper.from_array(
            rng.randn(1, 3*h_sz, h_sz).astype(np.float32) * np.float32(scale_R), f'{name}_R')
        B = numpy_helper.from_array(
            np.zeros((1, 6*h_sz), dtype=np.float32), f'{name}_B')
        return W, R, B

    gru1_W, gru1_R, gru1_B = make_gru_weights('gru1', input_size, hidden_size)
    gru2_W, gru2_R, gru2_B = make_gru_weights('gru2', hidden_size, hidden_size)
    gru3_W, gru3_R, gru3_B = make_gru_weights('gru3', hidden_size, hidden_size)

    # FC layer: [hidden_size, input_size] transposed for MatMul
    scale_fc = np.float32(np.sqrt(2.0 / (hidden_size + input_size)))
    fc_W_T = numpy_helper.from_array(
        rng.randn(hidden_size, input_size).astype(np.float32) * scale_fc, 'fc_W_T')
    fc_B = numpy_helper.from_array(
        np.zeros(input_size, dtype=np.float32), 'fc_B')

    # Squeeze axes constant (opset 13+)
    axes_1 = numpy_helper.from_array(np.array([1], dtype=np.int64), 'axes_1')

    nodes = [
        helper.make_node('GRU', ['input', 'gru1_W', 'gru1_R', 'gru1_B'],
                         ['gru1_Y', ''], hidden_size=hidden_size),
        helper.make_node('Squeeze', ['gru1_Y', 'axes_1'], ['gru1_out']),
        helper.make_node('GRU', ['gru1_out', 'gru2_W', 'gru2_R', 'gru2_B'],
                         ['gru2_Y', ''], hidden_size=hidden_size),
        helper.make_node('Squeeze', ['gru2_Y', 'axes_1'], ['gru2_out']),
        helper.make_node('GRU', ['gru2_out', 'gru3_W', 'gru3_R', 'gru3_B'],
                         ['gru3_Y', ''], hidden_size=hidden_size),
        helper.make_node('Squeeze', ['gru3_Y', 'axes_1'], ['gru3_out']),
        helper.make_node('MatMul', ['gru3_out', 'fc_W_T'], ['fc_raw']),
        helper.make_node('Add', ['fc_raw', 'fc_B'], ['fc_biased']),
        helper.make_node('Sigmoid', ['fc_biased'], ['output']),
    ]

    inits = [gru1_W, gru1_R, gru1_B, gru2_W, gru2_R, gru2_B,
             gru3_W, gru3_R, gru3_B, fc_W_T, fc_B, axes_1]

    graph = helper.make_graph(nodes, 'nsnet2', [X], [Y], initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 15)])
    model.ir_version = 8
    model.doc_string = "NSNet2-architecture noise suppression model (random weights)"

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    onnx.save(model, output_path)
    print(f"Saved NSNet2 model to {output_path}")
    print(f"  Input:  [batch, frames, {input_size}] (log-power spectrum)")
    print(f"  Output: [batch, frames, {input_size}] (suppression mask)")
    print(f"  Size:   {os.path.getsize(output_path) / 1024:.1f} KB")

    # Verify
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        test_in = np.random.randn(1, 1, input_size).astype(np.float32)
        result = sess.run(None, {'input': test_in})
        print(f"  Verified: output shape {result[0].shape}, range [{result[0].min():.4f}, {result[0].max():.4f}]")
    except ImportError:
        print("  (onnxruntime not installed, skipping verification)")

    return True


if __name__ == '__main__':
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else 'models/nsnet2.onnx'
    create_nsnet2_model(output)
