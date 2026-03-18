#pragma once

// MLProcessor: ONNX Runtime inference for ML-based spectral masking.
// Full implementation added in a later task.
class MLProcessor
{
public:
    MLProcessor() = default;
    ~MLProcessor() = default;

    bool loadModel(const char* modelPath);
    void reset();
};
