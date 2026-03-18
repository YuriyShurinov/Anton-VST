#pragma once
#include <vector>
#include <atomic>
#include <memory>
#include <thread>
#include <string>
#include "RingBuffer.h"

// Forward declare ONNX types to avoid header dependency
namespace Ort { class Env; class Session; class SessionOptions; class MemoryInfo; }

class MLProcessor
{
public:
    MLProcessor(int numBins, const std::string& modelDir, int fftSize);
    ~MLProcessor();

    bool isModelLoaded() const { return modelLoaded_.load(); }

    // Called from audio thread: submit a magnitude frame for async inference
    void submitFrame(const float* magnitude);

    // Called from audio thread: get the most recent inference result
    // Returns true if new masks are available
    bool getLatestMasks(float* noiseMask, float* reverbMask);

    void setDenoiseAmount(float amount) { denoiseAmount_ = amount; }
    void setDereverbAmount(float amount) { dereverbAmount_ = amount; }

    // Compute combined mask2 from noise and reverb masks
    void computeMask2(float* mask2);

    // Change FFT size (reloads model)
    void setFFTSize(int fftSize);

private:
    void inferenceThreadFunc();
    bool loadModel(const std::string& path);
    void runInference(const float* input, int numFrames);

    int numBins_;
    int numFrames_ = 4; // temporal context
    std::string modelDir_;
    int fftSize_;

    std::atomic<float> denoiseAmount_{0.5f};
    std::atomic<float> dereverbAmount_{0.5f};

    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::atomic<bool> modelLoaded_{false};

    // Frame history for temporal context
    std::vector<float> frameHistory_; // numFrames_ * numBins_

    // Async inference thread
    std::thread inferenceThread_;
    std::atomic<bool> running_{false};

    // SPSC communication: audio thread -> inference thread
    RingBuffer<float> inputQueue_;   // submits magnitude frames (numBins_ per frame)

    // Triple-buffered inference results
    std::vector<float> noiseMasks_[3];
    std::vector<float> reverbMasks_[3];
    std::atomic<int> latestMaskBuffer_{-1}; // -1 = none, 0/1/2 = which buffer is latest
    int writingBuffer_ = 0; // only accessed by inference thread

    // Pre-allocated temp buffers for computeMask2 (avoid audio-thread allocation)
    std::vector<float> tempNoiseMask_;
    std::vector<float> tempReverbMask_;
};
