#include "MLProcessor.h"
#include <cmath>
#include <cstring>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <algorithm>

// Only include ONNX Runtime when available
#ifndef DEFEEDBACK_TEST_MODE
#include <onnxruntime_cxx_api.h>
#endif

#ifdef DEFEEDBACK_TEST_MODE
// Minimal stubs for testing without ONNX Runtime headers in test build
// The real implementation uses onnxruntime_cxx_api.h
namespace Ort {
    class Env { public: Env(int, const char*) {} };
    class SessionOptions { public: void SetIntraOpNumThreads(int) {} };
    class Session {
    public:
        Session(Env&, const char*, SessionOptions&) { throw std::runtime_error("stub"); }
        Session(Env&, const wchar_t*, SessionOptions&) { throw std::runtime_error("stub"); }
    };
    class MemoryInfo {};
}
#endif

MLProcessor::MLProcessor(int numBins, const std::string& modelDir, int fftSize)
    : numBins_(numBins),
      modelDir_(modelDir),
      fftSize_(fftSize),
      inputQueue_(4 * numBins * 8) // buffer up to 8 full history submissions
{
    frameHistory_.resize(numFrames_ * numBins_, 0.0f);
    for (int i = 0; i < 3; ++i) {
        noiseMasks_[i].resize(numBins_, 1.0f);
        reverbMasks_[i].resize(numBins_, 1.0f);
    }
    tempNoiseMask_.resize(numBins_, 1.0f);
    tempReverbMask_.resize(numBins_, 1.0f);

#ifndef DEFEEDBACK_TEST_MODE
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DeFeedbackPro");
#else
    env_ = std::make_unique<Ort::Env>(2, "DeFeedbackPro");
#endif

    std::string modelPath = modelDir + "/denoiser_" + std::to_string(fftSize) + ".onnx";
    if (loadModel(modelPath))
    {
        running_ = true;
        inferenceThread_ = std::thread(&MLProcessor::inferenceThreadFunc, this);
    }
}

MLProcessor::~MLProcessor()
{
    running_ = false;
    if (inferenceThread_.joinable())
        inferenceThread_.join();
}

bool MLProcessor::loadModel(const std::string& path)
{
    try
    {
#ifndef DEFEEDBACK_TEST_MODE
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        #ifdef _WIN32
        std::wstring wpath(path.begin(), path.end());
        session_ = std::make_unique<Ort::Session>(*env_, wpath.c_str(), opts);
        #else
        session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), opts);
        #endif
        modelLoaded_ = true;
        return true;
#else
        // In test mode, check if file exists
        FILE* f = fopen(path.c_str(), "rb");
        if (f) { fclose(f); modelLoaded_ = true; return true; }
        return false;
#endif
    }
    catch (...)
    {
        modelLoaded_ = false;
        return false;
    }
}

void MLProcessor::submitFrame(const float* magnitude)
{
    // Shift history left and append new frame
    std::memmove(frameHistory_.data(),
                 frameHistory_.data() + numBins_,
                 (numFrames_ - 1) * numBins_ * sizeof(float));
    std::memcpy(frameHistory_.data() + (numFrames_ - 1) * numBins_,
                magnitude, numBins_ * sizeof(float));

    // Push full history to inference queue
    inputQueue_.push(frameHistory_.data(), numFrames_ * numBins_);
}

bool MLProcessor::getLatestMasks(float* noiseMask, float* reverbMask)
{
    int buf = latestMaskBuffer_.load(std::memory_order_acquire);
    if (buf < 0) return false;
    std::memcpy(noiseMask, noiseMasks_[buf].data(), numBins_ * sizeof(float));
    std::memcpy(reverbMask, reverbMasks_[buf].data(), numBins_ * sizeof(float));
    return true;
}

void MLProcessor::computeMask2(float* mask2)
{
    if (!getLatestMasks(tempNoiseMask_.data(), tempReverbMask_.data()))
    {
        std::fill(mask2, mask2 + numBins_, 1.0f);
        return;
    }

    float dn = denoiseAmount_.load();
    float dr = dereverbAmount_.load();

    for (int i = 0; i < numBins_; ++i)
    {
        float mn = std::pow(std::max(tempNoiseMask_[i], 1e-6f), dn);
        float mr = std::pow(std::max(tempReverbMask_[i], 1e-6f), dr);
        mask2[i] = mn * mr;
    }
}

void MLProcessor::inferenceThreadFunc()
{
    std::vector<float> inputData(numFrames_ * numBins_);

    while (running_)
    {
        if (inputQueue_.availableToRead() >= static_cast<size_t>(numFrames_ * numBins_))
        {
            // Drain to latest frame (skip old ones)
            while (inputQueue_.availableToRead() >= static_cast<size_t>(numFrames_ * numBins_ * 2))
                inputQueue_.pop(inputData.data(), numFrames_ * numBins_);

            inputQueue_.pop(inputData.data(), numFrames_ * numBins_);
            runInference(inputData.data(), numFrames_);
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void MLProcessor::runInference(const float* input, int numFrames)
{
#ifndef DEFEEDBACK_TEST_MODE
    if (!session_) return;

    try
    {
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 3> inputShape = {1, numFrames, numBins_};
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, const_cast<float*>(input),
            numFrames * numBins_, inputShape.data(), 3);

        const char* inputNames[] = {"input"};
        const char* outputNames[] = {"noise_mask", "reverb_mask"};

        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                      inputNames, &inputTensor, 1,
                                      outputNames, 2);

        const float* noiseData = outputs[0].GetTensorData<float>();
        const float* reverbData = outputs[1].GetTensorData<float>();

        // Pick a buffer that isn't the current latest (being read) or the last written
        int latest = latestMaskBuffer_.load(std::memory_order_acquire);
        int writeSlot = 0;
        for (int i = 0; i < 3; ++i) {
            if (i != latest && i != writingBuffer_) { writeSlot = i; break; }
        }

        auto& nDst = noiseMasks_[writeSlot];
        auto& rDst = reverbMasks_[writeSlot];

        std::memcpy(nDst.data(), noiseData, numBins_ * sizeof(float));
        std::memcpy(rDst.data(), reverbData, numBins_ * sizeof(float));

        latestMaskBuffer_.store(writeSlot, std::memory_order_release);
        writingBuffer_ = writeSlot;
    }
    catch (...) { /* reuse last mask */ }
#else
    // Test mode: generate sigmoid-like dummy masks
    // Pick a buffer that isn't the current latest (being read) or the last written
    int latest = latestMaskBuffer_.load(std::memory_order_acquire);
    int writeSlot = 0;
    for (int i = 0; i < 3; ++i) {
        if (i != latest && i != writingBuffer_) { writeSlot = i; break; }
    }

    auto& nDst = noiseMasks_[writeSlot];
    auto& rDst = reverbMasks_[writeSlot];

    for (int i = 0; i < numBins_; ++i)
    {
        float avg = 0.0f;
        for (int f = 0; f < numFrames; ++f)
            avg += input[f * numBins_ + i];
        avg /= numFrames;
        float sig = 1.0f / (1.0f + std::exp(-avg));
        nDst[i] = sig;
        rDst[i] = sig;
    }

    latestMaskBuffer_.store(writeSlot, std::memory_order_release);
    writingBuffer_ = writeSlot;
#endif
}

void MLProcessor::setFFTSize(int fftSize)
{
    running_ = false;
    if (inferenceThread_.joinable())
        inferenceThread_.join();

    fftSize_ = fftSize;
    numBins_ = fftSize / 2 + 1;
    frameHistory_.assign(numFrames_ * numBins_, 0.0f);
    for (int i = 0; i < 3; ++i) {
        noiseMasks_[i].assign(numBins_, 1.0f);
        reverbMasks_[i].assign(numBins_, 1.0f);
    }
    tempNoiseMask_.assign(numBins_, 1.0f);
    tempReverbMask_.assign(numBins_, 1.0f);
    latestMaskBuffer_ = -1;
    inputQueue_.reset();

    std::string modelPath = modelDir_ + "/denoiser_" + std::to_string(fftSize) + ".onnx";
    if (loadModel(modelPath))
    {
        running_ = true;
        inferenceThread_ = std::thread(&MLProcessor::inferenceThreadFunc, this);
    }
}
