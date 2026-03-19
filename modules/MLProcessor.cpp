#define _USE_MATH_DEFINES
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

MLProcessor::MLProcessor(int numBins, const std::string& modelDir, int fftSize, float sampleRate)
    : numBins_(numBins),
      modelDir_(modelDir),
      fftSize_(fftSize),
      sampleRate_(sampleRate),
      inputQueue_(numBins * 16) // buffer up to 16 frame submissions
{
    for (int i = 0; i < 3; ++i) {
        noiseMasks_[i].resize(numBins_, 1.0f);
        reverbMasks_[i].resize(numBins_, 1.0f);
    }
    tempNoiseMask_.resize(numBins_, 1.0f);
    tempReverbMask_.resize(numBins_, 1.0f);

    // NSNet2 buffers
    nsnet2Input_.resize(kNSNet2Bins, 0.0f);
    nsnet2Output_.resize(kNSNet2Bins, 1.0f);
    resampledMask_.resize(numBins_, 1.0f);

#ifndef DEFEEDBACK_TEST_MODE
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DeFeedbackPro");
#else
    env_ = std::make_unique<Ort::Env>(2, "DeFeedbackPro");
#endif

    // Load single NSNet2 model (architecture-independent of plugin FFT size)
    std::string modelPath = modelDir + "/nsnet2.onnx";
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

void MLProcessor::resampleSpectrum(const float* src, int srcBins, float* dst, int dstBins)
{
    // Linear interpolation to resample spectral bins
    // Maps frequency bin indices proportionally
    if (srcBins == dstBins) {
        std::memcpy(dst, src, dstBins * sizeof(float));
        return;
    }
    for (int i = 0; i < dstBins; ++i)
    {
        float srcPos = static_cast<float>(i) * (srcBins - 1) / (dstBins - 1);
        int lo = static_cast<int>(srcPos);
        int hi = std::min(lo + 1, srcBins - 1);
        float frac = srcPos - lo;
        dst[i] = src[lo] * (1.0f - frac) + src[hi] * frac;
    }
}

void MLProcessor::submitFrame(const float* magnitude)
{
    // Push raw magnitude frame to inference queue
    inputQueue_.push(magnitude, numBins_);
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
    std::vector<float> inputFrame(numBins_);

    while (running_)
    {
        if (inputQueue_.availableToRead() >= static_cast<size_t>(numBins_))
        {
            // Drain to latest frame (skip old ones)
            while (inputQueue_.availableToRead() >= static_cast<size_t>(numBins_ * 2))
                inputQueue_.pop(inputFrame.data(), numBins_);

            inputQueue_.pop(inputFrame.data(), numBins_);
            runInference(inputFrame.data(), numBins_);
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void MLProcessor::runInference(const float* magnitude, int numBins)
{
    // Step 1: Resample magnitude from plugin bins to NSNet2 bins (161)
    std::vector<float> mag161(kNSNet2Bins);
    resampleSpectrum(magnitude, numBins, mag161.data(), kNSNet2Bins);

    // Step 2: Convert to log-power spectrum (NSNet2 input format)
    for (int i = 0; i < kNSNet2Bins; ++i)
    {
        float power = mag161[i] * mag161[i];
        nsnet2Input_[i] = std::log10(std::max(power, 1e-12f));
    }

#ifndef DEFEEDBACK_TEST_MODE
    if (!session_) return;

    try
    {
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // NSNet2 input: [batch=1, frames=1, 161]
        std::array<int64_t, 3> inputShape = {1, 1, kNSNet2Bins};
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, nsnet2Input_.data(),
            kNSNet2Bins, inputShape.data(), 3);

        const char* inputNames[] = {"input"};
        const char* outputNames[] = {"output"};

        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                      inputNames, &inputTensor, 1,
                                      outputNames, 1);

        const float* maskData = outputs[0].GetTensorData<float>();

        // Step 3: Resample mask from 161 bins back to plugin's numBins_
        resampleSpectrum(maskData, kNSNet2Bins, resampledMask_.data(), numBins_);

        // Step 4: Write to triple buffer
        int latest = latestMaskBuffer_.load(std::memory_order_acquire);
        int writeSlot = 0;
        for (int i = 0; i < 3; ++i) {
            if (i != latest && i != writingBuffer_) { writeSlot = i; break; }
        }

        auto& nDst = noiseMasks_[writeSlot];
        auto& rDst = reverbMasks_[writeSlot];

        // NSNet2 outputs single suppression mask — use for both noise and reverb
        std::memcpy(nDst.data(), resampledMask_.data(), numBins_ * sizeof(float));
        std::memcpy(rDst.data(), resampledMask_.data(), numBins_ * sizeof(float));

        latestMaskBuffer_.store(writeSlot, std::memory_order_release);
        writingBuffer_ = writeSlot;
    }
    catch (...) { /* reuse last mask on error */ }
#else
    // Test mode: generate sigmoid-like dummy masks from input
    int latest = latestMaskBuffer_.load(std::memory_order_acquire);
    int writeSlot = 0;
    for (int i = 0; i < 3; ++i) {
        if (i != latest && i != writingBuffer_) { writeSlot = i; break; }
    }

    auto& nDst = noiseMasks_[writeSlot];
    auto& rDst = reverbMasks_[writeSlot];

    for (int i = 0; i < numBins_; ++i)
    {
        float sig = 1.0f / (1.0f + std::exp(-magnitude[i]));
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

    for (int i = 0; i < 3; ++i) {
        noiseMasks_[i].assign(numBins_, 1.0f);
        reverbMasks_[i].assign(numBins_, 1.0f);
    }
    tempNoiseMask_.assign(numBins_, 1.0f);
    tempReverbMask_.assign(numBins_, 1.0f);
    resampledMask_.assign(numBins_, 1.0f);
    latestMaskBuffer_ = -1;
    inputQueue_.reset();
    inputQueue_.resize(numBins_ * 16);

    // NSNet2 model is FFT-size independent — just reload it
    std::string modelPath = modelDir_ + "/nsnet2.onnx";
    if (loadModel(modelPath))
    {
        running_ = true;
        inferenceThread_ = std::thread(&MLProcessor::inferenceThreadFunc, this);
    }
}
