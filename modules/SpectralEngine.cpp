#define _USE_MATH_DEFINES
#include "SpectralEngine.h"
#include <cmath>
#include <cstring>
#include <algorithm>

SpectralEngine::SpectralEngine(int fftSize, float sampleRate)
    : fftSize_(fftSize), hopSize_(fftSize / 4), numBins_(fftSize / 2 + 1), sampleRate_(sampleRate)
{
    pffftSetup_ = pffft_new_setup(fftSize, PFFFT_REAL);
    pffftWork_ = (float*)pffft_aligned_malloc(fftSize * sizeof(float));

    window_.resize(fftSize);
    windowedInput_.resize(fftSize);
    complexSpectrum_.resize(fftSize);
    unpacked_.resize(numBins_ * 2);
    magnitude_.resize(numBins_);
    ifftOut_.resize(fftSize);
    currentMask_.resize(numBins_);
    overlapBuffer_.resize(fftSize + hopSize_, 0.0f);
    outputRing_.resize(fftSize * 2, 0.0f);
    inputRing_.resize(fftSize, 0.0f);
    inputWritePos_ = 0;
    outputReadPos_ = 0;
    hopCounter_ = 0;
    computeWindow();
}

SpectralEngine::~SpectralEngine()
{
    if (pffftSetup_) pffft_destroy_setup(pffftSetup_);
    if (pffftWork_) pffft_aligned_free(pffftWork_);
}

void SpectralEngine::computeWindow()
{
    for (int i = 0; i < fftSize_; ++i)
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / fftSize_));

    // Compute COLA normalization: sum of w[n-mR]^2 across overlapping frames
    // For Hann window with 75% overlap (hop=N/4), this is 1.5
    colaNorm_ = 0.0f;
    int numOverlaps = fftSize_ / hopSize_;
    for (int m = 0; m < numOverlaps; ++m)
    {
        int sampleIdx = m * hopSize_;
        if (sampleIdx < fftSize_)
            colaNorm_ += window_[sampleIdx] * window_[sampleIdx];
    }
}

void SpectralEngine::forwardFFT(const float* windowed)
{
    pffft_transform_ordered(pffftSetup_, windowed, complexSpectrum_.data(), pffftWork_, PFFFT_FORWARD);
    unpacked_[0] = complexSpectrum_[0];
    unpacked_[1] = 0.0f;
    for (int i = 1; i < numBins_ - 1; ++i)
    {
        unpacked_[i * 2]     = complexSpectrum_[i * 2];
        unpacked_[i * 2 + 1] = complexSpectrum_[i * 2 + 1];
    }
    unpacked_[(numBins_ - 1) * 2]     = complexSpectrum_[1];
    unpacked_[(numBins_ - 1) * 2 + 1] = 0.0f;
    for (int i = 0; i < numBins_; ++i)
    {
        float re = unpacked_[i * 2];
        float im = unpacked_[i * 2 + 1];
        magnitude_[i] = std::sqrt(re * re + im * im);
    }
}

void SpectralEngine::applyMask(const float* mask)
{
    for (int i = 0; i < numBins_; ++i)
    {
        unpacked_[i * 2]     *= mask[i];
        unpacked_[i * 2 + 1] *= mask[i];
    }
    complexSpectrum_[0] = unpacked_[0];
    complexSpectrum_[1] = unpacked_[(numBins_ - 1) * 2];
    for (int i = 1; i < numBins_ - 1; ++i)
    {
        complexSpectrum_[i * 2]     = unpacked_[i * 2];
        complexSpectrum_[i * 2 + 1] = unpacked_[i * 2 + 1];
    }
}

void SpectralEngine::inverseFFT()
{
    pffft_transform_ordered(pffftSetup_, complexSpectrum_.data(), ifftOut_.data(), pffftWork_, PFFFT_BACKWARD);
    float norm = 1.0f / fftSize_;
    for (int i = 0; i < fftSize_; ++i)
        ifftOut_[i] *= norm;
}

void SpectralEngine::processBlock(const float* inputSamples, float* outputSamples,
                                   int numSamples, MaskCallback maskCallback)
{
    for (int s = 0; s < numSamples; ++s)
    {
        inputRing_[inputWritePos_] = inputSamples[s];
        inputWritePos_ = (inputWritePos_ + 1) % fftSize_;
        hopCounter_++;

        if (hopCounter_ >= hopSize_)
        {
            hopCounter_ = 0;
            for (int i = 0; i < fftSize_; ++i)
            {
                int idx = (inputWritePos_ + i) % fftSize_;
                windowedInput_[i] = inputRing_[idx] * window_[i];
            }
            forwardFFT(windowedInput_.data());
            maskCallback(magnitude_.data(), currentMask_.data());
            applyMask(currentMask_.data());
            inverseFFT();
            for (int i = 0; i < fftSize_; ++i)
            {
                int idx = (outputReadPos_ + i) % (int)outputRing_.size();
                outputRing_[idx] += ifftOut_[i] * window_[i] / colaNorm_;
            }
        }

        outputSamples[s] = outputRing_[outputReadPos_];
        outputRing_[outputReadPos_] = 0.0f;
        outputReadPos_ = (outputReadPos_ + 1) % (int)outputRing_.size();
    }
}

void SpectralEngine::reset()
{
    std::fill(overlapBuffer_.begin(), overlapBuffer_.end(), 0.0f);
    std::fill(outputRing_.begin(), outputRing_.end(), 0.0f);
    std::fill(inputRing_.begin(), inputRing_.end(), 0.0f);
    inputWritePos_ = 0;
    outputReadPos_ = 0;
    hopCounter_ = 0;
}
