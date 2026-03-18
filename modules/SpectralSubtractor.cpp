#include "SpectralSubtractor.h"
#include <algorithm>
#include <cmath>

SpectralSubtractor::SpectralSubtractor(int numBins, float sampleRate, int fftSize)
    : numBins_(numBins), sampleRate_(sampleRate), fftSize_(fftSize),
      binResolution_(sampleRate / fftSize)
{
    noiseFloor_.resize(numBins, 0.0f);
    minBuffer_.resize(numBins, 1e30f);
    float hopDuration = (fftSize / 4.0f) / sampleRate;
    minWindowFrames_ = static_cast<int>(std::ceil(1.5f / hopDuration));
    smoothBins_ = std::max(1, static_cast<int>(std::round(200.0f / binResolution_)));
    smoothScratch_.resize(numBins);
}

void SpectralSubtractor::updateNoiseFloor(const float* magnitude)
{
    for (int i = 0; i < numBins_; ++i)
        minBuffer_[i] = std::min(minBuffer_[i], magnitude[i]);
    frameCounter_++;
    if (frameCounter_ >= minWindowFrames_)
    {
        for (int i = 0; i < numBins_; ++i)
        {
            noiseFloor_[i] = minBuffer_[i];
            minBuffer_[i] = magnitude[i];
        }
        frameCounter_ = 0;
    }
}

void SpectralSubtractor::smoothMask(float* mask) const
{
    if (smoothBins_ <= 1) return;
    int halfWidth = smoothBins_ / 2;
    for (int i = 0; i < numBins_; ++i)
    {
        float sum = 0.0f;
        int count = 0;
        for (int j = i - halfWidth; j <= i + halfWidth; ++j)
        {
            if (j >= 0 && j < numBins_) { sum += mask[j]; count++; }
        }
        smoothScratch_[i] = sum / count;
    }
    std::copy(smoothScratch_.begin(), smoothScratch_.begin() + numBins_, mask);
}

void SpectralSubtractor::process(const float* magnitude, float* mask)
{
    updateNoiseFloor(magnitude);
    float alpha = 1.0f + 3.0f * denoiseAmount_;
    for (int i = 0; i < numBins_; ++i)
    {
        if (magnitude[i] < 1e-10f) { mask[i] = 0.0f; continue; }
        mask[i] = std::max(0.0f, 1.0f - alpha * noiseFloor_[i] / magnitude[i]);
    }
    smoothMask(mask);
}
