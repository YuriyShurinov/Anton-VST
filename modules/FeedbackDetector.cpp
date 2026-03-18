#include "FeedbackDetector.h"
#include <algorithm>
#include <cmath>
#include <numeric>

FeedbackDetector::FeedbackDetector(int numBins, float sampleRate, int fftSize)
    : numBins_(numBins), sampleRate_(sampleRate), fftSize_(fftSize),
      binResolution_(sampleRate / fftSize)
{
    float hopDuration = (fftSize / 4.0f) / sampleRate;
    persistenceFrames_ = static_cast<int>(std::ceil(0.050f / hopDuration));
    peakCounter_.resize(numBins, 0);
    envelope_.resize(numBins, 0.0f);
    attackCoeff_ = 1.0f / 3.0f;
    releaseCoeff_ = 1.0f / 100.0f;
    notchWidthBins_ = std::max(1, static_cast<int>(std::ceil(100.0f / binResolution_)));
    peakThresholdLinear_ = std::pow(10.0f, 12.0f / 20.0f);
    medianScratch_.resize(numBins);
}

float FeedbackDetector::computeMedian(const float* data, int size) const
{
    std::copy(data, data + size, medianScratch_.begin());
    auto mid = medianScratch_.begin() + size / 2;
    std::nth_element(medianScratch_.begin(), mid, medianScratch_.begin() + size);
    return *mid;
}

void FeedbackDetector::applyGaussianNotch(float* mask, int centerBin, float depth) const
{
    float sigma = notchWidthBins_ / 2.0f;
    int halfWidth = notchWidthBins_;
    int lo = std::max(0, centerBin - halfWidth);
    int hi = std::min(numBins_ - 1, centerBin + halfWidth);
    for (int i = lo; i <= hi; ++i)
    {
        float dist = static_cast<float>(i - centerBin);
        float gauss = std::exp(-(dist * dist) / (2.0f * sigma * sigma));
        float suppression = 1.0f - depth * gauss;
        mask[i] = std::min(mask[i], suppression);
    }
}

void FeedbackDetector::process(const float* magnitude, float* mask)
{
    std::fill(mask, mask + numBins_, 1.0f);
    float median = computeMedian(magnitude, numBins_);
    float threshold = median * peakThresholdLinear_;
    for (int i = 0; i < numBins_; ++i)
    {
        if (magnitude[i] > threshold)
            peakCounter_[i]++;
        else
            peakCounter_[i] = std::max(0, peakCounter_[i] - 1);
    }
    for (int i = 0; i < numBins_; ++i)
    {
        float target = (peakCounter_[i] >= persistenceFrames_) ? feedbackAmount_ : 0.0f;
        if (target > envelope_[i])
            envelope_[i] += attackCoeff_ * (target - envelope_[i]);
        else
            envelope_[i] += releaseCoeff_ * (target - envelope_[i]);
        if (envelope_[i] > 0.01f)
            applyGaussianNotch(mask, i, envelope_[i]);
    }
}
