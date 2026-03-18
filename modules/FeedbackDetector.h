#pragma once
#include <vector>

class FeedbackDetector
{
public:
    FeedbackDetector(int numBins, float sampleRate, int fftSize);
    void process(const float* magnitude, float* mask);
    int getPersistenceFrames() const { return persistenceFrames_; }
    void setFeedbackAmount(float amount) { feedbackAmount_ = amount; }

private:
    float computeMedian(const float* data, int size) const;
    void applyGaussianNotch(float* mask, int centerBin, float depth) const;

    int numBins_;
    float sampleRate_;
    int fftSize_;
    float binResolution_;
    float feedbackAmount_ = 1.0f;
    int persistenceFrames_;
    std::vector<int> peakCounter_;
    std::vector<float> envelope_;
    float attackCoeff_;
    float releaseCoeff_;
    int notchWidthBins_;
    float peakThresholdLinear_;
    mutable std::vector<float> medianScratch_;
};
