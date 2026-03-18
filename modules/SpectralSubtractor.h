#pragma once
#include <vector>

class SpectralSubtractor
{
public:
    SpectralSubtractor(int numBins, float sampleRate, int fftSize);
    void process(const float* magnitude, float* mask);
    void setDenoiseAmount(float amount) { denoiseAmount_ = amount; }

private:
    int numBins_;
    float sampleRate_;
    int fftSize_;
    float binResolution_;
    float denoiseAmount_ = 0.5f;
    int smoothBins_;
    std::vector<float> noiseFloor_;
    std::vector<float> minBuffer_;
    int minWindowFrames_;
    int frameCounter_ = 0;
    void updateNoiseFloor(const float* magnitude);
    void smoothMask(float* mask) const;
    mutable std::vector<float> smoothScratch_;
};
