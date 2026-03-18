#pragma once
#include <vector>
#include <memory>
#include <functional>
#include "pffft.h"

class SpectralEngine
{
public:
    SpectralEngine(int fftSize, float sampleRate);
    ~SpectralEngine();

    using MaskCallback = std::function<void(const float* magnitude, float* mask)>;

    void processBlock(const float* inputSamples, float* outputSamples,
                      int numSamples, MaskCallback maskCallback);

    int getFFTSize() const { return fftSize_; }
    int getHopSize() const { return hopSize_; }
    int getNumBins() const { return numBins_; }
    float getBinResolution() const { return sampleRate_ / fftSize_; }
    const float* getMagnitudeSpectrum() const { return magnitude_.data(); }
    void reset();

private:
    void computeWindow();
    void forwardFFT(const float* windowed);
    void applyMask(const float* mask);
    void inverseFFT();

    int fftSize_;
    int hopSize_;
    int numBins_;
    float sampleRate_;
    float colaNorm_ = 1.0f; // COLA normalization for Hann^2 overlap-add

    PFFFT_Setup* pffftSetup_ = nullptr;
    float* pffftWork_ = nullptr;

    std::vector<float> window_;
    std::vector<float> windowedInput_;
    std::vector<float> complexSpectrum_;
    std::vector<float> unpacked_;
    std::vector<float> magnitude_;
    std::vector<float> ifftOut_;
    std::vector<float> overlapBuffer_;
    std::vector<float> outputRing_;
    std::vector<float> inputRing_;
    int inputWritePos_ = 0;
    int outputReadPos_ = 0;
    int hopCounter_ = 0;
    std::vector<float> currentMask_;
};
