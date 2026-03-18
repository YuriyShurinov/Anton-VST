#pragma once

// SpectralEngine: STFT-based spectral analysis and resynthesis engine.
// Full implementation added in a later task.
class SpectralEngine
{
public:
    SpectralEngine() = default;
    ~SpectralEngine() = default;

    void prepare(double sampleRate, int samplesPerBlock, int fftSize);
    void reset();
};
