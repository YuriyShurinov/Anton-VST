#pragma once

// SpectralSubtractor: Applies spectral subtraction to suppress feedback/noise.
// Full implementation added in a later task.
class SpectralSubtractor
{
public:
    SpectralSubtractor() = default;
    ~SpectralSubtractor() = default;

    void prepare(double sampleRate, int fftSize);
    void reset();
};
