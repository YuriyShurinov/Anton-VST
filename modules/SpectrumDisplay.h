#pragma once

// SpectrumDisplay: GUI component for real-time spectrum visualization.
// Full implementation added in a later task.
class SpectrumDisplay
{
public:
    SpectrumDisplay() = default;
    ~SpectrumDisplay() = default;

    void prepare(double sampleRate, int fftSize);
    void reset();
};
