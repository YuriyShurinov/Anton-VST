#pragma once

// FeedbackDetector: Detects feedback frequencies in the spectral domain.
// Full implementation added in a later task.
class FeedbackDetector
{
public:
    FeedbackDetector() = default;
    ~FeedbackDetector() = default;

    void prepare(double sampleRate, int fftSize);
    void reset();
};
