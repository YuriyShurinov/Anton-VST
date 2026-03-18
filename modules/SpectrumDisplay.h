#pragma once
#include <JuceHeader.h>
#include "RingBuffer.h"
#include <vector>

class SpectrumDisplay : public juce::Component, private juce::Timer
{
public:
    SpectrumDisplay(RingBuffer<float>& spectrumSource, int numBins);
    ~SpectrumDisplay() override;

    void paint(juce::Graphics& g) override;
    void setNumBins(int numBins);

private:
    void timerCallback() override;

    RingBuffer<float>& spectrumSource_;
    int numBins_;
    std::vector<float> displayMagnitude_;
    std::vector<float> smoothedMagnitude_;
    float smoothingCoeff_ = 0.7f;
};
