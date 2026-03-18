#include "SpectrumDisplay.h"
#include <cmath>

SpectrumDisplay::SpectrumDisplay(RingBuffer<float>& spectrumSource, int numBins)
    : spectrumSource_(spectrumSource), numBins_(numBins)
{
    displayMagnitude_.resize(numBins, 0.0f);
    smoothedMagnitude_.resize(numBins, 0.0f);
    startTimerHz(30);
}

SpectrumDisplay::~SpectrumDisplay()
{
    stopTimer();
}

void SpectrumDisplay::setNumBins(int numBins)
{
    numBins_ = numBins;
    displayMagnitude_.resize(numBins, 0.0f);
    smoothedMagnitude_.resize(numBins, 0.0f);
}

void SpectrumDisplay::timerCallback()
{
    std::vector<float> temp(numBins_);
    bool updated = false;
    while (spectrumSource_.availableToRead() >= static_cast<size_t>(numBins_))
    {
        spectrumSource_.pop(temp.data(), numBins_);
        updated = true;
    }

    if (updated)
    {
        for (int i = 0; i < numBins_; ++i)
        {
            float db = 20.0f * std::log10(std::max(temp[i], 1e-6f));
            db = std::max(-80.0f, std::min(0.0f, db));
            float normalized = (db + 80.0f) / 80.0f;

            smoothedMagnitude_[i] = smoothingCoeff_ * smoothedMagnitude_[i]
                                  + (1.0f - smoothingCoeff_) * normalized;
        }
        repaint();
    }
}

void SpectrumDisplay::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat();
    g.fillAll(juce::Colour(0xff1a1a2e));

    if (numBins_ <= 1) return;

    juce::Path spectrumPath;
    float xStep = bounds.getWidth() / (numBins_ - 1);

    spectrumPath.startNewSubPath(0, bounds.getBottom());
    for (int i = 0; i < numBins_; ++i)
    {
        float x = i * xStep;
        float y = bounds.getBottom() - smoothedMagnitude_[i] * bounds.getHeight();
        if (i == 0)
            spectrumPath.startNewSubPath(x, y);
        else
            spectrumPath.lineTo(x, y);
    }

    g.setColour(juce::Colour(0xffe0e0e0));
    g.strokePath(spectrumPath, juce::PathStrokeType(1.5f));

    spectrumPath.lineTo(bounds.getRight(), bounds.getBottom());
    spectrumPath.lineTo(0, bounds.getBottom());
    spectrumPath.closeSubPath();
    g.setColour(juce::Colour(0x30e0e0e0));
    g.fillPath(spectrumPath);
}
