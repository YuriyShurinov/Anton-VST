#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

class DeFeedbackProEditor : public juce::AudioProcessorEditor
{
public:
    explicit DeFeedbackProEditor(DeFeedbackProProcessor&);
    ~DeFeedbackProEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    DeFeedbackProProcessor& processor;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeFeedbackProEditor)
};
