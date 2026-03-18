#include "PluginEditor.h"

DeFeedbackProEditor::DeFeedbackProEditor(DeFeedbackProProcessor& p)
    : AudioProcessorEditor(&p), processor(p)
{
    setSize(500, 300);
}

DeFeedbackProEditor::~DeFeedbackProEditor() = default;

void DeFeedbackProEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colours::black);
    g.setColour(juce::Colours::white);
    g.setFont(15.0f);
    g.drawFittedText("DeFeedback Pro", getLocalBounds(), juce::Justification::centred, 1);
}

void DeFeedbackProEditor::resized() {}
