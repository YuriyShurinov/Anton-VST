#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "SpectrumDisplay.h"

class DeFeedbackProEditor : public juce::AudioProcessorEditor
{
public:
    explicit DeFeedbackProEditor(DeFeedbackProProcessor&);
    ~DeFeedbackProEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    DeFeedbackProProcessor& processor;

    SpectrumDisplay spectrumDisplay_;

    juce::Slider feedbackSlider_, denoiseSlider_, dereverbSlider_, mixSlider_;
    juce::Label feedbackLabel_, denoiseLabel_, dereverbLabel_, mixLabel_;

    juce::ComboBox fftSizeCombo_;
    juce::ToggleButton bypassButton_{"Bypass"};

    using Attachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ComboAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;

    std::unique_ptr<Attachment> feedbackAtt_, denoiseAtt_, dereverbAtt_, mixAtt_;
    std::unique_ptr<ComboAttachment> fftSizeAtt_;
    std::unique_ptr<ButtonAttachment> bypassAtt_;

    void setupSlider(juce::Slider& slider, juce::Label& label, const juce::String& text);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeFeedbackProEditor)
};
