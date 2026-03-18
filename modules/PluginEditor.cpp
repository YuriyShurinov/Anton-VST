#include "PluginEditor.h"

DeFeedbackProEditor::DeFeedbackProEditor(DeFeedbackProProcessor& p)
    : AudioProcessorEditor(&p),
      processor(p),
      spectrumDisplay_(p.getSpectrumBuffer(), p.getNumBins())
{
    setSize(500, 300);

    addAndMakeVisible(spectrumDisplay_);

    setupSlider(feedbackSlider_, feedbackLabel_, "Feedback");
    setupSlider(denoiseSlider_, denoiseLabel_, "Denoise");
    setupSlider(dereverbSlider_, dereverbLabel_, "Dereverb");
    setupSlider(mixSlider_, mixLabel_, "Mix");

    auto& apvts = processor.getAPVTS();
    feedbackAtt_ = std::make_unique<Attachment>(apvts, "feedback", feedbackSlider_);
    denoiseAtt_ = std::make_unique<Attachment>(apvts, "denoise", denoiseSlider_);
    dereverbAtt_ = std::make_unique<Attachment>(apvts, "dereverb", dereverbSlider_);
    mixAtt_ = std::make_unique<Attachment>(apvts, "mix", mixSlider_);

    fftSizeCombo_.addItemList({"128", "256", "512", "1024", "2048"}, 1);
    addAndMakeVisible(fftSizeCombo_);
    fftSizeAtt_ = std::make_unique<ComboAttachment>(apvts, "fftSize", fftSizeCombo_);

    addAndMakeVisible(bypassButton_);
    bypassAtt_ = std::make_unique<ButtonAttachment>(apvts, "bypass", bypassButton_);
}

DeFeedbackProEditor::~DeFeedbackProEditor() = default;

void DeFeedbackProEditor::setupSlider(juce::Slider& slider, juce::Label& label, const juce::String& text)
{
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 18);
    addAndMakeVisible(slider);

    label.setText(text, juce::dontSendNotification);
    label.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(label);
}

void DeFeedbackProEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff16213e));

    g.setColour(juce::Colours::white);
    g.setFont(18.0f);
    g.drawText("DeFeedback Pro", 10, 5, 200, 25, juce::Justification::centredLeft);
}

void DeFeedbackProEditor::resized()
{
    auto area = getLocalBounds().reduced(10);

    auto topBar = area.removeFromTop(30);
    bypassButton_.setBounds(topBar.removeFromRight(70));
    fftSizeCombo_.setBounds(topBar.removeFromRight(80).reduced(2));

    area.removeFromTop(5);

    spectrumDisplay_.setBounds(area.removeFromTop(120));

    area.removeFromTop(10);

    auto knobArea = area.removeFromTop(100);
    int knobWidth = knobArea.getWidth() / 4;

    auto setupKnobBounds = [&](juce::Slider& slider, juce::Label& label, juce::Rectangle<int> bounds)
    {
        label.setBounds(bounds.removeFromTop(18));
        slider.setBounds(bounds);
    };

    setupKnobBounds(feedbackSlider_, feedbackLabel_, knobArea.removeFromLeft(knobWidth));
    setupKnobBounds(denoiseSlider_, denoiseLabel_, knobArea.removeFromLeft(knobWidth));
    setupKnobBounds(dereverbSlider_, dereverbLabel_, knobArea.removeFromLeft(knobWidth));
    setupKnobBounds(mixSlider_, mixLabel_, knobArea);
}
