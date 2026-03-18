#include "PluginProcessor.h"
#include "PluginEditor.h"

DeFeedbackProProcessor::DeFeedbackProProcessor()
    : AudioProcessor(BusesProperties()
          .withInput("Input", juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "Parameters", createParameterLayout())
{
}

DeFeedbackProProcessor::~DeFeedbackProProcessor() = default;

juce::AudioProcessorValueTreeState::ParameterLayout
DeFeedbackProProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"feedback", 1}, "Feedback", 0.0f, 1.0f, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"denoise", 1}, "Denoise", 0.0f, 1.0f, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"dereverb", 1}, "Dereverb", 0.0f, 1.0f, 0.5f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"mix", 1}, "Mix", 0.0f, 1.0f, 1.0f));
    params.push_back(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID{"fftSize", 1}, "FFT Size",
        juce::StringArray{"128", "256", "512", "1024", "2048"}, 1));
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{"bypass", 1}, "Bypass", false));

    return { params.begin(), params.end() };
}

void DeFeedbackProProcessor::prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/)
{
    // TODO: initialize SpectralEngine and modules
}

void DeFeedbackProProcessor::releaseResources() {}

void DeFeedbackProProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    // TODO: spectral processing pipeline
    (void)buffer;
}

juce::AudioProcessorEditor* DeFeedbackProProcessor::createEditor()
{
    return new DeFeedbackProEditor(*this);
}

void DeFeedbackProProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void DeFeedbackProProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xml(getXmlFromBinary(data, sizeInBytes));
    if (xml && xml->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DeFeedbackProProcessor();
}
