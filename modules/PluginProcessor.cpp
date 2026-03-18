#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cstring>

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

void DeFeedbackProProcessor::initModules(int fftSize, int maxBlockSize)
{
    currentFFTSize_ = fftSize;
    int numBins = fftSize / 2 + 1;
    int numChannels = getTotalNumInputChannels();

    channels_.clear();
    channels_.resize(numChannels);

    for (auto& ch : channels_)
    {
        ch.engine = std::make_unique<SpectralEngine>(fftSize, kInternalSampleRate);
        ch.feedbackDetector = std::make_unique<FeedbackDetector>(numBins, kInternalSampleRate, fftSize);
        ch.spectralSubtractor = std::make_unique<SpectralSubtractor>(numBins, kInternalSampleRate, fftSize);
        ch.maskCombiner = std::make_unique<MaskCombiner>(numBins);
    }

    // Shared ML processor
    #if JUCE_MAC
    juce::File modelDir = juce::File::getSpecialLocation(
        juce::File::currentApplicationFile).getChildFile("Contents/Resources/models");
    #else
    juce::File modelDir = juce::File::getSpecialLocation(
        juce::File::currentExecutableFile).getParentDirectory().getChildFile("models");
    #endif
    mlProcessor_ = std::make_unique<MLProcessor>(numBins, modelDir.getFullPathName().toStdString(), fftSize);

    mask1_.resize(numBins);
    mask2_.resize(numBins);
    mask3_.resize(numBins);

    int maxResampled = static_cast<int>(maxBlockSize * resampleRatio_) + 64;
    maxResampled = std::max(maxResampled, 4096);
    resampledInput_.resize(maxResampled);
    resampledOutput_.resize(maxResampled);

    int hopSize = fftSize / 4;
    int latencySamples = hopSize;
    if (needsResampling_)
        latencySamples += 10;
    setLatencySamples(latencySamples);
}

int DeFeedbackProProcessor::getFFTSizeFromParam() const
{
    int idx = static_cast<int>(*apvts.getRawParameterValue("fftSize"));
    constexpr int sizes[] = {128, 256, 512, 1024, 2048};
    return sizes[std::clamp(idx, 0, 4)];
}

void DeFeedbackProProcessor::updateParametersForBlock()
{
    float feedback = *apvts.getRawParameterValue("feedback");
    float denoise = *apvts.getRawParameterValue("denoise");
    float dereverb = *apvts.getRawParameterValue("dereverb");
    float mix = *apvts.getRawParameterValue("mix");

    for (auto& ch : channels_)
    {
        ch.feedbackDetector->setFeedbackAmount(feedback);
        ch.spectralSubtractor->setDenoiseAmount(denoise);
        ch.maskCombiner->setParams(feedback, denoise, dereverb, mix);
    }

    if (mlProcessor_)
    {
        mlProcessor_->setDenoiseAmount(denoise);
        mlProcessor_->setDereverbAmount(dereverb);
    }
}

void DeFeedbackProProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    hostSampleRate_ = sampleRate;
    needsResampling_ = (std::abs(sampleRate - kInternalSampleRate) > 1.0);

    if (needsResampling_)
    {
        resampleRatio_ = kInternalSampleRate / sampleRate;
        resamplers_.resize(getTotalNumInputChannels());
        for (auto& r : resamplers_) { r.input.reset(); r.output.reset(); }
    }

    initModules(getFFTSizeFromParam(), samplesPerBlock);
    startTimer(50);
}

void DeFeedbackProProcessor::releaseResources() {}

void DeFeedbackProProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    bool bypass = *apvts.getRawParameterValue("bypass") > 0.5f;
    if (bypass) return;

    int requestedFFT = getFFTSizeFromParam();
    if (requestedFFT != currentFFTSize_)
        pendingFFTSize_.store(requestedFFT);

    updateParametersForBlock();

    int numSamples = buffer.getNumSamples();
    int numChannels = std::min(buffer.getNumChannels(), (int)channels_.size());
    int numBins = currentFFTSize_ / 2 + 1;

    for (int ch = 0; ch < numChannels; ++ch)
    {
        float* channelData = buffer.getWritePointer(ch);
        auto& state = channels_[ch];

        float* processData = channelData;
        int processSamples = numSamples;

        if (needsResampling_ && ch < (int)resamplers_.size())
        {
            processSamples = resamplers_[ch].input.process(
                resampleRatio_, channelData, resampledInput_.data(),
                numSamples);
            processData = resampledInput_.data();
        }

        bool isFirstChannel = (ch == 0);
        auto maskCallback = [&](const float* magnitude, float* mask) {
            state.feedbackDetector->process(magnitude, mask1_.data());

            if (isFirstChannel && mlProcessor_)
                mlProcessor_->submitFrame(magnitude);
            if (mlProcessor_)
                mlProcessor_->computeMask2(mask2_.data());
            else
                std::fill(mask2_.begin(), mask2_.end(), 1.0f);

            state.spectralSubtractor->process(magnitude, mask3_.data());

            state.maskCombiner->combine(mask1_.data(), mask2_.data(),
                                         mask3_.data(), mask);

            if (isFirstChannel)
                spectrumForGUI_.push(magnitude, numBins);
        };

        state.engine->processBlock(processData, resampledOutput_.data(),
                                    processSamples, maskCallback);

        if (needsResampling_ && ch < (int)resamplers_.size())
        {
            resamplers_[ch].output.process(
                1.0 / resampleRatio_, resampledOutput_.data(), channelData,
                processSamples);
        }
        else
        {
            std::memcpy(channelData, resampledOutput_.data(), numSamples * sizeof(float));
        }
    }
}

void DeFeedbackProProcessor::timerCallback()
{
    int pending = pendingFFTSize_.exchange(0);
    if (pending > 0 && pending != currentFFTSize_)
    {
        initModules(pending);
    }
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
