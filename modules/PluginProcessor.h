#pragma once
#include <JuceHeader.h>
#include "SpectralEngine.h"
#include "FeedbackDetector.h"
#include "SpectralSubtractor.h"
#include "MaskCombiner.h"
#include "MLProcessor.h"
#include "RingBuffer.h"

class DeFeedbackProProcessor : public juce::AudioProcessor, private juce::Timer
{
public:
    DeFeedbackProProcessor();
    ~DeFeedbackProProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState& getAPVTS() { return apvts; }
    RingBuffer<float>& getSpectrumBuffer() { return spectrumForGUI_; }
    int getNumBins() const { return currentFFTSize_ / 2 + 1; }

private:
    juce::AudioProcessorValueTreeState apvts;
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    static constexpr int kDefaultFFTSize = 256;
    static constexpr float kInternalSampleRate = 48000.0f;

    struct ChannelState
    {
        std::unique_ptr<SpectralEngine> engine;
        std::unique_ptr<FeedbackDetector> feedbackDetector;
        std::unique_ptr<SpectralSubtractor> spectralSubtractor;
        std::unique_ptr<MaskCombiner> maskCombiner;
    };

    std::vector<ChannelState> channels_;
    std::unique_ptr<MLProcessor> mlProcessor_;
    int currentFFTSize_ = kDefaultFFTSize;
    double hostSampleRate_ = 48000.0;

    struct ResamplerState {
        juce::LagrangeInterpolator input, output;
    };
    std::vector<ResamplerState> resamplers_;
    double resampleRatio_ = 1.0;
    bool needsResampling_ = false;
    std::vector<float> resampledInput_, resampledOutput_;

    std::vector<float> mask1_, mask2_, mask3_;

    std::atomic<int> pendingFFTSize_{0};

    RingBuffer<float> spectrumForGUI_{2048};

    void initModules(int fftSize, int maxBlockSize = 4096);
    void updateParametersForBlock();
    int getFFTSizeFromParam() const;
    void timerCallback() override;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeFeedbackProProcessor)
};
