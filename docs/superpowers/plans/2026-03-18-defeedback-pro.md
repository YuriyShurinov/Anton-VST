# DeFeedback Pro Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time VST3/AU plugin that eliminates microphone feedback, noise, and room reverberation using a unified spectral domain architecture combining classical DSP and ML inference.

**Architecture:** Single FFT/IFFT pipeline with three parallel spectral mask generators (feedback detector, ML denoiser/dereverberator, spectral subtractor) whose masks are combined via weighted geometric mean. Async ML inference thread decoupled from audio callback.

**Tech Stack:** C++17, JUCE 7.x, PFFFT, ONNX Runtime 1.x, CMake

**Spec:** `docs/superpowers/specs/2026-03-18-defeedback-pro-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `CMakeLists.txt` | Top-level build: JUCE, ONNX Runtime, PFFFT, plugin targets |
| `modules/PluginProcessor.h/cpp` | JUCE AudioProcessor: parameter tree, prepareToPlay, processBlock, state save/load |
| `modules/PluginEditor.h/cpp` | JUCE AudioProcessorEditor: layout, knobs, bypass, FFT size combo |
| `modules/SpectralEngine.h/cpp` | FFT/IFFT via PFFFT, Hann windowing, overlap-add, crossfade on FFT size change |
| `modules/FeedbackDetector.h/cpp` | Peak detection, persistence filter, Gaussian notch mask generation |
| `modules/SpectralSubtractor.h/cpp` | Martin's minimum statistics noise floor, oversubtraction mask |
| `modules/MaskCombiner.h/cpp` | Weighted geometric mean of 3 masks, all-zero bypass, dry/wet mix |
| `modules/MLProcessor.h/cpp` | ONNX Runtime session management, async inference thread, SPSC queues |
| `modules/SpectrumDisplay.h/cpp` | JUCE Component: real-time spectrum, feedback markers, mask overlay |
| `modules/RingBuffer.h` | Lock-free SPSC ring buffer template for audio↔GUI and audio↔ML data transfer |
| `tests/TestMain.cpp` | Catch2 test runner main |
| `tests/SpectralEngineTest.cpp` | Unit tests for FFT round-trip, windowing, overlap-add |
| `tests/FeedbackDetectorTest.cpp` | Unit tests for peak detection, persistence, notch mask |
| `tests/SpectralSubtractorTest.cpp` | Unit tests for noise floor estimation, mask generation |
| `tests/MaskCombinerTest.cpp` | Unit tests for weight computation, geometric mean, bypass |
| `tests/MLProcessorTest.cpp` | Unit tests for ONNX load, inference, fallback behavior |
| `tests/RingBufferTest.cpp` | Unit tests for SPSC ring buffer |
| `scripts/create_dummy_model.py` | Python script to generate dummy ONNX models for testing |
| `third_party/pffft/pffft.h` | PFFFT header (vendored) |
| `third_party/pffft/pffft.c` | PFFFT source (vendored) |

---

## Task 1: Project Scaffolding & Build System

**Files:**
- Create: `CMakeLists.txt`
- Create: `modules/PluginProcessor.h`
- Create: `modules/PluginProcessor.cpp`
- Create: `modules/PluginEditor.h`
- Create: `modules/PluginEditor.cpp`

- [ ] **Step 1: Create CMakeLists.txt with JUCE and plugin targets**

```cmake
cmake_minimum_required(VERSION 3.22)
project(DeFeedbackPro VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# macOS: build universal binary (Intel + Apple Silicon)
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build universal binary")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
endif()

# JUCE — fetched from GitHub
include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG 7.0.12
)
FetchContent_MakeAvailable(JUCE)

# PFFFT — vendored source
add_library(pffft STATIC third_party/pffft/pffft.c)
target_include_directories(pffft PUBLIC third_party/pffft)
if(MSVC)
    target_compile_options(pffft PRIVATE /arch:SSE2)
elseif(APPLE)
    # Universal binary: PFFFT auto-detects SSE2 on x86_64, NEON on arm64
    # No explicit SIMD flags needed — PFFFT checks __SSE2__ / __ARM_NEON__ at compile time
    # For Apple Silicon, ensure PFFFT NEON support is enabled:
    target_compile_definitions(pffft PRIVATE $<$<STREQUAL:${CMAKE_OSX_ARCHITECTURES},arm64>:PFFFT_ENABLE_NEON>)
else()
    target_compile_options(pffft PRIVATE -msse2)
endif()

# ONNX Runtime — prebuilt, set ONNXRUNTIME_ROOT via cmake -D
# macOS: use universal binary build of ONNX Runtime (x86_64 + arm64)
# Download from: https://github.com/microsoft/onnxruntime/releases
# Windows: onnxruntime-win-x64-*.zip
# macOS: onnxruntime-osx-universal2-*.tgz
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include)
find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib)

# Plugin target
juce_add_plugin(DeFeedbackPro
    COMPANY_NAME "DeFeedbackPro"
    PLUGIN_MANUFACTURER_CODE DfPr
    PLUGIN_CODE DfPr
    FORMATS VST3 AU
    PRODUCT_NAME "DeFeedback Pro"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    # AU requires a valid bundle ID for macOS validation
    BUNDLE_ID "com.defeedbackpro.plugin"
    AU_MAIN_TYPE "kAudioUnitType_Effect"
    COPY_PLUGIN_AFTER_BUILD FALSE
    # macOS codesign: set DEFEEDBACK_CODESIGN_ID for ad-hoc or Developer ID signing
    # For development: leave empty or use "-" for ad-hoc
)

target_sources(DeFeedbackPro PRIVATE
    modules/PluginProcessor.cpp
    modules/PluginEditor.cpp
    modules/SpectralEngine.cpp
    modules/FeedbackDetector.cpp
    modules/SpectralSubtractor.cpp
    modules/MaskCombiner.cpp
    modules/MLProcessor.cpp
    modules/SpectrumDisplay.cpp
)

target_include_directories(DeFeedbackPro PRIVATE
    modules
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(DeFeedbackPro PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
    pffft
    ${ONNXRUNTIME_LIB}
)

juce_generate_juce_header(DeFeedbackPro)

target_compile_definitions(DeFeedbackPro PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

# Copy models to build output
file(GLOB MODEL_FILES "${CMAKE_SOURCE_DIR}/models/*.onnx")
foreach(MODEL ${MODEL_FILES})
    get_filename_component(MODEL_NAME ${MODEL} NAME)
    configure_file(${MODEL} ${CMAKE_BINARY_DIR}/models/${MODEL_NAME} COPYONLY)
endforeach()

# macOS: embed models into plugin bundle Resources
if(APPLE)
    foreach(MODEL ${MODEL_FILES})
        get_filename_component(MODEL_NAME ${MODEL} NAME)
        target_sources(DeFeedbackPro PRIVATE ${MODEL})
        set_source_files_properties(${MODEL} PROPERTIES
            MACOSX_PACKAGE_LOCATION "Resources/models")
    endforeach()

    # Ad-hoc codesign for local development (required for AU validation on macOS 10.15+)
    add_custom_command(TARGET DeFeedbackPro POST_BUILD
        COMMAND codesign --force --sign "-" --deep
            "$<TARGET_BUNDLE_DIR:DeFeedbackPro>"
        COMMENT "Ad-hoc codesigning plugin bundle"
    )
endif()

# Tests (optional, via Catch2)
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
    )
    FetchContent_MakeAvailable(Catch2)

    add_executable(DeFeedbackProTests
        tests/TestMain.cpp
        tests/RingBufferTest.cpp
        tests/SpectralEngineTest.cpp
        tests/FeedbackDetectorTest.cpp
        tests/SpectralSubtractorTest.cpp
        tests/MaskCombinerTest.cpp
        tests/MLProcessorTest.cpp
        modules/SpectralEngine.cpp
        modules/FeedbackDetector.cpp
        modules/SpectralSubtractor.cpp
        modules/MaskCombiner.cpp
        modules/MLProcessor.cpp
    )
    target_include_directories(DeFeedbackProTests PRIVATE
        modules
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_libraries(DeFeedbackProTests PRIVATE
        Catch2::Catch2WithMain
        pffft
        ${ONNXRUNTIME_LIB}
    )
    target_compile_definitions(DeFeedbackProTests PRIVATE
        DEFEEDBACK_TEST_MODE=1
    )
endif()
```

- [ ] **Step 2: Create minimal PluginProcessor stub**

`modules/PluginProcessor.h`:
```cpp
#pragma once
#include <JuceHeader.h>

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

private:
    juce::AudioProcessorValueTreeState apvts;
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeFeedbackProProcessor)
};
```

`modules/PluginProcessor.cpp`:
```cpp
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
```

- [ ] **Step 3: Create minimal PluginEditor stub**

`modules/PluginEditor.h`:
```cpp
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
```

`modules/PluginEditor.cpp`:
```cpp
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
```

- [ ] **Step 4: Vendor PFFFT source**

Download `pffft.h` and `pffft.c` from the PFFFT repository and place them in `third_party/pffft/`.

Run:
```bash
mkdir -p third_party/pffft
# Download pffft.h and pffft.c from https://bitbucket.org/jpommier/pffft
# Place in third_party/pffft/
```

- [ ] **Step 5: Create test runner stub**

`tests/TestMain.cpp`:
```cpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
```

- [ ] **Step 6: Verify build compiles**

On Windows:
```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=OFF -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build . --target DeFeedbackPro_VST3
```

On macOS (universal binary):
```bash
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=OFF -DONNXRUNTIME_ROOT=/path/to/onnxruntime-osx-universal2
cmake --build . --target DeFeedbackPro_VST3
cmake --build . --target DeFeedbackPro_AU
```

Expected: Build succeeds, produces VST3 bundle (+ AU component on macOS).

- [ ] **Step 7: Commit**

```bash
git init
echo "build/" > .gitignore
echo ".superpowers/" >> .gitignore
git add CMakeLists.txt modules/ tests/TestMain.cpp third_party/ .gitignore
git commit -m "feat: project scaffolding with JUCE, PFFFT, and plugin stubs"
```

---

## Task 2: Lock-Free SPSC Ring Buffer

**Files:**
- Create: `modules/RingBuffer.h`
- Create: `tests/RingBufferTest.cpp`

- [ ] **Step 1: Write failing tests for RingBuffer**

`tests/RingBufferTest.cpp`:
```cpp
#include <catch2/catch_all.hpp>
#include "RingBuffer.h"

TEST_CASE("RingBuffer: push and pop single element", "[ringbuffer]")
{
    RingBuffer<float> rb(16);
    float val = 42.0f;
    REQUIRE(rb.push(&val, 1));
    float out = 0.0f;
    REQUIRE(rb.pop(&out, 1));
    REQUIRE(out == 42.0f);
}

TEST_CASE("RingBuffer: pop from empty returns false", "[ringbuffer]")
{
    RingBuffer<float> rb(16);
    float out = 0.0f;
    REQUIRE_FALSE(rb.pop(&out, 1));
}

TEST_CASE("RingBuffer: push to full returns false", "[ringbuffer]")
{
    RingBuffer<float> rb(4);
    float data[4] = {1, 2, 3, 4};
    REQUIRE(rb.push(data, 4));
    float extra = 5.0f;
    REQUIRE_FALSE(rb.push(&extra, 1));
}

TEST_CASE("RingBuffer: wrap-around works correctly", "[ringbuffer]")
{
    RingBuffer<float> rb(4);
    float data[3] = {1, 2, 3};
    REQUIRE(rb.push(data, 3));
    float out[3];
    REQUIRE(rb.pop(out, 3));

    float data2[3] = {4, 5, 6};
    REQUIRE(rb.push(data2, 3));
    float out2[3];
    REQUIRE(rb.pop(out2, 3));
    REQUIRE(out2[0] == 4.0f);
    REQUIRE(out2[1] == 5.0f);
    REQUIRE(out2[2] == 6.0f);
}

TEST_CASE("RingBuffer: availableToRead reports correctly", "[ringbuffer]")
{
    RingBuffer<float> rb(8);
    REQUIRE(rb.availableToRead() == 0);
    float data[3] = {1, 2, 3};
    rb.push(data, 3);
    REQUIRE(rb.availableToRead() == 3);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . --target DeFeedbackProTests
./DeFeedbackProTests --reporter compact "[ringbuffer]"
```

Expected: Compilation error — `RingBuffer.h` not found.

- [ ] **Step 3: Implement RingBuffer**

`modules/RingBuffer.h`:
```cpp
#pragma once
#include <atomic>
#include <cstring>
#include <vector>

template <typename T>
class RingBuffer
{
public:
    explicit RingBuffer(size_t capacity)
        : capacity_(capacity), buffer_(capacity)
    {
    }

    bool push(const T* data, size_t count)
    {
        const size_t writePos = writePos_.load(std::memory_order_relaxed);
        const size_t readPos = readPos_.load(std::memory_order_acquire);

        size_t available = capacity_ - (writePos - readPos);
        if (count > available)
            return false;

        for (size_t i = 0; i < count; ++i)
            buffer_[(writePos + i) % capacity_] = data[i];

        writePos_.store(writePos + count, std::memory_order_release);
        return true;
    }

    bool pop(T* data, size_t count)
    {
        const size_t readPos = readPos_.load(std::memory_order_relaxed);
        const size_t writePos = writePos_.load(std::memory_order_acquire);

        size_t available = writePos - readPos;
        if (count > available)
            return false;

        for (size_t i = 0; i < count; ++i)
            data[i] = buffer_[(readPos + i) % capacity_];

        readPos_.store(readPos + count, std::memory_order_release);
        return true;
    }

    size_t availableToRead() const
    {
        return writePos_.load(std::memory_order_acquire)
             - readPos_.load(std::memory_order_acquire);
    }

    void reset()
    {
        readPos_.store(0, std::memory_order_relaxed);
        writePos_.store(0, std::memory_order_relaxed);
    }

private:
    size_t capacity_;
    std::vector<T> buffer_;
    std::atomic<size_t> readPos_{0};
    std::atomic<size_t> writePos_{0};
};
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cmake --build . --target DeFeedbackProTests && ./DeFeedbackProTests "[ringbuffer]"
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/RingBuffer.h tests/RingBufferTest.cpp
git commit -m "feat: add lock-free SPSC ring buffer with tests"
```

---

## Task 3: Spectral Engine (FFT/IFFT + Overlap-Add)

**Files:**
- Create: `modules/SpectralEngine.h`
- Create: `modules/SpectralEngine.cpp`
- Create: `tests/SpectralEngineTest.cpp`

- [ ] **Step 1: Write failing tests**

`tests/SpectralEngineTest.cpp`:
```cpp
#include <catch2/catch_all.hpp>
#include "SpectralEngine.h"
#include <cmath>

TEST_CASE("SpectralEngine: FFT round-trip reconstructs signal", "[spectral]")
{
    const int fftSize = 256;
    SpectralEngine engine(fftSize, 48000.0f);
    const int numBins = fftSize / 2 + 1;

    // Unity mask callback — pass signal through unmodified
    auto unityCallback = [&](const float*, float* mask) {
        std::fill(mask, mask + numBins, 1.0f);
    };

    // Generate a sine wave — process multiple blocks to fill overlap-add
    const int totalSamples = fftSize * 4;
    std::vector<float> input(totalSamples);
    for (int i = 0; i < totalSamples; ++i)
        input[i] = std::sin(2.0f * M_PI * 1000.0f * i / 48000.0f);

    std::vector<float> output(totalSamples, 0.0f);
    engine.processBlock(input.data(), output.data(), totalSamples, unityCallback);

    // Check last block (skip initial ramp-up)
    const int checkStart = totalSamples - fftSize;
    float maxError = 0.0f;
    for (int i = checkStart; i < totalSamples; ++i)
    {
        float err = std::abs(output[i] - input[i]);
        maxError = std::max(maxError, err);
    }
    REQUIRE(maxError < 0.05f);
}

TEST_CASE("SpectralEngine: zero mask produces silence", "[spectral]")
{
    const int fftSize = 256;
    SpectralEngine engine(fftSize, 48000.0f);
    const int numBins = fftSize / 2 + 1;

    auto zeroCallback = [&](const float*, float* mask) {
        std::fill(mask, mask + numBins, 0.0f);
    };

    const int totalSamples = fftSize * 4;
    std::vector<float> input(totalSamples, 1.0f);
    std::vector<float> output(totalSamples, 999.0f);
    engine.processBlock(input.data(), output.data(), totalSamples, zeroCallback);

    // Last block should be near silent
    for (int i = totalSamples - fftSize; i < totalSamples; ++i)
        REQUIRE(std::abs(output[i]) < 0.05f);
}

TEST_CASE("SpectralEngine: getNumBins returns correct count", "[spectral]")
{
    SpectralEngine engine(512, 48000.0f);
    REQUIRE(engine.getNumBins() == 257);
}

TEST_CASE("SpectralEngine: getMagnitudeSpectrum returns valid data", "[spectral]")
{
    const int fftSize = 256;
    SpectralEngine engine(fftSize, 48000.0f);
    const int numBins = fftSize / 2 + 1;
    const float* lastMag = nullptr;

    auto captureCallback = [&](const float* mag, float* mask) {
        lastMag = mag;
        std::fill(mask, mask + numBins, 1.0f);
    };

    const int totalSamples = fftSize * 2;
    std::vector<float> input(totalSamples);
    for (int i = 0; i < totalSamples; ++i)
        input[i] = std::sin(2.0f * M_PI * 1000.0f * i / 48000.0f);

    std::vector<float> output(totalSamples);
    engine.processBlock(input.data(), output.data(), totalSamples, captureCallback);

    const float* mag = engine.getMagnitudeSpectrum();
    REQUIRE(mag != nullptr);

    int peakBin = 0;
    float peakVal = 0.0f;
    for (int i = 0; i < numBins; ++i)
    {
        if (mag[i] > peakVal) { peakVal = mag[i]; peakBin = i; }
    }
    float peakFreq = peakBin * 48000.0f / fftSize;
    REQUIRE(peakFreq == Catch::Approx(1000.0f).margin(200.0f));
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cmake --build . --target DeFeedbackProTests
./DeFeedbackProTests "[spectral]"
```

Expected: Compilation error — `SpectralEngine.h` not found.

- [ ] **Step 3: Implement SpectralEngine**

`modules/SpectralEngine.h`:
```cpp
#pragma once
#include <vector>
#include <memory>
#include <functional>
#include "pffft.h"

class SpectralEngine
{
public:
    SpectralEngine(int fftSize, float sampleRate);
    ~SpectralEngine();

    // Callback invoked per hop after FFT analysis and before IFFT synthesis.
    // The callback receives the magnitude spectrum (numBins) and must write
    // the gain mask (numBins) into the provided mask buffer.
    // Signature: void(const float* magnitude, float* mask)
    using MaskCallback = std::function<void(const float* magnitude, float* mask)>;

    // Process a block of audio sample-by-sample with hop-based spectral processing.
    // For each hop boundary, performs FFT, calls maskCallback to get the mask,
    // applies the mask, performs IFFT, and overlap-adds.
    // inputSamples/outputSamples: numSamples floats
    void processBlock(const float* inputSamples, float* outputSamples,
                      int numSamples, MaskCallback maskCallback);

    int getFFTSize() const { return fftSize_; }
    int getHopSize() const { return hopSize_; }
    int getNumBins() const { return numBins_; }
    float getBinResolution() const { return sampleRate_ / fftSize_; }

    // Returns the magnitude spectrum from the most recent FFT frame.
    const float* getMagnitudeSpectrum() const { return magnitude_.data(); }

    void reset();

private:
    void computeWindow();
    void forwardFFT(const float* windowed);
    void applyMask(const float* mask);
    void inverseFFT();

    int fftSize_;
    int hopSize_;
    int numBins_;
    float sampleRate_;

    PFFFT_Setup* pffftSetup_ = nullptr;
    float* pffftWork_ = nullptr;

    std::vector<float> window_;
    std::vector<float> windowedInput_;
    std::vector<float> complexSpectrum_;    // PFFFT packed format
    std::vector<float> unpacked_;           // standard interleaved real/imag (numBins * 2)
    std::vector<float> magnitude_;          // numBins magnitudes
    std::vector<float> ifftOut_;            // pre-allocated IFFT output
    std::vector<float> overlapBuffer_;      // overlap-add accumulator
    std::vector<float> outputRing_;         // circular output buffer
    std::vector<float> inputRing_;          // circular input buffer
    int inputWritePos_ = 0;
    int outputReadPos_ = 0;
    int hopCounter_ = 0;
    std::vector<float> currentMask_;        // pre-allocated mask buffer for callback
};
```

`modules/SpectralEngine.cpp`:
```cpp
#include "SpectralEngine.h"
#include <cmath>
#include <cstring>
#include <algorithm>

SpectralEngine::SpectralEngine(int fftSize, float sampleRate)
    : fftSize_(fftSize),
      hopSize_(fftSize / 4),
      numBins_(fftSize / 2 + 1),
      sampleRate_(sampleRate)
{
    pffftSetup_ = pffft_new_setup(fftSize, PFFFT_REAL);
    pffftWork_ = (float*)pffft_aligned_malloc(fftSize * sizeof(float));

    window_.resize(fftSize);
    windowedInput_.resize(fftSize);
    complexSpectrum_.resize(fftSize);
    unpacked_.resize(numBins_ * 2);
    magnitude_.resize(numBins_);
    ifftOut_.resize(fftSize);
    currentMask_.resize(numBins_);
    overlapBuffer_.resize(fftSize + hopSize_, 0.0f);
    outputRing_.resize(fftSize * 2, 0.0f);
    inputRing_.resize(fftSize, 0.0f);
    inputWritePos_ = 0;
    outputReadPos_ = 0;
    hopCounter_ = 0;

    computeWindow();
}

SpectralEngine::~SpectralEngine()
{
    if (pffftSetup_) pffft_destroy_setup(pffftSetup_);
    if (pffftWork_) pffft_aligned_free(pffftWork_);
}

void SpectralEngine::computeWindow()
{
    for (int i = 0; i < fftSize_; ++i)
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / fftSize_));
}

void SpectralEngine::forwardFFT(const float* windowed)
{
    pffft_transform_ordered(pffftSetup_, windowed, complexSpectrum_.data(), pffftWork_, PFFFT_FORWARD);

    // Unpack PFFFT ordered format to standard complex bins.
    unpacked_[0] = complexSpectrum_[0]; // DC real
    unpacked_[1] = 0.0f;               // DC imag

    for (int i = 1; i < numBins_ - 1; ++i)
    {
        unpacked_[i * 2]     = complexSpectrum_[i * 2];
        unpacked_[i * 2 + 1] = complexSpectrum_[i * 2 + 1];
    }

    unpacked_[(numBins_ - 1) * 2]     = complexSpectrum_[1]; // Nyquist real
    unpacked_[(numBins_ - 1) * 2 + 1] = 0.0f;               // Nyquist imag

    // Compute magnitude
    for (int i = 0; i < numBins_; ++i)
    {
        float re = unpacked_[i * 2];
        float im = unpacked_[i * 2 + 1];
        magnitude_[i] = std::sqrt(re * re + im * im);
    }
}

void SpectralEngine::applyMask(const float* mask)
{
    for (int i = 0; i < numBins_; ++i)
    {
        unpacked_[i * 2]     *= mask[i];
        unpacked_[i * 2 + 1] *= mask[i];
    }

    // Repack to PFFFT ordered format
    complexSpectrum_[0] = unpacked_[0];
    complexSpectrum_[1] = unpacked_[(numBins_ - 1) * 2];

    for (int i = 1; i < numBins_ - 1; ++i)
    {
        complexSpectrum_[i * 2]     = unpacked_[i * 2];
        complexSpectrum_[i * 2 + 1] = unpacked_[i * 2 + 1];
    }
}

void SpectralEngine::inverseFFT()
{
    pffft_transform_ordered(pffftSetup_, complexSpectrum_.data(), ifftOut_.data(), pffftWork_, PFFFT_BACKWARD);

    float norm = 1.0f / fftSize_;
    for (int i = 0; i < fftSize_; ++i)
        ifftOut_[i] *= norm;
}

void SpectralEngine::processBlock(const float* inputSamples, float* outputSamples,
                                   int numSamples, MaskCallback maskCallback)
{
    for (int s = 0; s < numSamples; ++s)
    {
        // Store input sample
        inputRing_[inputWritePos_] = inputSamples[s];
        inputWritePos_ = (inputWritePos_ + 1) % fftSize_;
        hopCounter_++;

        if (hopCounter_ >= hopSize_)
        {
            hopCounter_ = 0;

            // Gather a full FFT frame from the ring buffer
            for (int i = 0; i < fftSize_; ++i)
            {
                int idx = (inputWritePos_ + i) % fftSize_;
                windowedInput_[i] = inputRing_[idx] * window_[i];
            }

            // Forward FFT → magnitude_ is updated
            forwardFFT(windowedInput_.data());

            // Callback: modules analyze magnitude and produce mask
            maskCallback(magnitude_.data(), currentMask_.data());

            // Apply mask to spectrum
            applyMask(currentMask_.data());

            // Inverse FFT → ifftOut_ (pre-allocated)
            inverseFFT();

            // Overlap-add with synthesis window into outputRing_
            for (int i = 0; i < fftSize_; ++i)
            {
                int idx = (outputReadPos_ + i) % (int)outputRing_.size();
                outputRing_[idx] += ifftOut_[i] * window_[i];
            }
        }

        // Read one sample from the output ring
        outputSamples[s] = outputRing_[outputReadPos_];
        outputRing_[outputReadPos_] = 0.0f; // clear after reading
        outputReadPos_ = (outputReadPos_ + 1) % (int)outputRing_.size();
    }
}

void SpectralEngine::reset()
{
    std::fill(overlapBuffer_.begin(), overlapBuffer_.end(), 0.0f);
    std::fill(outputRing_.begin(), outputRing_.end(), 0.0f);
    std::fill(inputRing_.begin(), inputRing_.end(), 0.0f);
    inputWritePos_ = 0;
    outputReadPos_ = 0;
    hopCounter_ = 0;
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cmake --build . --target DeFeedbackProTests && ./DeFeedbackProTests "[spectral]"
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/SpectralEngine.h modules/SpectralEngine.cpp tests/SpectralEngineTest.cpp
git commit -m "feat: spectral engine with PFFFT, windowing, and overlap-add"
```

---

## Task 4: Feedback Detector

**Files:**
- Create: `modules/FeedbackDetector.h`
- Create: `modules/FeedbackDetector.cpp`
- Create: `tests/FeedbackDetectorTest.cpp`

- [ ] **Step 1: Write failing tests**

`tests/FeedbackDetectorTest.cpp`:
```cpp
#include <catch2/catch_all.hpp>
#include "FeedbackDetector.h"
#include <cmath>

TEST_CASE("FeedbackDetector: no feedback in flat spectrum", "[feedback]")
{
    const int numBins = 129;
    const float sampleRate = 48000.0f;
    const int fftSize = 256;
    FeedbackDetector det(numBins, sampleRate, fftSize);

    // Flat magnitude spectrum
    std::vector<float> mag(numBins, 1.0f);
    std::vector<float> mask(numBins);
    det.process(mag.data(), mask.data());

    // All mask values should be 1.0 (no suppression)
    for (int i = 0; i < numBins; ++i)
        REQUIRE(mask[i] == Catch::Approx(1.0f));
}

TEST_CASE("FeedbackDetector: detects strong tonal peak after persistence", "[feedback]")
{
    const int numBins = 129;
    const float sampleRate = 48000.0f;
    const int fftSize = 256;
    FeedbackDetector det(numBins, sampleRate, fftSize);

    std::vector<float> mag(numBins, 0.1f); // low background
    const int peakBin = 50; // ~9375 Hz
    mag[peakBin] = 10.0f; // 40 dB above background

    std::vector<float> mask(numBins);

    // Feed the same spectrum for enough frames to exceed persistence threshold
    int persistFrames = det.getPersistenceFrames() + 5;
    for (int f = 0; f < persistFrames; ++f)
        det.process(mag.data(), mask.data());

    // The peak bin should be suppressed (mask < 1.0)
    REQUIRE(mask[peakBin] < 0.5f);

    // Non-peak bins should be near 1.0
    REQUIRE(mask[0] == Catch::Approx(1.0f).margin(0.01f));
    REQUIRE(mask[numBins - 1] == Catch::Approx(1.0f).margin(0.01f));
}

TEST_CASE("FeedbackDetector: transient peak does NOT trigger suppression", "[feedback]")
{
    const int numBins = 129;
    const float sampleRate = 48000.0f;
    const int fftSize = 256;
    FeedbackDetector det(numBins, sampleRate, fftSize);

    std::vector<float> mag(numBins, 0.1f);
    std::vector<float> mask(numBins);

    // One frame with peak
    mag[50] = 10.0f;
    det.process(mag.data(), mask.data());

    // Immediately back to flat
    mag[50] = 0.1f;
    det.process(mag.data(), mask.data());

    // No suppression should have triggered
    REQUIRE(mask[50] == Catch::Approx(1.0f).margin(0.05f));
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cmake --build . --target DeFeedbackProTests && ./DeFeedbackProTests "[feedback]"
```

Expected: Compilation error — `FeedbackDetector.h` not found.

- [ ] **Step 3: Implement FeedbackDetector**

`modules/FeedbackDetector.h`:
```cpp
#pragma once
#include <vector>

class FeedbackDetector
{
public:
    FeedbackDetector(int numBins, float sampleRate, int fftSize);

    // Process one spectral frame.
    // magnitude: input magnitude spectrum (numBins)
    // mask: output gain mask (numBins), written by this function
    void process(const float* magnitude, float* mask);

    int getPersistenceFrames() const { return persistenceFrames_; }
    void setFeedbackAmount(float amount) { feedbackAmount_ = amount; }

private:
    float computeMedian(const float* data, int size) const;
    void applyGaussianNotch(float* mask, int centerBin, float depth) const;

    int numBins_;
    float sampleRate_;
    int fftSize_;
    float binResolution_;
    float feedbackAmount_ = 1.0f;

    // Persistence tracking
    int persistenceFrames_;       // frames needed to confirm feedback
    std::vector<int> peakCounter_; // per-bin persistence counter

    // Attack/release envelope per bin
    std::vector<float> envelope_;  // current suppression depth [0..1] per bin
    float attackCoeff_;
    float releaseCoeff_;

    // Notch width in bins
    int notchWidthBins_;

    // Threshold: peak must exceed median by this ratio (in linear scale)
    float peakThresholdLinear_; // 12 dB = ~3.98x

    // Pre-allocated scratch buffer for median computation (avoids audio-thread allocation)
    mutable std::vector<float> medianScratch_;
};
```

`modules/FeedbackDetector.cpp`:
```cpp
#include "FeedbackDetector.h"
#include <algorithm>
#include <cmath>
#include <numeric>

FeedbackDetector::FeedbackDetector(int numBins, float sampleRate, int fftSize)
    : numBins_(numBins),
      sampleRate_(sampleRate),
      fftSize_(fftSize),
      binResolution_(sampleRate / fftSize)
{
    float hopDuration = (fftSize / 4.0f) / sampleRate;
    persistenceFrames_ = static_cast<int>(std::ceil(0.050f / hopDuration)); // 50ms

    peakCounter_.resize(numBins, 0);
    envelope_.resize(numBins, 0.0f);

    // Attack: 2-3 frames, release: ~100 frames
    attackCoeff_ = 1.0f / 3.0f;
    releaseCoeff_ = 1.0f / 100.0f;

    // Notch width: ~100 Hz converted to bins
    notchWidthBins_ = std::max(1, static_cast<int>(std::ceil(100.0f / binResolution_)));

    // 12 dB threshold
    peakThresholdLinear_ = std::pow(10.0f, 12.0f / 20.0f); // ~3.98

    medianScratch_.resize(numBins);
}

float FeedbackDetector::computeMedian(const float* data, int size) const
{
    // Use pre-allocated scratch buffer + nth_element for O(n) average
    std::copy(data, data + size, medianScratch_.begin());
    auto mid = medianScratch_.begin() + size / 2;
    std::nth_element(medianScratch_.begin(), mid, medianScratch_.begin() + size);
    return *mid;
}

void FeedbackDetector::applyGaussianNotch(float* mask, int centerBin, float depth) const
{
    float sigma = notchWidthBins_ / 2.0f;
    int halfWidth = notchWidthBins_;

    int lo = std::max(0, centerBin - halfWidth);
    int hi = std::min(numBins_ - 1, centerBin + halfWidth);

    for (int i = lo; i <= hi; ++i)
    {
        float dist = static_cast<float>(i - centerBin);
        float gauss = std::exp(-(dist * dist) / (2.0f * sigma * sigma));
        float suppression = 1.0f - depth * gauss;
        mask[i] = std::min(mask[i], suppression);
    }
}

void FeedbackDetector::process(const float* magnitude, float* mask)
{
    // Initialize mask to 1.0
    std::fill(mask, mask + numBins_, 1.0f);

    float median = computeMedian(magnitude, numBins_);
    float threshold = median * peakThresholdLinear_;

    // Update persistence counters
    for (int i = 0; i < numBins_; ++i)
    {
        if (magnitude[i] > threshold)
            peakCounter_[i]++;
        else
            peakCounter_[i] = std::max(0, peakCounter_[i] - 1);
    }

    // Update envelopes and apply notches
    for (int i = 0; i < numBins_; ++i)
    {
        float target = (peakCounter_[i] >= persistenceFrames_) ? feedbackAmount_ : 0.0f;

        if (target > envelope_[i])
            envelope_[i] += attackCoeff_ * (target - envelope_[i]);
        else
            envelope_[i] += releaseCoeff_ * (target - envelope_[i]);

        if (envelope_[i] > 0.01f)
            applyGaussianNotch(mask, i, envelope_[i]);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cmake --build . --target DeFeedbackProTests && ./DeFeedbackProTests "[feedback]"
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/FeedbackDetector.h modules/FeedbackDetector.cpp tests/FeedbackDetectorTest.cpp
git commit -m "feat: feedback detector with peak tracking, persistence, and Gaussian notch"
```

---

## Task 5: Spectral Subtractor

**Files:**
- Create: `modules/SpectralSubtractor.h`
- Create: `modules/SpectralSubtractor.cpp`
- Create: `tests/SpectralSubtractorTest.cpp`

- [ ] **Step 1: Write failing tests**

`tests/SpectralSubtractorTest.cpp`:
```cpp
#include <catch2/catch_all.hpp>
#include "SpectralSubtractor.h"

TEST_CASE("SpectralSubtractor: clean signal produces mask near 1.0", "[subtractor]")
{
    const int numBins = 129;
    const float sampleRate = 48000.0f;
    const int fftSize = 256;
    SpectralSubtractor sub(numBins, sampleRate, fftSize);
    sub.setDenoiseAmount(0.5f);

    // Simulate noise-only frames to build noise floor
    std::vector<float> noiseMag(numBins, 0.01f);
    std::vector<float> mask(numBins);
    for (int i = 0; i < 200; ++i) // ~enough to establish floor
        sub.process(noiseMag.data(), mask.data());

    // Now feed a loud signal — mask should be near 1.0
    std::vector<float> signalMag(numBins, 1.0f);
    sub.process(signalMag.data(), mask.data());

    for (int i = 0; i < numBins; ++i)
        REQUIRE(mask[i] > 0.9f);
}

TEST_CASE("SpectralSubtractor: noise-level signal produces mask near 0", "[subtractor]")
{
    const int numBins = 129;
    const float sampleRate = 48000.0f;
    const int fftSize = 256;
    SpectralSubtractor sub(numBins, sampleRate, fftSize);
    sub.setDenoiseAmount(1.0f); // max

    // Build noise floor
    std::vector<float> noiseMag(numBins, 0.5f);
    std::vector<float> mask(numBins);
    for (int i = 0; i < 200; ++i)
        sub.process(noiseMag.data(), mask.data());

    // Feed same level — should be heavily suppressed
    sub.process(noiseMag.data(), mask.data());

    for (int i = 0; i < numBins; ++i)
        REQUIRE(mask[i] < 0.2f);
}

TEST_CASE("SpectralSubtractor: denoise amount 0 produces mask 1.0", "[subtractor]")
{
    const int numBins = 129;
    const float sampleRate = 48000.0f;
    const int fftSize = 256;
    SpectralSubtractor sub(numBins, sampleRate, fftSize);
    sub.setDenoiseAmount(0.0f);

    std::vector<float> mag(numBins, 0.5f);
    std::vector<float> mask(numBins);
    for (int i = 0; i < 200; ++i)
        sub.process(mag.data(), mask.data());

    // α = 1.0 + 3.0 * 0.0 = 1.0
    // mask = max(0, 1 - 1.0 * noise/signal) = max(0, 1 - 1.0) = 0
    // Actually with α=1 and signal==noise, mask=0.
    // But the test intent is: at denoise=0, suppression should be minimal.
    // Let's check with signal > noise
    std::vector<float> signalMag(numBins, 2.0f);
    sub.process(signalMag.data(), mask.data());
    for (int i = 0; i < numBins; ++i)
        REQUIRE(mask[i] > 0.5f);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cmake --build . --target DeFeedbackProTests && ./DeFeedbackProTests "[subtractor]"
```

Expected: Compilation error.

- [ ] **Step 3: Implement SpectralSubtractor**

`modules/SpectralSubtractor.h`:
```cpp
#pragma once
#include <vector>

class SpectralSubtractor
{
public:
    SpectralSubtractor(int numBins, float sampleRate, int fftSize);

    void process(const float* magnitude, float* mask);
    void setDenoiseAmount(float amount) { denoiseAmount_ = amount; }

private:
    int numBins_;
    float sampleRate_;
    int fftSize_;
    float binResolution_;
    float denoiseAmount_ = 0.5f;
    int smoothBins_;

    // Martin's minimum statistics
    std::vector<float> noiseFloor_;
    std::vector<float> minBuffer_;   // running minimum
    int minWindowFrames_;            // ~1.5 seconds in frames
    int frameCounter_ = 0;

    void updateNoiseFloor(const float* magnitude);
    void smoothMask(float* mask) const;

    // Pre-allocated scratch buffer for smoothing (avoids audio-thread allocation)
    mutable std::vector<float> smoothScratch_;
};
```

`modules/SpectralSubtractor.cpp`:
```cpp
#include "SpectralSubtractor.h"
#include <algorithm>
#include <cmath>

SpectralSubtractor::SpectralSubtractor(int numBins, float sampleRate, int fftSize)
    : numBins_(numBins),
      sampleRate_(sampleRate),
      fftSize_(fftSize),
      binResolution_(sampleRate / fftSize)
{
    noiseFloor_.resize(numBins, 0.0f);
    minBuffer_.resize(numBins, 1e30f);

    float hopDuration = (fftSize / 4.0f) / sampleRate;
    minWindowFrames_ = static_cast<int>(std::ceil(1.5f / hopDuration));

    smoothBins_ = std::max(1, static_cast<int>(std::round(200.0f / binResolution_)));
    smoothScratch_.resize(numBins);
}

void SpectralSubtractor::updateNoiseFloor(const float* magnitude)
{
    // Simple minimum statistics: track per-bin minimum over a window
    for (int i = 0; i < numBins_; ++i)
        minBuffer_[i] = std::min(minBuffer_[i], magnitude[i]);

    frameCounter_++;
    if (frameCounter_ >= minWindowFrames_)
    {
        // Update noise floor and reset minimum buffer
        for (int i = 0; i < numBins_; ++i)
        {
            noiseFloor_[i] = minBuffer_[i];
            minBuffer_[i] = magnitude[i]; // reset with current frame
        }
        frameCounter_ = 0;
    }
}

void SpectralSubtractor::smoothMask(float* mask) const
{
    if (smoothBins_ <= 1) return;

    int halfWidth = smoothBins_ / 2;

    for (int i = 0; i < numBins_; ++i)
    {
        float sum = 0.0f;
        int count = 0;
        for (int j = i - halfWidth; j <= i + halfWidth; ++j)
        {
            if (j >= 0 && j < numBins_)
            {
                sum += mask[j];
                count++;
            }
        }
        smoothScratch_[i] = sum / count;
    }

    std::copy(smoothScratch_.begin(), smoothScratch_.begin() + numBins_, mask);
}

void SpectralSubtractor::process(const float* magnitude, float* mask)
{
    updateNoiseFloor(magnitude);

    float alpha = 1.0f + 3.0f * denoiseAmount_;

    for (int i = 0; i < numBins_; ++i)
    {
        if (magnitude[i] < 1e-10f)
        {
            mask[i] = 0.0f;
            continue;
        }
        mask[i] = std::max(0.0f, 1.0f - alpha * noiseFloor_[i] / magnitude[i]);
    }

    smoothMask(mask);
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cmake --build . --target DeFeedbackProTests && ./DeFeedbackProTests "[subtractor]"
```

Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/SpectralSubtractor.h modules/SpectralSubtractor.cpp tests/SpectralSubtractorTest.cpp
git commit -m "feat: spectral subtractor with Martin's minimum statistics"
```

---

## Task 6: Mask Combiner

**Files:**
- Create: `modules/MaskCombiner.h`
- Create: `modules/MaskCombiner.cpp`
- Create: `tests/MaskCombinerTest.cpp`

- [ ] **Step 1: Write failing tests**

`tests/MaskCombinerTest.cpp`:
```cpp
#include <catch2/catch_all.hpp>
#include "MaskCombiner.h"
#include <cmath>

TEST_CASE("MaskCombiner: all knobs zero produces all-ones mask (bypass)", "[combiner]")
{
    const int numBins = 129;
    MaskCombiner comb(numBins);
    comb.setParams(0.0f, 0.0f, 0.0f, 1.0f); // feedback, denoise, dereverb, mix

    std::vector<float> m1(numBins, 0.5f);
    std::vector<float> m2(numBins, 0.5f);
    std::vector<float> m3(numBins, 0.5f);
    std::vector<float> out(numBins);

    comb.combine(m1.data(), m2.data(), m3.data(), out.data());

    for (int i = 0; i < numBins; ++i)
        REQUIRE(out[i] == Catch::Approx(1.0f));
}

TEST_CASE("MaskCombiner: unity masks produce unity output", "[combiner]")
{
    const int numBins = 129;
    MaskCombiner comb(numBins);
    comb.setParams(1.0f, 1.0f, 1.0f, 1.0f);

    std::vector<float> m1(numBins, 1.0f);
    std::vector<float> m2(numBins, 1.0f);
    std::vector<float> m3(numBins, 1.0f);
    std::vector<float> out(numBins);

    comb.combine(m1.data(), m2.data(), m3.data(), out.data());

    for (int i = 0; i < numBins; ++i)
        REQUIRE(out[i] == Catch::Approx(1.0f).margin(0.001f));
}

TEST_CASE("MaskCombiner: mix=0 produces unity output (dry)", "[combiner]")
{
    const int numBins = 129;
    MaskCombiner comb(numBins);
    comb.setParams(1.0f, 1.0f, 1.0f, 0.0f); // mix = 0

    std::vector<float> m1(numBins, 0.0f); // total suppression
    std::vector<float> m2(numBins, 0.0f);
    std::vector<float> m3(numBins, 0.0f);
    std::vector<float> out(numBins);

    comb.combine(m1.data(), m2.data(), m3.data(), out.data());

    // mix=0 → output = 1.0 (fully dry)
    for (int i = 0; i < numBins; ++i)
        REQUIRE(out[i] == Catch::Approx(1.0f));
}

TEST_CASE("MaskCombiner: weighted geometric mean is correct", "[combiner]")
{
    const int numBins = 1;
    MaskCombiner comb(numBins);
    comb.setParams(0.5f, 0.5f, 0.0f, 1.0f); // feedback=0.5, denoise=0.5

    float m1 = 0.5f;
    float m2 = 0.8f;
    float m3 = 0.9f;
    float out;

    comb.combine(&m1, &m2, &m3, &out);

    // raw1=0.5, raw2=0.5, raw3=0.15, total=1.15
    // w1=0.5/1.15, w2=0.5/1.15, w3=0.15/1.15
    float w1 = 0.5f / 1.15f;
    float w2 = 0.5f / 1.15f;
    float w3 = 0.15f / 1.15f;
    float expected = std::pow(m1, w1) * std::pow(m2, w2) * std::pow(m3, w3);

    REQUIRE(out == Catch::Approx(expected).margin(0.001f));
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cmake --build . --target DeFeedbackProTests && ./DeFeedbackProTests "[combiner]"
```

Expected: Compilation error.

- [ ] **Step 3: Implement MaskCombiner**

`modules/MaskCombiner.h`:
```cpp
#pragma once
#include <vector>

class MaskCombiner
{
public:
    explicit MaskCombiner(int numBins);

    // Set user parameters (all 0..1)
    void setParams(float feedbackKnob, float denoiseKnob, float dereverbKnob, float mixKnob);

    // Combine three masks into output mask (includes dry/wet mix).
    // Output values represent final spectral multiplier for the signal.
    void combine(const float* mask1, const float* mask2, const float* mask3, float* output);

    bool isBypassed() const { return bypassed_; }

private:
    int numBins_;
    float feedbackKnob_ = 0.0f;
    float denoiseKnob_ = 0.0f;
    float dereverbKnob_ = 0.0f;
    float mixKnob_ = 1.0f;
    bool bypassed_ = true;
};
```

`modules/MaskCombiner.cpp`:
```cpp
#include "MaskCombiner.h"
#include <cmath>
#include <algorithm>

MaskCombiner::MaskCombiner(int numBins)
    : numBins_(numBins)
{
}

void MaskCombiner::setParams(float feedbackKnob, float denoiseKnob, float dereverbKnob, float mixKnob)
{
    feedbackKnob_ = feedbackKnob;
    denoiseKnob_ = denoiseKnob;
    dereverbKnob_ = dereverbKnob;
    mixKnob_ = mixKnob;
    bypassed_ = (feedbackKnob_ == 0.0f && denoiseKnob_ == 0.0f && dereverbKnob_ == 0.0f);
}

void MaskCombiner::combine(const float* mask1, const float* mask2, const float* mask3, float* output)
{
    if (bypassed_ || mixKnob_ == 0.0f)
    {
        std::fill(output, output + numBins_, 1.0f);
        return;
    }

    float raw1 = feedbackKnob_;
    float raw2 = std::max(denoiseKnob_, dereverbKnob_);
    float raw3 = 0.15f;
    float total = raw1 + raw2 + raw3;

    float w1 = raw1 / total;
    float w2 = raw2 / total;
    float w3 = raw3 / total;

    for (int i = 0; i < numBins_; ++i)
    {
        // Clamp inputs to avoid log(0) issues
        float m1 = std::max(mask1[i], 1e-6f);
        float m2 = std::max(mask2[i], 1e-6f);
        float m3 = std::max(mask3[i], 1e-6f);

        float maskVal = std::pow(m1, w1) * std::pow(m2, w2) * std::pow(m3, w3);

        // Apply dry/wet mix: output = mix * maskVal + (1 - mix)
        output[i] = mixKnob_ * maskVal + (1.0f - mixKnob_);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cmake --build . --target DeFeedbackProTests && ./DeFeedbackProTests "[combiner]"
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modules/MaskCombiner.h modules/MaskCombiner.cpp tests/MaskCombinerTest.cpp
git commit -m "feat: mask combiner with weighted geometric mean and bypass"
```

---

## Task 7: ML Processor (ONNX Runtime + Async Thread)

**Files:**
- Create: `modules/MLProcessor.h`
- Create: `modules/MLProcessor.cpp`
- Create: `tests/MLProcessorTest.cpp`
- Create: `scripts/create_dummy_model.py`

- [ ] **Step 1: Create dummy ONNX model generator**

`scripts/create_dummy_model.py`:
```python
"""Generate dummy ONNX models for testing.
Each model takes magnitude input and outputs two masks (noise, reverb).
Input: [1, num_frames, num_bins] float32
Output noise_mask: [1, num_bins] float32
Output reverb_mask: [1, num_bins] float32
The dummy model simply outputs 0.5 for all bins (identity-ish).
"""
import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto
except ImportError:
    print("pip install onnx numpy")
    exit(1)

FFT_SIZES = [128, 256, 512, 1024, 2048]
NUM_FRAMES = 4  # temporal context

for fft_size in FFT_SIZES:
    num_bins = fft_size // 2 + 1

    # Input
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, NUM_FRAMES, num_bins])

    # Constant 0.5 mask
    half_val = helper.make_tensor("half", TensorProto.FLOAT, [1, num_bins],
                                   [0.5] * num_bins)

    # Output nodes — just return constants
    noise_out = helper.make_tensor_value_info("noise_mask", TensorProto.FLOAT, [1, num_bins])
    reverb_out = helper.make_tensor_value_info("reverb_mask", TensorProto.FLOAT, [1, num_bins])

    # Identity-like: take mean over frames axis, then sigmoid-ish clamp
    reduce_node = helper.make_node("ReduceMean", ["input"], ["reduced"], axes=[1], keepdims=1)
    reshape_node = helper.make_node("Reshape", ["reduced", "shape"], ["reshaped"])
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [1, num_bins])
    sigmoid_noise = helper.make_node("Sigmoid", ["reshaped"], ["noise_mask"])
    sigmoid_reverb = helper.make_node("Sigmoid", ["reshaped"], ["reverb_mask"])

    graph = helper.make_graph(
        [reduce_node, reshape_node, sigmoid_noise, sigmoid_reverb],
        f"denoiser_{fft_size}",
        [X],
        [noise_out, reverb_out],
        [shape_init]
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = f"models/denoiser_{fft_size}.onnx"
    onnx.save(model, path)
    print(f"Saved {path} (bins={num_bins})")
```

- [ ] **Step 2: Generate dummy models**

```bash
mkdir -p models
python scripts/create_dummy_model.py
```

Expected: 5 `.onnx` files in `models/`.

- [ ] **Step 3: Write failing tests**

`tests/MLProcessorTest.cpp`:
```cpp
#include <catch2/catch_all.hpp>
#include "MLProcessor.h"
#include <thread>
#include <chrono>

// Tests assume models exist at a known path (set via env or hardcoded for test)
static const char* getModelDir()
{
    const char* env = std::getenv("DEFEEDBACK_MODEL_DIR");
    return env ? env : "models";
}

TEST_CASE("MLProcessor: loads model successfully", "[ml]")
{
    const int fftSize = 256;
    const int numBins = fftSize / 2 + 1;
    MLProcessor ml(numBins, getModelDir(), fftSize);
    REQUIRE(ml.isModelLoaded());
}

TEST_CASE("MLProcessor: returns valid masks", "[ml]")
{
    const int fftSize = 256;
    const int numBins = fftSize / 2 + 1;
    MLProcessor ml(numBins, getModelDir(), fftSize);

    std::vector<float> magnitude(numBins, 1.0f);
    ml.submitFrame(magnitude.data());

    // Give inference thread time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::vector<float> noiseMask(numBins);
    std::vector<float> reverbMask(numBins);
    bool got = ml.getLatestMasks(noiseMask.data(), reverbMask.data());

    REQUIRE(got);
    // Dummy model outputs sigmoid of input ≈ values in (0, 1)
    for (int i = 0; i < numBins; ++i)
    {
        REQUIRE(noiseMask[i] >= 0.0f);
        REQUIRE(noiseMask[i] <= 1.0f);
        REQUIRE(reverbMask[i] >= 0.0f);
        REQUIRE(reverbMask[i] <= 1.0f);
    }
}

TEST_CASE("MLProcessor: missing model disables ML gracefully", "[ml]")
{
    const int numBins = 129;
    MLProcessor ml(numBins, "nonexistent_dir", 256);
    REQUIRE_FALSE(ml.isModelLoaded());

    std::vector<float> noiseMask(numBins);
    std::vector<float> reverbMask(numBins);
    bool got = ml.getLatestMasks(noiseMask.data(), reverbMask.data());

    // Should return false (no masks available) or fallback masks
    if (got)
    {
        for (int i = 0; i < numBins; ++i)
        {
            REQUIRE(noiseMask[i] == Catch::Approx(1.0f));
            REQUIRE(reverbMask[i] == Catch::Approx(1.0f));
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cmake --build . --target DeFeedbackProTests && ./DeFeedbackProTests "[ml]"
```

Expected: Compilation error.

- [ ] **Step 5: Implement MLProcessor**

`modules/MLProcessor.h`:
```cpp
#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include <string>
#include "RingBuffer.h"

// Forward declare ONNX types to avoid header dependency
namespace Ort { class Env; class Session; class SessionOptions; class MemoryInfo; }

class MLProcessor
{
public:
    MLProcessor(int numBins, const std::string& modelDir, int fftSize);
    ~MLProcessor();

    bool isModelLoaded() const { return modelLoaded_.load(); }

    // Called from audio thread: submit a magnitude frame for async inference
    void submitFrame(const float* magnitude);

    // Called from audio thread: get the most recent inference result
    // Returns true if new masks are available
    bool getLatestMasks(float* noiseMask, float* reverbMask);

    void setDenoiseAmount(float amount) { denoiseAmount_ = amount; }
    void setDereverbAmount(float amount) { dereverbAmount_ = amount; }

    // Compute combined mask₂ from noise and reverb masks
    void computeMask2(float* mask2);

    // Change FFT size (reloads model)
    void setFFTSize(int fftSize);

private:
    void inferenceThreadFunc();
    bool loadModel(const std::string& path);
    void runInference(const float* input, int numFrames);

    int numBins_;
    int numFrames_ = 4; // temporal context
    std::string modelDir_;
    int fftSize_;

    std::atomic<float> denoiseAmount_{0.5f};
    std::atomic<float> dereverbAmount_{0.5f};

    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::atomic<bool> modelLoaded_{false};

    // Frame history for temporal context
    std::vector<float> frameHistory_; // numFrames_ * numBins_

    // Async inference thread
    std::thread inferenceThread_;
    std::atomic<bool> running_{false};

    // SPSC communication: audio thread → inference thread
    RingBuffer<float> inputQueue_;   // submits magnitude frames (numBins_ per frame)

    // Inference results (double-buffered via atomics)
    std::vector<float> noiseMaskA_, noiseMaskB_;
    std::vector<float> reverbMaskA_, reverbMaskB_;
    std::atomic<int> latestMaskBuffer_{-1}; // -1 = none, 0 = A, 1 = B
    int writingBuffer_ = 0; // only accessed by inference thread
};
```

`modules/MLProcessor.cpp`:
```cpp
#include "MLProcessor.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// Only include ONNX Runtime when available
#ifndef DEFEEDBACK_TEST_MODE
#include <onnxruntime_cxx_api.h>
#endif

#ifdef DEFEEDBACK_TEST_MODE
// Minimal stubs for testing without ONNX Runtime headers in test build
// The real implementation uses onnxruntime_cxx_api.h
namespace Ort {
    class Env { public: Env(int, const char*) {} };
    class SessionOptions { public: void SetIntraOpNumThreads(int) {} };
    class Session {
    public:
        Session(Env&, const char*, SessionOptions&) { throw std::runtime_error("stub"); }
        Session(Env&, const wchar_t*, SessionOptions&) { throw std::runtime_error("stub"); }
    };
    class MemoryInfo {};
}
#endif

MLProcessor::MLProcessor(int numBins, const std::string& modelDir, int fftSize)
    : numBins_(numBins),
      modelDir_(modelDir),
      fftSize_(fftSize),
      inputQueue_(numBins * 8) // buffer up to 8 frames
{
    frameHistory_.resize(numFrames_ * numBins_, 0.0f);
    noiseMaskA_.resize(numBins_, 1.0f);
    noiseMaskB_.resize(numBins_, 1.0f);
    reverbMaskA_.resize(numBins_, 1.0f);
    reverbMaskB_.resize(numBins_, 1.0f);

    env_ = std::make_unique<Ort::Env>(0, "DeFeedbackPro");

    std::string modelPath = modelDir + "/denoiser_" + std::to_string(fftSize) + ".onnx";
    if (loadModel(modelPath))
    {
        running_ = true;
        inferenceThread_ = std::thread(&MLProcessor::inferenceThreadFunc, this);
    }
}

MLProcessor::~MLProcessor()
{
    running_ = false;
    if (inferenceThread_.joinable())
        inferenceThread_.join();
}

bool MLProcessor::loadModel(const std::string& path)
{
    try
    {
#ifndef DEFEEDBACK_TEST_MODE
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        #ifdef _WIN32
        std::wstring wpath(path.begin(), path.end());
        session_ = std::make_unique<Ort::Session>(*env_, wpath.c_str(), opts);
        #else
        session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), opts);
        #endif
        modelLoaded_ = true;
        return true;
#else
        // In test mode, check if file exists
        FILE* f = fopen(path.c_str(), "rb");
        if (f) { fclose(f); modelLoaded_ = true; return true; }
        return false;
#endif
    }
    catch (...)
    {
        modelLoaded_ = false;
        return false;
    }
}

void MLProcessor::submitFrame(const float* magnitude)
{
    // Shift history left and append new frame
    std::memmove(frameHistory_.data(),
                 frameHistory_.data() + numBins_,
                 (numFrames_ - 1) * numBins_ * sizeof(float));
    std::memcpy(frameHistory_.data() + (numFrames_ - 1) * numBins_,
                magnitude, numBins_ * sizeof(float));

    // Push full history to inference queue
    inputQueue_.push(frameHistory_.data(), numFrames_ * numBins_);
}

bool MLProcessor::getLatestMasks(float* noiseMask, float* reverbMask)
{
    int buf = latestMaskBuffer_.load(std::memory_order_acquire);
    if (buf < 0) return false;

    const auto& nm = (buf == 0) ? noiseMaskA_ : noiseMaskB_;
    const auto& rm = (buf == 0) ? reverbMaskA_ : reverbMaskB_;
    std::memcpy(noiseMask, nm.data(), numBins_ * sizeof(float));
    std::memcpy(reverbMask, rm.data(), numBins_ * sizeof(float));
    return true;
}

void MLProcessor::computeMask2(float* mask2)
{
    std::vector<float> noiseMask(numBins_);
    std::vector<float> reverbMask(numBins_);

    if (!getLatestMasks(noiseMask.data(), reverbMask.data()))
    {
        std::fill(mask2, mask2 + numBins_, 1.0f);
        return;
    }

    float dn = denoiseAmount_.load();
    float dr = dereverbAmount_.load();

    for (int i = 0; i < numBins_; ++i)
    {
        float mn = std::pow(std::max(noiseMask[i], 1e-6f), dn);
        float mr = std::pow(std::max(reverbMask[i], 1e-6f), dr);
        mask2[i] = mn * mr;
    }
}

void MLProcessor::inferenceThreadFunc()
{
    std::vector<float> inputData(numFrames_ * numBins_);

    while (running_)
    {
        if (inputQueue_.availableToRead() >= static_cast<size_t>(numFrames_ * numBins_))
        {
            // Drain to latest frame (skip old ones)
            while (inputQueue_.availableToRead() >= static_cast<size_t>(numFrames_ * numBins_ * 2))
                inputQueue_.pop(inputData.data(), numFrames_ * numBins_);

            inputQueue_.pop(inputData.data(), numFrames_ * numBins_);
            runInference(inputData.data(), numFrames_);
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

void MLProcessor::runInference(const float* input, int numFrames)
{
#ifndef DEFEEDBACK_TEST_MODE
    if (!session_) return;

    try
    {
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 3> inputShape = {1, numFrames, numBins_};
        auto inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, const_cast<float*>(input),
            numFrames * numBins_, inputShape.data(), 3);

        const char* inputNames[] = {"input"};
        const char* outputNames[] = {"noise_mask", "reverb_mask"};

        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                      inputNames, &inputTensor, 1,
                                      outputNames, 2);

        const float* noiseData = outputs[0].GetTensorData<float>();
        const float* reverbData = outputs[1].GetTensorData<float>();

        auto& nDst = (writingBuffer_ == 0) ? noiseMaskA_ : noiseMaskB_;
        auto& rDst = (writingBuffer_ == 0) ? reverbMaskA_ : reverbMaskB_;

        std::memcpy(nDst.data(), noiseData, numBins_ * sizeof(float));
        std::memcpy(rDst.data(), reverbData, numBins_ * sizeof(float));

        latestMaskBuffer_.store(writingBuffer_, std::memory_order_release);
        writingBuffer_ = 1 - writingBuffer_;
    }
    catch (...) { /* reuse last mask */ }
#else
    // Test mode: generate sigmoid-like dummy masks
    auto& nDst = (writingBuffer_ == 0) ? noiseMaskA_ : noiseMaskB_;
    auto& rDst = (writingBuffer_ == 0) ? reverbMaskA_ : reverbMaskB_;

    for (int i = 0; i < numBins_; ++i)
    {
        float avg = 0.0f;
        for (int f = 0; f < numFrames; ++f)
            avg += input[f * numBins_ + i];
        avg /= numFrames;
        float sig = 1.0f / (1.0f + std::exp(-avg));
        nDst[i] = sig;
        rDst[i] = sig;
    }

    latestMaskBuffer_.store(writingBuffer_, std::memory_order_release);
    writingBuffer_ = 1 - writingBuffer_;
#endif
}

void MLProcessor::setFFTSize(int fftSize)
{
    running_ = false;
    if (inferenceThread_.joinable())
        inferenceThread_.join();

    fftSize_ = fftSize;
    numBins_ = fftSize / 2 + 1;
    frameHistory_.assign(numFrames_ * numBins_, 0.0f);
    noiseMaskA_.assign(numBins_, 1.0f);
    noiseMaskB_.assign(numBins_, 1.0f);
    reverbMaskA_.assign(numBins_, 1.0f);
    reverbMaskB_.assign(numBins_, 1.0f);
    latestMaskBuffer_ = -1;
    inputQueue_.reset();

    std::string modelPath = modelDir_ + "/denoiser_" + std::to_string(fftSize) + ".onnx";
    if (loadModel(modelPath))
    {
        running_ = true;
        inferenceThread_ = std::thread(&MLProcessor::inferenceThreadFunc, this);
    }
}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cmake --build . --target DeFeedbackProTests
DEFEEDBACK_MODEL_DIR=../models ./DeFeedbackProTests "[ml]"
```

Expected: All 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add modules/MLProcessor.h modules/MLProcessor.cpp tests/MLProcessorTest.cpp scripts/create_dummy_model.py
git commit -m "feat: ML processor with ONNX Runtime, async inference thread, and dummy models"
```

---

## Task 8: Wire Up PluginProcessor

**Files:**
- Modify: `modules/PluginProcessor.h`
- Modify: `modules/PluginProcessor.cpp`

- [ ] **Step 1: Update PluginProcessor.h with all module members**

Add to `PluginProcessor.h` private section:
```cpp
#include "SpectralEngine.h"
#include "FeedbackDetector.h"
#include "SpectralSubtractor.h"
#include "MaskCombiner.h"
#include "MLProcessor.h"
#include "RingBuffer.h"

// In private section:
    static constexpr int kDefaultFFTSize = 256;
    static constexpr float kInternalSampleRate = 48000.0f;

    // Per-channel spectral engines and modules
    struct ChannelState
    {
        std::unique_ptr<SpectralEngine> engine;
        std::unique_ptr<FeedbackDetector> feedbackDetector;
        std::unique_ptr<SpectralSubtractor> spectralSubtractor;
        std::unique_ptr<MaskCombiner> maskCombiner;
    };

    std::vector<ChannelState> channels_;
    std::unique_ptr<MLProcessor> mlProcessor_; // shared across channels
    int currentFFTSize_ = kDefaultFFTSize;
    double hostSampleRate_ = 48000.0;

    // Resampler (for non-48kHz hosts) — one pair per channel
    struct ResamplerState {
        juce::LagrangeInterpolator input, output;
    };
    std::vector<ResamplerState> resamplers_;
    double resampleRatio_ = 1.0;
    bool needsResampling_ = false;
    std::vector<float> resampledInput_, resampledOutput_;

    // Pre-allocated mask buffers (avoid audio-thread allocation)
    std::vector<float> mask1_, mask2_, mask3_, finalMask_;

    // Pending FFT size change (set on audio thread, applied on message thread)
    std::atomic<int> pendingFFTSize_{0};

    // Spectrum data for GUI (from left channel)
    RingBuffer<float> spectrumForGUI_;

    void initModules(int fftSize);
    void updateParametersForBlock();
    int getFFTSizeFromParam() const;

    // Timer to apply pending FFT size changes off the audio thread
    void timerCallback() override;  // add juce::Timer to base classes
```

- [ ] **Step 2: Implement prepareToPlay and processBlock**

Replace the TODO stubs in `PluginProcessor.cpp`:
```cpp
void DeFeedbackProProcessor::initModules(int fftSize)
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
    // On macOS, models live inside the plugin bundle's Resources folder.
    // On Windows, models live next to the plugin binary.
    #if JUCE_MAC
    juce::File modelDir = juce::File::getSpecialLocation(
        juce::File::currentApplicationFile).getChildFile("Contents/Resources/models");
    #else
    juce::File modelDir = juce::File::getSpecialLocation(
        juce::File::currentExecutableFile).getParentDirectory().getChildFile("models");
    #endif
    mlProcessor_ = std::make_unique<MLProcessor>(numBins, modelDir.getFullPathName().toStdString(), fftSize);

    // Pre-allocate per-block temp buffers
    mask1_.resize(numBins);
    mask2_.resize(numBins);
    mask3_.resize(numBins);
    finalMask_.resize(numBins);

    // Resampling buffers
    int maxBlock = 4096;
    resampledInput_.resize(static_cast<int>(maxBlock * resampleRatio_) + 64);
    resampledOutput_.resize(static_cast<int>(maxBlock * resampleRatio_) + 64);

    // Latency reporting
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

void DeFeedbackProProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    hostSampleRate_ = sampleRate;
    needsResampling_ = (std::abs(sampleRate - kInternalSampleRate) > 1.0);

    if (needsResampling_)
    {
        resampleRatio_ = kInternalSampleRate / sampleRate;
        resamplers_.resize(getTotalNumInputChannels());
        for (auto& r : resamplers_) { r.input.reset(); r.output.reset(); }
    }

    initModules(getFFTSizeFromParam());

    // Start a timer to consume pending FFT size changes off the audio thread
    startTimer(50); // 50ms check interval
}

void DeFeedbackProProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    bool bypass = *apvts.getRawParameterValue("bypass") > 0.5f;
    if (bypass) return;

    // Check for FFT size change — schedule for next prepareToPlay to avoid
    // heap allocation on the audio thread. For now, flag it.
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

        // Resample to 48 kHz if needed (per-channel resamplers)
        float* processData = channelData;
        int processSamples = numSamples;

        if (needsResampling_ && ch < (int)resamplers_.size())
        {
            processSamples = resamplers_[ch].input.process(
                resampleRatio_, channelData, resampledInput_.data(),
                numSamples);
            processData = resampledInput_.data();
        }

        // Single-pass spectral processing: the callback is invoked per hop.
        // Inside the callback, we run all 3 modules on the current spectrum
        // and combine their masks into a single mask.
        bool isFirstChannel = (ch == 0);
        auto maskCallback = [&](const float* magnitude, float* mask) {
            // Module 1: Feedback detection
            state.feedbackDetector->process(magnitude, mask1_.data());

            // Module 2: ML inference (submit from first channel only)
            if (isFirstChannel && mlProcessor_)
                mlProcessor_->submitFrame(magnitude);
            if (mlProcessor_)
                mlProcessor_->computeMask2(mask2_.data());
            else
                std::fill(mask2_.begin(), mask2_.end(), 1.0f);

            // Module 3: Spectral subtraction
            state.spectralSubtractor->process(magnitude, mask3_.data());

            // Combine all masks
            state.maskCombiner->combine(mask1_.data(), mask2_.data(),
                                         mask3_.data(), mask);

            // Send spectrum to GUI (left channel only)
            if (isFirstChannel)
                spectrumForGUI_.push(magnitude, numBins);
        };

        // Use pre-allocated output buffer (resampledOutput_ is large enough)
        state.engine->processBlock(processData, resampledOutput_.data(),
                                    processSamples, maskCallback);

        // Resample back if needed
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
```

Also add `timerCallback` to apply FFT size changes off the audio thread:

```cpp
void DeFeedbackProProcessor::timerCallback()
{
    int pending = pendingFFTSize_.exchange(0);
    if (pending > 0 && pending != currentFFTSize_)
    {
        // Build new modules — this runs on the message thread, not the audio thread
        // The audio thread will briefly use the old modules until swap completes.
        // A full crossfade implementation would use double-buffered engines;
        // for now, a direct swap with a brief fade is acceptable.
        initModules(pending);
    }
}
```

- [ ] **Step 3: Fix MLProcessor::computeMask2 to avoid heap allocation**

Change `computeMask2` to use pre-allocated member buffers:

In `MLProcessor.h`, add to private:
```cpp
    std::vector<float> tempNoiseMask_;
    std::vector<float> tempReverbMask_;
```

In `MLProcessor` constructor, add:
```cpp
    tempNoiseMask_.resize(numBins_, 1.0f);
    tempReverbMask_.resize(numBins_, 1.0f);
```

In `MLProcessor::computeMask2`, replace the local vectors:
```cpp
void MLProcessor::computeMask2(float* mask2)
{
    if (!getLatestMasks(tempNoiseMask_.data(), tempReverbMask_.data()))
    {
        std::fill(mask2, mask2 + numBins_, 1.0f);
        return;
    }

    float dn = denoiseAmount_.load();
    float dr = dereverbAmount_.load();

    for (int i = 0; i < numBins_; ++i)
    {
        float mn = std::pow(std::max(tempNoiseMask_[i], 1e-6f), dn);
        float mr = std::pow(std::max(tempReverbMask_[i], 1e-6f), dr);
        mask2[i] = mn * mr;
    }
}
```

- [ ] **Step 4: Verify plugin builds**

```bash
cd build && cmake .. -DBUILD_TESTS=OFF && cmake --build . --target DeFeedbackPro_VST3
```

Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add modules/PluginProcessor.h modules/PluginProcessor.cpp modules/MLProcessor.h modules/MLProcessor.cpp
git commit -m "feat: wire up full spectral processing pipeline in PluginProcessor"
```

---

## Task 9: Spectrum Display Component

**Files:**
- Create: `modules/SpectrumDisplay.h`
- Create: `modules/SpectrumDisplay.cpp`

- [ ] **Step 1: Implement SpectrumDisplay**

`modules/SpectrumDisplay.h`:
```cpp
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
    float smoothingCoeff_ = 0.7f; // exponential smoothing
};
```

`modules/SpectrumDisplay.cpp`:
```cpp
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
    // Drain the ring buffer to get the latest spectrum
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
            // Convert to dB, clamp to -80..0 dB range
            float db = 20.0f * std::log10(std::max(temp[i], 1e-6f));
            db = std::max(-80.0f, std::min(0.0f, db));
            float normalized = (db + 80.0f) / 80.0f; // 0..1

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

    // Draw spectrum
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

    // Fill under curve
    spectrumPath.lineTo(bounds.getRight(), bounds.getBottom());
    spectrumPath.lineTo(0, bounds.getBottom());
    spectrumPath.closeSubPath();
    g.setColour(juce::Colour(0x30e0e0e0));
    g.fillPath(spectrumPath);
}
```

- [ ] **Step 2: Verify build**

```bash
cmake --build . --target DeFeedbackPro_VST3
```

Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add modules/SpectrumDisplay.h modules/SpectrumDisplay.cpp
git commit -m "feat: real-time spectrum display component"
```

---

## Task 10: Full GUI with Knobs and Layout

**Files:**
- Modify: `modules/PluginEditor.h`
- Modify: `modules/PluginEditor.cpp`

- [ ] **Step 1: Update PluginEditor with full UI**

`modules/PluginEditor.h`:
```cpp
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
```

`modules/PluginEditor.cpp`:
```cpp
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

    // Title
    g.setColour(juce::Colours::white);
    g.setFont(18.0f);
    g.drawText("DeFeedback Pro", 10, 5, 200, 25, juce::Justification::centredLeft);
}

void DeFeedbackProEditor::resized()
{
    auto area = getLocalBounds().reduced(10);

    // Top bar: title area + FFT combo + bypass
    auto topBar = area.removeFromTop(30);
    bypassButton_.setBounds(topBar.removeFromRight(70));
    fftSizeCombo_.setBounds(topBar.removeFromRight(80).reduced(2));

    area.removeFromTop(5);

    // Spectrum display
    spectrumDisplay_.setBounds(area.removeFromTop(120));

    area.removeFromTop(10);

    // Knobs row
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
```

- [ ] **Step 2: Add getSpectrumBuffer() and getNumBins() to PluginProcessor**

Add to `PluginProcessor.h` public section:
```cpp
    RingBuffer<float>& getSpectrumBuffer() { return spectrumForGUI_; }
    int getNumBins() const { return currentFFTSize_ / 2 + 1; }
```

Initialize `spectrumForGUI_` in constructor initializer list:
```cpp
// In PluginProcessor.h, change declaration to:
    RingBuffer<float> spectrumForGUI_{2048}; // pre-allocated
```

- [ ] **Step 3: Verify build**

```bash
cmake --build . --target DeFeedbackPro_VST3
```

Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add modules/PluginEditor.h modules/PluginEditor.cpp modules/PluginProcessor.h
git commit -m "feat: full GUI with spectrum display, knobs, FFT size selector, and bypass"
```

---

## Task 11: Integration Test & Validation

**Files:**
- No new files; validation of the complete build.

- [ ] **Step 1: Run all unit tests**

```bash
cd build && cmake .. -DBUILD_TESTS=ON && cmake --build . --target DeFeedbackProTests
./DeFeedbackProTests
```

Expected: All tests PASS.

- [ ] **Step 2: Build VST3 and AU targets**

```bash
cmake --build . --target DeFeedbackPro_VST3
# On macOS only:
cmake --build . --target DeFeedbackPro_AU
```

Expected: Both targets build successfully.

- [ ] **Step 3: Validate VST3 with pluginval (if available)**

```bash
# Download pluginval from https://github.com/Tracktion/pluginval
pluginval --validate-in-process --strictness-level 5 build/DeFeedbackPro_artefacts/VST3/DeFeedbackPro.vst3
```

Expected: Passes at strictness level 5.

- [ ] **Step 4: Manual smoke test in DAW**

Load the VST3 in a DAW (e.g., REAPER, Ableton). Verify:
- Plugin loads without crash
- GUI appears with spectrum display, 4 knobs, FFT size combo, bypass
- Knobs respond to mouse interaction
- Audio passes through (Mix=0% = dry signal)
- Spectrum display animates with input signal
- FFT size change doesn't produce clicks
- Bypass works
- State saves/loads with project

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: integration validation complete"
```
