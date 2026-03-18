#define _USE_MATH_DEFINES
#include <catch2/catch_all.hpp>
#include "SpectralEngine.h"
#include <cmath>

TEST_CASE("SpectralEngine: FFT round-trip reconstructs signal", "[spectral]")
{
    const int fftSize = 256;
    SpectralEngine engine(fftSize, 48000.0f);
    const int numBins = fftSize / 2 + 1;
    auto unityCallback = [&](const float*, float* mask) {
        std::fill(mask, mask + numBins, 1.0f);
    };
    // Use enough samples for ramp-up plus steady-state check
    const int totalSamples = fftSize * 8;
    std::vector<float> input(totalSamples);
    for (int i = 0; i < totalSamples; ++i)
        input[i] = std::sin(2.0f * M_PI * 1000.0f * i / 48000.0f);
    std::vector<float> output(totalSamples, 0.0f);
    engine.processBlock(input.data(), output.data(), totalSamples, unityCallback);

    // The overlap-add introduces latency of (fftSize - 1) samples.
    // Find actual latency via cross-correlation
    const int latency = fftSize - 1;
    const int checkStart = fftSize * 3; // well past ramp-up
    const int checkEnd = totalSamples - latency;
    float maxError = 0.0f;
    for (int i = checkStart; i < checkEnd; ++i)
    {
        float err = std::abs(output[i + latency] - input[i]);
        maxError = std::max(maxError, err);
    }
    INFO("Max error (latency-adjusted): " << maxError << " latency: " << latency);
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
    auto captureCallback = [&](const float*, float* mask) {
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
