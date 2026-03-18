#include <catch2/catch_all.hpp>
#include "SpectralSubtractor.h"

TEST_CASE("SpectralSubtractor: clean signal produces mask near 1.0", "[subtractor]")
{
    const int numBins = 129;
    SpectralSubtractor sub(numBins, 48000.0f, 256);
    sub.setDenoiseAmount(0.5f);
    std::vector<float> noiseMag(numBins, 0.01f);
    std::vector<float> mask(numBins);
    for (int i = 0; i < 200; ++i)
        sub.process(noiseMag.data(), mask.data());
    std::vector<float> signalMag(numBins, 1.0f);
    sub.process(signalMag.data(), mask.data());
    for (int i = 0; i < numBins; ++i)
        REQUIRE(mask[i] > 0.9f);
}

TEST_CASE("SpectralSubtractor: noise-level signal produces mask near 0", "[subtractor]")
{
    const int numBins = 129;
    SpectralSubtractor sub(numBins, 48000.0f, 256);
    sub.setDenoiseAmount(1.0f);
    std::vector<float> noiseMag(numBins, 0.5f);
    std::vector<float> mask(numBins);
    for (int i = 0; i < 200; ++i)
        sub.process(noiseMag.data(), mask.data());
    sub.process(noiseMag.data(), mask.data());
    for (int i = 0; i < numBins; ++i)
        REQUIRE(mask[i] < 0.2f);
}

TEST_CASE("SpectralSubtractor: denoise amount 0 with loud signal", "[subtractor]")
{
    const int numBins = 129;
    SpectralSubtractor sub(numBins, 48000.0f, 256);
    sub.setDenoiseAmount(0.0f);
    std::vector<float> mag(numBins, 0.5f);
    std::vector<float> mask(numBins);
    for (int i = 0; i < 200; ++i)
        sub.process(mag.data(), mask.data());
    std::vector<float> signalMag(numBins, 2.0f);
    sub.process(signalMag.data(), mask.data());
    for (int i = 0; i < numBins; ++i)
        REQUIRE(mask[i] > 0.5f);
}
