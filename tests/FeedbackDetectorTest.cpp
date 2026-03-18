#include <catch2/catch_all.hpp>
#include "FeedbackDetector.h"
#include <cmath>

TEST_CASE("FeedbackDetector: no feedback in flat spectrum", "[feedback]")
{
    const int numBins = 129;
    FeedbackDetector det(numBins, 48000.0f, 256);
    std::vector<float> mag(numBins, 1.0f);
    std::vector<float> mask(numBins);
    det.process(mag.data(), mask.data());
    for (int i = 0; i < numBins; ++i)
        REQUIRE(mask[i] == Catch::Approx(1.0f));
}

TEST_CASE("FeedbackDetector: detects strong tonal peak after persistence", "[feedback]")
{
    const int numBins = 129;
    FeedbackDetector det(numBins, 48000.0f, 256);
    std::vector<float> mag(numBins, 0.1f);
    mag[50] = 10.0f;
    std::vector<float> mask(numBins);
    int persistFrames = det.getPersistenceFrames() + 5;
    for (int f = 0; f < persistFrames; ++f)
        det.process(mag.data(), mask.data());
    REQUIRE(mask[50] < 0.5f);
    REQUIRE(mask[0] == Catch::Approx(1.0f).margin(0.01f));
    REQUIRE(mask[numBins - 1] == Catch::Approx(1.0f).margin(0.01f));
}

TEST_CASE("FeedbackDetector: transient peak does NOT trigger suppression", "[feedback]")
{
    const int numBins = 129;
    FeedbackDetector det(numBins, 48000.0f, 256);
    std::vector<float> mag(numBins, 0.1f);
    std::vector<float> mask(numBins);
    mag[50] = 10.0f;
    det.process(mag.data(), mask.data());
    mag[50] = 0.1f;
    det.process(mag.data(), mask.data());
    REQUIRE(mask[50] == Catch::Approx(1.0f).margin(0.05f));
}
