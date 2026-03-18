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
    // Dummy model outputs sigmoid of input ~ values in (0, 1)
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
