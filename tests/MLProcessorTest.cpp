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

TEST_CASE("MLProcessor: loads nsnet2 model successfully", "[ml]")
{
    const int fftSize = 256;
    const int numBins = fftSize / 2 + 1;
    MLProcessor ml(numBins, getModelDir(), fftSize, 48000.0f);
    REQUIRE(ml.isModelLoaded());
}

TEST_CASE("MLProcessor: returns valid masks", "[ml]")
{
    const int fftSize = 256;
    const int numBins = fftSize / 2 + 1;
    MLProcessor ml(numBins, getModelDir(), fftSize, 48000.0f);

    std::vector<float> magnitude(numBins, 1.0f);
    ml.submitFrame(magnitude.data());

    // Give inference thread time to process
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::vector<float> noiseMask(numBins);
    std::vector<float> reverbMask(numBins);
    bool got = ml.getLatestMasks(noiseMask.data(), reverbMask.data());

    REQUIRE(got);
    // Model outputs values in (0, 1) range (sigmoid output)
    for (int i = 0; i < numBins; ++i)
    {
        REQUIRE(noiseMask[i] >= 0.0f);
        REQUIRE(noiseMask[i] <= 1.0f);
        REQUIRE(reverbMask[i] >= 0.0f);
        REQUIRE(reverbMask[i] <= 1.0f);
    }
}

TEST_CASE("MLProcessor: works with different FFT sizes", "[ml]")
{
    // NSNet2 model is FFT-size independent via spectral resampling
    for (int fftSize : {128, 512, 1024, 2048})
    {
        SECTION("FFT size = " + std::to_string(fftSize))
        {
            const int numBins = fftSize / 2 + 1;
            MLProcessor ml(numBins, getModelDir(), fftSize, 48000.0f);
            REQUIRE(ml.isModelLoaded());

            std::vector<float> mag(numBins, 0.5f);
            ml.submitFrame(mag.data());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            std::vector<float> mask2(numBins);
            ml.computeMask2(mask2.data());
            for (int i = 0; i < numBins; ++i)
            {
                REQUIRE(mask2[i] >= 0.0f);
                REQUIRE(mask2[i] <= 1.0f);
            }
        }
    }
}

TEST_CASE("MLProcessor: missing model disables ML gracefully", "[ml]")
{
    const int numBins = 129;
    MLProcessor ml(numBins, "nonexistent_dir", 256, 48000.0f);
    REQUIRE_FALSE(ml.isModelLoaded());

    std::vector<float> noiseMask(numBins);
    std::vector<float> reverbMask(numBins);
    bool got = ml.getLatestMasks(noiseMask.data(), reverbMask.data());

    if (got)
    {
        for (int i = 0; i < numBins; ++i)
        {
            REQUIRE(noiseMask[i] == Catch::Approx(1.0f));
            REQUIRE(reverbMask[i] == Catch::Approx(1.0f));
        }
    }
}

TEST_CASE("MLProcessor: computeMask2 respects denoise/dereverb amounts", "[ml]")
{
    const int numBins = 129;
    MLProcessor ml(numBins, getModelDir(), 256, 48000.0f);

    std::vector<float> mag(numBins, 1.0f);
    ml.submitFrame(mag.data());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // With amount=0, mask should be 1.0 (no suppression)
    ml.setDenoiseAmount(0.0f);
    ml.setDereverbAmount(0.0f);
    std::vector<float> mask2(numBins);
    ml.computeMask2(mask2.data());
    for (int i = 0; i < numBins; ++i)
        REQUIRE(mask2[i] == Catch::Approx(1.0f).margin(0.01f));
}
