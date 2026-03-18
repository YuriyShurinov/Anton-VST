#include <catch2/catch_all.hpp>
#include "MaskCombiner.h"
#include <cmath>

TEST_CASE("MaskCombiner: all knobs zero produces all-ones mask (bypass)", "[combiner]")
{
    const int numBins = 129;
    MaskCombiner comb(numBins);
    comb.setParams(0.0f, 0.0f, 0.0f, 1.0f);
    std::vector<float> m1(numBins, 0.5f), m2(numBins, 0.5f), m3(numBins, 0.5f), out(numBins);
    comb.combine(m1.data(), m2.data(), m3.data(), out.data());
    for (int i = 0; i < numBins; ++i)
        REQUIRE(out[i] == Catch::Approx(1.0f));
}

TEST_CASE("MaskCombiner: unity masks produce unity output", "[combiner]")
{
    const int numBins = 129;
    MaskCombiner comb(numBins);
    comb.setParams(1.0f, 1.0f, 1.0f, 1.0f);
    std::vector<float> m1(numBins, 1.0f), m2(numBins, 1.0f), m3(numBins, 1.0f), out(numBins);
    comb.combine(m1.data(), m2.data(), m3.data(), out.data());
    for (int i = 0; i < numBins; ++i)
        REQUIRE(out[i] == Catch::Approx(1.0f).margin(0.001f));
}

TEST_CASE("MaskCombiner: mix=0 produces unity output (dry)", "[combiner]")
{
    const int numBins = 129;
    MaskCombiner comb(numBins);
    comb.setParams(1.0f, 1.0f, 1.0f, 0.0f);
    std::vector<float> m1(numBins, 0.0f), m2(numBins, 0.0f), m3(numBins, 0.0f), out(numBins);
    comb.combine(m1.data(), m2.data(), m3.data(), out.data());
    for (int i = 0; i < numBins; ++i)
        REQUIRE(out[i] == Catch::Approx(1.0f));
}

TEST_CASE("MaskCombiner: weighted geometric mean is correct", "[combiner]")
{
    const int numBins = 1;
    MaskCombiner comb(numBins);
    comb.setParams(0.5f, 0.5f, 0.0f, 1.0f);
    float m1 = 0.5f, m2 = 0.8f, m3 = 0.9f, out;
    comb.combine(&m1, &m2, &m3, &out);
    float w1 = 0.5f / 1.15f, w2 = 0.5f / 1.15f, w3 = 0.15f / 1.15f;
    float expected = std::pow(m1, w1) * std::pow(m2, w2) * std::pow(m3, w3);
    REQUIRE(out == Catch::Approx(expected).margin(0.001f));
}
