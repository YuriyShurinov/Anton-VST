#include "MaskCombiner.h"
#include <cmath>
#include <algorithm>

MaskCombiner::MaskCombiner(int numBins) : numBins_(numBins) {}

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
        float m1 = std::max(mask1[i], 1e-6f);
        float m2 = std::max(mask2[i], 1e-6f);
        float m3 = std::max(mask3[i], 1e-6f);
        float maskVal = std::pow(m1, w1) * std::pow(m2, w2) * std::pow(m3, w3);
        output[i] = mixKnob_ * maskVal + (1.0f - mixKnob_);
    }
}
