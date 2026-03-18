#pragma once
#include <vector>

class MaskCombiner
{
public:
    explicit MaskCombiner(int numBins);
    void setParams(float feedbackKnob, float denoiseKnob, float dereverbKnob, float mixKnob);
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
