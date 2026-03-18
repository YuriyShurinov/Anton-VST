#pragma once

// MaskCombiner: Combines spectral masks from multiple processing stages.
// Full implementation added in a later task.
class MaskCombiner
{
public:
    MaskCombiner() = default;
    ~MaskCombiner() = default;

    void prepare(int fftSize);
    void reset();
};
