# DeFeedback Pro — VST Plugin Design Spec

## Overview

A real-time audio VST plugin that eliminates microphone feedback, reduces noise, and removes room reverberation using a unified spectral domain architecture. Combines classical DSP and ML inference in a single FFT/IFFT pipeline for minimal latency.

Comparable product: [Alpha Labs Audio DeFeedback](https://www.alphalabsaudio.com/defeedback/)

## Requirements

- **Formats:** VST3 + Audio Unit (AU)
- **Framework:** JUCE 7.x
- **Platforms:** Windows x64 (VST3), macOS universal binary x86_64 + arm64 (VST3 + AU)
- **Latency target:** < 5ms (configurable via FFT size)
- **Sample rates:** 44.1 kHz, 48 kHz (internal resampling to 48 kHz if host runs at different rate)
- **Channel config:** Mono and stereo. Stereo input is processed per-channel independently (separate spectral engine per channel, shared ML model weights).

## Architecture: Unified Spectral Domain

All processing happens in the frequency domain. One FFT transforms the input, three parallel analyzers produce spectral masks, masks are combined, and one IFFT reconstructs the output.

```
Input → Window → FFT → [Spectral Frame]
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
              Feedback    ML Model   Spectral
              Detector   (noise +   Subtraction
              (peak      dereverb)  (stationary
              tracking,             noise floor
              notch      → mask₂    estimation)
              masks)                → mask₃
              → mask₁
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                     Mask Combiner
                    (weighted geometric mean)
                              │
                              ▼
                    Combined Spectral Mask
                              │
                              ▼
                 Apply Mask → IFFT → Overlap-Add → Output
```

### FFT Configuration

| Parameter | Value |
|-----------|-------|
| FFT size | 128 / 256 / 512 / 1024 / 2048 (user-selectable) |
| Hop size | FFT_size / 4 (75% overlap) |
| Window | Hann |
| Algorithmic latency | hop_size / sample_rate = FFT_size / 4 / sample_rate (0.67ms — 10.7ms @ 48kHz) |
| Default FFT size | 256 |

When the user changes FFT size at runtime, the engine performs a crossfade transition:

1. A second SpectralEngine instance is pre-built with the new FFT size in the background.
2. Once ready, a 10ms linear crossfade blends the old engine's output into the new engine's output.
3. The old engine is deallocated after the crossfade completes.

This ensures glitch-free FFT size changes with no audio dropouts.

**Memory note:** During crossfade, two full engine instances coexist (4 engines total for stereo). Peak memory cost is ~2x normal. At FFT=2048 stereo, this is roughly 200-400 KB of additional buffers — acceptable for a desktop plugin.

## Processing Modules

### Module 1: Feedback Detector

Detects and suppresses tonal feedback frequencies (microphone → speaker → microphone resonance loops).

**Algorithm:**

1. **Peak Detection:** On each spectral frame, find bins where magnitude exceeds median by > 12 dB (peak-to-median ratio).
2. **Persistence Filter:** A peak is classified as feedback only if it persists for a configurable duration (default 50ms). Frame count is derived: `N = ceil(50ms / hop_duration)`. This adapts automatically to FFT size changes.
3. **Narrow Notch Mask:** For confirmed feedback frequencies, attenuate the peak bin and adjacent bins using a Gaussian taper. The notch width is defined in Hz (default ~100 Hz), converted to bin count: `width_bins = ceil(100 / bin_resolution)` where `bin_resolution = sample_rate / FFT_size`. This ensures consistent behavior across FFT sizes.
4. **Attack/Release Envelope:** Fast attack (2-3 frames) for instant suppression. Slow release (50-100 frames) for gradual recovery when feedback stops.

**Output:** `mask₁` — array of [0..1] gain values per frequency bin.

### Module 2: ML Denoiser / Dereverberator

Neural network that produces spectral masks for noise and reverberation removal.

**Architecture:**

- Based on DeepFilterNet: GRU-based recurrent network operating on spectral magnitudes.
- **Input:** Magnitude spectrum of current frame + 2-4 previous frames (temporal context).
- **Output:** Two gain masks — `mask_noise` [0..1] and `mask_reverb` [0..1] per frequency bin. These are combined into `mask₂` based on user-controlled Denoise and Dereverb knob weights.
- **Inference engine:** ONNX Runtime (CPU, AVX2 on x86, NEON on ARM).
- **Model size target:** 2-5 MB per FFT size variant.
- **Models:** One ONNX model per FFT size (5 total: denoiser_128.onnx through denoiser_2048.onnx). Each model's input/output dimensions match FFT_size/2 + 1 bins.

**mask₂ computation:**

```
mask₂[i] = mask_noise[i]^denoise_amount · mask_reverb[i]^dereverb_amount
```

Where `denoise_amount` and `dereverb_amount` are user knob values [0..1].

### Module 3: Spectral Subtraction

Classical statistical noise reduction as a safety net for the ML module.

**Algorithm:**

1. **Noise Floor Estimation:** Minimum statistics (Martin's algorithm) — tracks the minimum spectral magnitude over a sliding window of 1-2 seconds.
2. **Oversubtraction:** `mask₃[i] = max(0, 1 - α · noise_floor[i] / magnitude[i])` where α is linearly mapped from the Denoise knob: `α = 1.0 + 3.0 * denoise_amount` (range 1.0 to 4.0).
3. **Spectral Smoothing:** Apply frequency-domain smoothing with a width of ~200 Hz, converted to bins: `smooth_bins = max(1, round(200 / bin_resolution))`. This maintains consistent smoothing across FFT sizes.

**Output:** `mask₃` — array of [0..1] gain values per frequency bin.

## Mask Combiner

Combines the three masks using a weighted geometric mean:

```
mask_final[i] = mask₁[i]^w₁ · mask₂[i]^w₂ · mask₃[i]^w₃
```

Where `w₁ + w₂ + w₃ = 1`. Weights are derived from the user's knob settings:

```
raw₁ = feedback_knob       (0..1)
raw₂ = max(denoise_knob, dereverb_knob)  (0..1)
raw₃ = 0.15                (fixed safety net)

# If all knobs are zero, all raw values are zero → pass-through (see below)
total = raw₁ + raw₂ + raw₃
w₁ = raw₁ / total
w₂ = raw₂ / total
w₃ = raw₃ / total
```

**All-knobs-at-zero behavior:** When Feedback = 0, Denoise = 0, and Dereverb = 0, the all-zero check is performed **before** computing raw values. All three modules are bypassed regardless of Mix setting — the output equals the input (clean pass-through). The safety net (raw₃ = 0.15) only applies when at least one user knob is > 0.

**Double-influence note:** The Denoise knob affects both mask₂ generation (via `denoise_amount` exponent) and the combiner weight w₂. This is intentional — at low values the ML mask is both gentle and lightly weighted, at high values it is both aggressive and dominant. The response curve is smooth and predictable.

The final output is computed as:

```
output_spectrum[i] = input_spectrum[i] · (mix · mask_final[i] + (1 - mix))
```

Where `mix` is the Dry/Wet knob value.

## GUI

Window size: ~500 x 300 px. Minimal, functional design.

```
┌─────────────────────────────────────────────────────┐
│  DeFeedback Pro         [FFT: 256 ▾]       [bypass] │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────── Spectrum Display ──────────────────┐  │
│  │  Real-time magnitude spectrum (white/gray)     │  │
│  │  Detected feedback frequencies (red markers)   │  │
│  │  Applied mask curve (yellow, inverted)         │  │
│  │  Height: ~120px                                │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐  │
│  │Feedback │  │ Denoise │  │ Dereverb│  │  Mix   │  │
│  │   (o)   │  │   (o)   │  │   (o)   │  │  (o)   │  │
│  │  0-100% │  │  0-100% │  │  0-100% │  │ 0-100% │  │
│  └─────────┘  └─────────┘  └─────────┘  └────────┘  │
│                                                      │
│  In: ████████░░  Out: █████░░░░░          GR: -4dB   │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Parameters (automatable)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Feedback | 0 — 100% | 50% | Feedback suppression strength |
| Denoise | 0 — 100% | 50% | ML noise reduction amount |
| Dereverb | 0 — 100% | 50% | ML dereverberation amount |
| Mix | 0 — 100% | 100% | Dry/Wet balance |
| FFT Size | 128/256/512/1024/2048 | 256 | Spectral resolution (changes latency) |
| Bypass | on/off | off | Bypass all processing |

### Spectrum Display

- JUCE `Component` rendering at ~30 FPS via `Timer`
- Data transferred from audio thread via lock-free ring buffer (single-producer single-consumer)
- Draws: magnitude spectrum (log scale), feedback markers, mask overlay

## Project Structure

```
VST/
├── CMakeLists.txt
├── modules/
│   ├── PluginProcessor.cpp/h       # JUCE AudioProcessor
│   ├── PluginEditor.cpp/h          # JUCE AudioProcessorEditor
│   ├── SpectralEngine.cpp/h        # FFT/IFFT, windowing, overlap-add
│   ├── FeedbackDetector.cpp/h      # Module 1: peak tracking + notch masks
│   ├── MLProcessor.cpp/h           # Module 2: ONNX Runtime inference
│   ├── SpectralSubtractor.cpp/h    # Module 3: noise floor + subtraction
│   ├── MaskCombiner.cpp/h          # Weighted geometric mean of masks
│   └── SpectrumDisplay.cpp/h       # Real-time spectrum GUI component
├── models/
│   ├── denoiser_128.onnx
│   ├── denoiser_256.onnx
│   ├── denoiser_512.onnx
│   ├── denoiser_1024.onnx
│   └── denoiser_2048.onnx
└── resources/
```

## Dependencies

| Dependency | Purpose | Integration |
|------------|---------|-------------|
| JUCE 7.x | Framework, GUI, audio I/O | CMake `add_subdirectory` or `find_package` |
| ONNX Runtime 1.x | ML inference | Static linking, prebuilt libs |
| PFFFT | Fast FFT (SIMD-optimized) | Source inclusion (single .c/.h) |

## Thread Safety

- **Audio thread:** SpectralEngine + all 3 modules — fully lock-free. No allocations, no locks.
- **GUI thread:** Reads spectral data via atomic SPSC ring buffer. Timer-driven repaint at 30 FPS.
- **ML inference:** Runs on a dedicated real-time inference thread. The audio thread posts spectral frames to a lock-free SPSC queue; the inference thread processes them and writes mask results back via a second SPSC queue. The audio thread always has the latest completed mask available. If inference is slower than the hop rate, the audio thread reuses the most recent mask (graceful degradation with no glitches). This decouples ML latency spikes from the audio callback.
  - **Latency budget:** At FFT=128/48kHz, hop time is 0.67ms — too tight for synchronous ONNX inference. The async thread adds one hop of latency (mask is one frame behind) but guarantees no audio dropouts.
  - **At larger FFT sizes** (512+), hop time is 2.7ms+, and inference comfortably fits within budget.
  - **Thread priority:** The inference thread runs at elevated priority (below audio thread): `SetThreadPriority(THREAD_PRIORITY_ABOVE_NORMAL)` on Windows, `pthread_setschedparam` with `SCHED_OTHER` at priority 20 on macOS. This prevents inference starvation from background OS tasks.

## Sample Rate Handling

The ML models are trained at 48 kHz. All spectral processing operates at 48 kHz internally.

- If the host sample rate is 48 kHz: no resampling needed.
- If the host sample rate is 44.1 kHz (or any other rate): input is resampled to 48 kHz before the spectral engine, output is resampled back after overlap-add. Resampling uses JUCE's `LagrangeInterpolator` (low overhead, sufficient quality for this use case).
- **Resampler latency:** `LagrangeInterpolator` adds ~5 samples of group delay per direction (~10 samples round-trip). The plugin reports total latency to the host via `setLatencySamples(hop_size + resampler_delay)` so the DAW can compensate for mix alignment.
- The feedback detector thresholds, persistence timings, and noise floor estimation windows are all defined in absolute time units (ms, seconds) and derived to frame counts at runtime, so they adapt automatically.

## State Persistence

All automatable parameters are saved/restored via JUCE `getStateInformation` / `setStateInformation` using `AudioProcessorValueTreeState`. This includes: Feedback, Denoise, Dereverb, Mix, FFT Size, Bypass. The host can save and recall these with sessions and presets.

## Error Handling

- **ONNX model load failure** (missing/corrupt file): Module 2 (ML) is disabled. Modules 1 and 3 continue operating. A warning flag is exposed to the GUI (e.g., "ML Unavailable" indicator).
- **ONNX inference failure** (runtime error): Reuse last valid mask. If no valid mask exists yet, mask₂ = all 1.0 (no ML attenuation).

## Build System

CMake-based build. Targets:

- `DeFeedbackPro_VST3` — Windows + macOS
- `DeFeedbackPro_AU` — macOS only

ONNX Runtime linked statically to avoid runtime dependency. PFFFT compiled from source as part of the build.

## ML Model Training (Out of Scope for Initial Build)

The initial build will use a pretrained DeepFilterNet-inspired model. DeepFilterNet's native architecture uses a specific STFT configuration (480-sample frames at 48 kHz with custom band structure) that does not directly map to arbitrary FFT sizes. Therefore:

1. **Phase 1 (initial build):** Use a custom lightweight GRU model architecture (inspired by DeepFilterNet but simplified) that accepts standard FFT magnitude bins as input. Train one model per FFT size using publicly available noise/reverb datasets (DNS Challenge, WHAMR!). This requires a separate training pipeline (Python/PyTorch → ONNX export).
2. **Phase 2 (future):** Optionally adapt the full DeepFilterNet architecture or train larger models for improved quality.

The plugin architecture supports hot-swapping models by replacing .onnx files without recompilation.

## PFFFT Output Format

PFFFT uses a packed real FFT format different from standard complex output. The SpectralEngine must unpack PFFFT output into standard magnitude + phase arrays (FFT_size/2 + 1 complex bins) before passing data to the processing modules. This unpacking is handled internally in SpectralEngine and is transparent to the modules.
