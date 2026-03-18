# DeFeedback Pro — ML Model Training

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchaudio onnx onnxruntime numpy soundfile
```

### 2. Prepare Data

Download training data and organize:

```
data/
  clean/    ← Clean speech .wav files
  noise/    ← Environmental noise .wav files
  rir/      ← Room impulse responses (optional)
```

**Recommended datasets:**
- **Clean speech:** [DNS Challenge](https://github.com/microsoft/DNS-Challenge) clean set, or [LibriSpeech](https://www.openslr.org/12)
- **Noise:** [DNS Challenge](https://github.com/microsoft/DNS-Challenge) noise set, or [DEMAND](https://zenodo.org/record/1227121)
- **RIRs:** [MIT IR Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html), [OpenAIR](https://www.openair.hosted.york.ac.uk/)

### 3. Train

```bash
python training/train.py \
  --clean-dir data/clean \
  --noise-dir data/noise \
  --rir-dir data/rir \
  --output-dir models \
  --epochs 50 \
  --batch-size 32
```

Train a single FFT size for testing:
```bash
python training/train.py --clean-dir data/clean --noise-dir data/noise --fft-sizes 256 --epochs 10
```

### 4. Output

Trained ONNX models are saved to `models/`:
- `denoiser_128.onnx`
- `denoiser_256.onnx`
- `denoiser_512.onnx`
- `denoiser_1024.onnx`
- `denoiser_2048.onnx`

PyTorch checkpoints are saved to `models/checkpoints/`.

## Architecture

- **Input:** `[1, 4, num_bins]` — 4 frames of magnitude spectrum (log-compressed)
- **Model:** LayerNorm → Linear → 2-layer GRU → two separate Linear heads
- **Output:** `noise_mask [1, num_bins]` + `reverb_mask [1, num_bins]`, both in [0, 1]
- **Training target:** Ideal Ratio Mask (IRM) = clean / noisy

Model sizes:
| FFT Size | num_bins | hidden_size | Parameters | ONNX Size |
|----------|----------|-------------|------------|-----------|
| 128      | 65       | 128         | ~230K      | ~1 MB     |
| 256      | 129      | 128         | ~250K      | ~1 MB     |
| 512      | 257      | 192         | ~530K      | ~2 MB     |
| 1024     | 513      | 256         | ~1.2M      | ~5 MB     |
| 2048     | 1025     | 256         | ~1.5M      | ~6 MB     |
