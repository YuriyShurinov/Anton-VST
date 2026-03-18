"""
DeFeedback Pro — Training Dataset

Generates training pairs: (noisy_magnitude, clean_magnitude) from audio files.
Uses online mixing of clean speech + noise + optional room impulse response.

Data sources (download separately):
- Clean speech: DNS Challenge clean set, LibriSpeech, or any clean speech corpus
- Noise: DNS Challenge noise set, DEMAND, or any environmental noise corpus
- RIRs (optional): MIT IR Survey, OpenAIR, or simulated RIRs

Directory structure:
  data/
    clean/    - clean speech .wav files (16-bit, any sample rate)
    noise/    - noise .wav files
    rir/      - room impulse response .wav files (optional)
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf


class SpectralDataset(Dataset):
    """Online mixing dataset that generates spectral training pairs."""

    def __init__(self, clean_dir: str, noise_dir: str, rir_dir: str | None = None,
                 fft_size: int = 256, num_frames: int = 4, sample_rate: int = 48000,
                 segment_length: float = 2.0, snr_range: tuple = (-5, 20),
                 num_samples: int = 10000):
        self.clean_files = self._find_audio(clean_dir)
        self.noise_files = self._find_audio(noise_dir)
        self.rir_files = self._find_audio(rir_dir) if rir_dir else []

        self.fft_size = fft_size
        self.hop_size = fft_size // 4
        self.num_frames = num_frames
        self.num_bins = fft_size // 2 + 1
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.snr_range = snr_range
        self.num_samples = num_samples

        self.window = torch.hann_window(fft_size)

        if len(self.clean_files) == 0:
            raise ValueError(f"No audio files found in {clean_dir}")
        if len(self.noise_files) == 0:
            raise ValueError(f"No audio files found in {noise_dir}")

    @staticmethod
    def _find_audio(directory: str) -> list[str]:
        """Find all wav/flac files in directory."""
        if not directory or not os.path.isdir(directory):
            return []
        extensions = {'.wav', '.flac', '.ogg'}
        files = []
        for root, _, filenames in os.walk(directory):
            for f in filenames:
                if os.path.splitext(f)[1].lower() in extensions:
                    files.append(os.path.join(root, f))
        return sorted(files)

    def _load_random_segment(self, file_list: list[str]) -> np.ndarray:
        """Load a random segment from a random file, resampled to target SR."""
        path = random.choice(file_list)
        info = sf.info(path)
        total_samples = int(info.frames)

        # Calculate how many samples we need from the source file
        ratio = info.samplerate / self.sample_rate
        src_samples_needed = int(self.segment_samples * ratio)

        if total_samples <= src_samples_needed:
            audio, sr = sf.read(path, dtype='float32', always_2d=False)
        else:
            start = random.randint(0, total_samples - src_samples_needed)
            audio, sr = sf.read(path, start=start, stop=start + src_samples_needed,
                                dtype='float32', always_2d=False)

        # Mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed (simple linear interpolation)
        if sr != self.sample_rate:
            target_len = int(len(audio) * self.sample_rate / sr)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        # Pad or trim to exact length
        if len(audio) < self.segment_samples:
            audio = np.pad(audio, (0, self.segment_samples - len(audio)))
        else:
            audio = audio[:self.segment_samples]

        return audio

    def _apply_rir(self, audio: np.ndarray) -> np.ndarray:
        """Convolve with a random RIR for reverberation."""
        if not self.rir_files:
            return audio

        rir_path = random.choice(self.rir_files)
        rir, sr = sf.read(rir_path, dtype='float32', always_2d=False)
        if rir.ndim > 1:
            rir = rir[:, 0]

        # Resample RIR if needed
        if sr != self.sample_rate:
            target_len = int(len(rir) * self.sample_rate / sr)
            indices = np.linspace(0, len(rir) - 1, target_len)
            rir = np.interp(indices, np.arange(len(rir)), rir).astype(np.float32)

        # Normalize RIR
        rir = rir / (np.abs(rir).max() + 1e-8)

        # Convolve
        reverbed = np.convolve(audio, rir, mode='full')[:len(audio)]

        # Normalize to match original level
        scale = (np.sqrt(np.mean(audio ** 2)) + 1e-8) / (np.sqrt(np.mean(reverbed ** 2)) + 1e-8)
        return reverbed * scale

    def _mix_at_snr(self, clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
        """Mix clean signal with noise at given SNR."""
        clean_power = np.mean(clean ** 2) + 1e-8
        noise_power = np.mean(noise ** 2) + 1e-8
        target_noise_power = clean_power / (10 ** (snr_db / 10))
        scale = np.sqrt(target_noise_power / noise_power)
        return clean + noise * scale

    def _compute_stft_magnitude(self, audio: np.ndarray) -> torch.Tensor:
        """Compute STFT magnitude frames."""
        audio_t = torch.from_numpy(audio)
        stft = torch.stft(audio_t, n_fft=self.fft_size, hop_length=self.hop_size,
                          win_length=self.fft_size, window=self.window,
                          return_complex=True)
        # stft shape: [num_bins, num_time_frames]
        magnitude = stft.abs()
        return magnitude.T  # [num_time_frames, num_bins]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            noisy_frames: [num_frames, num_bins] — input magnitude context
            noise_mask: [num_bins] — ideal ratio mask for noise
            reverb_mask: [num_bins] — ideal ratio mask for reverb
        """
        # Load random clean and noise segments
        clean = self._load_random_segment(self.clean_files)
        noise = self._load_random_segment(self.noise_files)

        # Optionally apply reverberation to clean speech
        use_reverb = len(self.rir_files) > 0 and random.random() < 0.5
        if use_reverb:
            reverbed = self._apply_rir(clean)
        else:
            reverbed = clean.copy()

        # Mix at random SNR
        snr = random.uniform(*self.snr_range)
        noisy = self._mix_at_snr(reverbed, noise, snr)

        # Compute magnitudes
        clean_mag = self._compute_stft_magnitude(clean)       # [T, bins]
        noisy_mag = self._compute_stft_magnitude(noisy)       # [T, bins]
        reverbed_mag = self._compute_stft_magnitude(reverbed)  # [T, bins]

        # Pick a random frame position (need num_frames context)
        max_start = min(clean_mag.shape[0], noisy_mag.shape[0]) - self.num_frames
        if max_start < 1:
            max_start = 1
        start = random.randint(0, max_start - 1)

        # Input: noisy magnitude context
        noisy_frames = noisy_mag[start:start + self.num_frames]  # [num_frames, bins]

        # Target masks: Ideal Ratio Mask (IRM)
        # noise_mask = clean / noisy (how much to keep to remove noise)
        # reverb_mask = clean / reverbed (how much to keep to remove reverb)
        noisy_last = noisy_mag[start + self.num_frames - 1]    # [bins]
        clean_last = clean_mag[start + self.num_frames - 1]
        reverbed_last = reverbed_mag[start + self.num_frames - 1]

        eps = 1e-8
        noise_mask = torch.clamp(clean_last / (noisy_last + eps), 0.0, 1.0)
        reverb_mask = torch.clamp(clean_last / (reverbed_last + eps), 0.0, 1.0)

        return noisy_frames, noise_mask, reverb_mask
