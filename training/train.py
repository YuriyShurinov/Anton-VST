"""
DeFeedback Pro — Training Script

Trains spectral mask estimation models for all FFT sizes.
Exports trained models to ONNX format for plugin inference.

Usage:
    python training/train.py --clean-dir data/clean --noise-dir data/noise [--rir-dir data/rir]

The script trains one model per FFT size (128, 256, 512, 1024, 2048),
saves PyTorch checkpoints and exports ONNX models to models/ directory.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import create_model, get_model_config
from dataset import SpectralDataset


FFT_SIZES = [128, 256, 512, 1024, 2048]
NUM_FRAMES = 4


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for noisy_frames, target_noise_mask, target_reverb_mask in dataloader:
        noisy_frames = noisy_frames.to(device)
        target_noise_mask = target_noise_mask.to(device)
        target_reverb_mask = target_reverb_mask.to(device)

        optimizer.zero_grad()
        pred_noise_mask, pred_reverb_mask = model(noisy_frames)

        loss_noise = criterion(pred_noise_mask, target_noise_mask)
        loss_reverb = criterion(pred_reverb_mask, target_reverb_mask)
        loss = loss_noise + loss_reverb

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for noisy_frames, target_noise_mask, target_reverb_mask in dataloader:
        noisy_frames = noisy_frames.to(device)
        target_noise_mask = target_noise_mask.to(device)
        target_reverb_mask = target_reverb_mask.to(device)

        pred_noise_mask, pred_reverb_mask = model(noisy_frames)

        loss_noise = criterion(pred_noise_mask, target_noise_mask)
        loss_reverb = criterion(pred_reverb_mask, target_reverb_mask)
        loss = loss_noise + loss_reverb

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def export_onnx(model, fft_size, output_path, device):
    """Export trained model to ONNX format matching plugin expectations."""
    model.eval()
    num_bins = fft_size // 2 + 1

    dummy_input = torch.rand(1, NUM_FRAMES, num_bins, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["noise_mask", "reverb_mask"],
        dynamic_axes=None,  # Fixed batch size of 1
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"  Exported ONNX: {output_path}")


def train_for_fft_size(fft_size, args, device):
    print(f"\n{'='*60}")
    print(f"Training model for FFT size {fft_size}")
    print(f"{'='*60}")

    config = get_model_config(fft_size)
    num_bins = config['num_bins']
    print(f"  num_bins={num_bins}, hidden_size={config['hidden_size']}")

    model = create_model(fft_size).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Datasets
    train_dataset = SpectralDataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        rir_dir=args.rir_dir,
        fft_size=fft_size,
        num_frames=NUM_FRAMES,
        sample_rate=48000,
        segment_length=2.0,
        snr_range=(-5, 20),
        num_samples=args.train_samples,
    )

    val_dataset = SpectralDataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        rir_dir=args.rir_dir,
        fft_size=fft_size,
        num_frames=NUM_FRAMES,
        sample_rate=48000,
        segment_length=2.0,
        snr_range=(-5, 20),
        num_samples=args.val_samples,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, f'denoiser_{fft_size}_best.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'fft_size': fft_size,
                'epoch': epoch,
                'val_loss': val_loss,
            }, ckpt_path)

    # Load best and export
    ckpt_path = os.path.join(checkpoint_dir, f'denoiser_{fft_size}_best.pt')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded best checkpoint (epoch {checkpoint['epoch']}, "
              f"val_loss={checkpoint['val_loss']:.6f})")

    # Export ONNX
    onnx_path = os.path.join(args.output_dir, f'denoiser_{fft_size}.onnx')
    export_onnx(model, fft_size, onnx_path, device)

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Train DeFeedback Pro denoiser models')
    parser.add_argument('--clean-dir', required=True, help='Directory with clean speech files')
    parser.add_argument('--noise-dir', required=True, help='Directory with noise files')
    parser.add_argument('--rir-dir', default=None, help='Directory with RIR files (optional)')
    parser.add_argument('--output-dir', default='models', help='Output directory for ONNX models')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per FFT size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--train-samples', type=int, default=10000,
                        help='Training samples per epoch')
    parser.add_argument('--val-samples', type=int, default=2000,
                        help='Validation samples per epoch')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--fft-sizes', nargs='+', type=int, default=FFT_SIZES,
                        help='FFT sizes to train')
    parser.add_argument('--device', default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    for fft_size in args.fft_sizes:
        best_loss = train_for_fft_size(fft_size, args, device)
        results[fft_size] = best_loss

    print(f"\n{'='*60}")
    print("Training complete! Results:")
    print(f"{'='*60}")
    for fft_size, loss in results.items():
        onnx_path = os.path.join(args.output_dir, f'denoiser_{fft_size}.onnx')
        size_mb = os.path.getsize(onnx_path) / 1024 / 1024 if os.path.exists(onnx_path) else 0
        print(f"  FFT {fft_size:5d}: val_loss={loss:.6f}, model={size_mb:.2f} MB")


if __name__ == '__main__':
    main()
