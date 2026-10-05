

"""
model.py

CF-ResNet-1D baseline and ResNet9 baselines for the MOSBNET / MIRU pipeline.

Original CF-ResNet-1D source reference:
https://github.com/szbela87/insect_wingbeat_classification_fno/blob/main/model.py

The CF-ResNet-1D implementation below is adapted from the original paper code used
for insect wingbeat sound classification. The original code processes one waveform
at a time with input shape [N, T] and returns [N, num_classes].

MOSBNET pipeline adaptation:
Your dataloader returns one 10-second recording as a sequence of short segments:
    [B, S, 1, T]
where:
    B = batch size
    S = number of segments per recording, usually 20 for 10 s / 0.5 s
    T = samples per segment, usually 4000 for 0.5 s at 8 kHz

Your train/eval loop flattens predictions with:
    logits = outputs.view(-1, outputs.shape[-1])
    y = targets.view(-1)

Therefore the model must return:
    [B, S, num_classes]

To make CF-ResNet-1D work correctly in this pipeline, this file implements:
    1. CFResNet1DSegment: original-style segment classifier, input [N, 1, T]
    2. CFResNet1DSequence: wrapper that flattens [B, S, 1, T] to [B*S, 1, T],
       runs the segment classifier, and reshapes output back to [B, S, C]

Small safety adaptation:
The original SpectralConv1d writes the requested number of Fourier modes directly.
In this pipeline, repeated pool_size=5 can shrink 4000 samples to 32 samples before
later residual blocks. That gives only 17 rFFT bins. To avoid shape errors when
modes is larger than available bins, SpectralConv1d clamps used_modes to the
available FFT bins at runtime. This does not change the layer idea. It only makes
it robust to shorter segment lengths after pooling.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio




# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _check_odd_kernel(kernel_size: int) -> None:
    if int(kernel_size) % 2 == 0:
        raise ValueError(
            f"kernel_size must be odd to preserve temporal length, got {kernel_size}."
        )


def _as_segment_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Convert segment input to [N, 1, T].

    Accepted:
        [N, T]
        [N, 1, T]
    """
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        if x.shape[1] != 1:
            raise ValueError(f"Expected channel dimension 1, got shape {tuple(x.shape)}")
        return x
    raise ValueError(f"Expected [N, T] or [N, 1, T], got shape {tuple(x.shape)}")


# -----------------------------------------------------------------------------
# Plain ResNet9 baselines from the original folder code
# -----------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Plain Conv1D residual block from the original ResNet9 baseline."""

    def __init__(self, channels: int, kernel_size: int, padding: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=True,
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=True,
        )
        self.bn1 = nn.BatchNorm1d(num_features=channels)
        self.bn2 = nn.BatchNorm1d(num_features=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.conv1(x))
        out = self.bn1(out)
        out = F.gelu(self.conv2(out))
        out = self.bn2(out)
        return out + residual


class ResNet9Segment(nn.Module):
    """
    Plain ResNet9 segment classifier.

    This is useful as a non-Fourier baseline alongside CF-ResNet-1D.

    Input:
        [N, T] or [N, 1, T]

    Output:
        [N, num_classes]
    """

    CHANNELS = {
        "small": (32, 64, 96, 128),
        "medium": (32, 64, 128, 256),
        "large": (64, 128, 256, 512),
    }

    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        pool_size: int = 5,
        kernel_size: int = 11,
    ):
        super().__init__()
        _check_odd_kernel(kernel_size)
        if variant not in self.CHANNELS:
            raise ValueError(f"variant must be one of {sorted(self.CHANNELS)}, got {variant}")

        c1, c2, c3, c4 = self.CHANNELS[variant]
        padding = kernel_size // 2

        self.variant = variant
        self.pool_size = int(pool_size)
        self.kernel_size = int(kernel_size)

        self.conv1 = nn.Conv1d(1, c1, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn1 = nn.BatchNorm1d(c1)

        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(c2)

        self.rb1 = ResidualBlock(c2, kernel_size=kernel_size, padding=padding, stride=1)

        self.conv3 = nn.Conv1d(c2, c3, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn3 = nn.BatchNorm1d(c3)

        self.conv4 = nn.Conv1d(c3, c4, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn4 = nn.BatchNorm1d(c4)

        self.rb2 = ResidualBlock(c4, kernel_size=kernel_size, padding=padding, stride=1)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c4, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_segment_batch(x)
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.gelu(x)
        x = self.bn2(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = self.rb1(x)

        x = self.conv3(x)
        x = F.gelu(x)
        x = self.bn3(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)

        x = self.conv4(x)
        x = F.gelu(x)
        x = self.bn4(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = self.rb2(x)

        x = self.gap(x)
        x = x.reshape(batch_size, -1)
        return self.fc(x)


class ResNet9Sequence(nn.Module):
    """
    Sequence wrapper for plain ResNet9.

    Input:
        [B, S, 1, T]

    Output:
        [B, S, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        pool_size: int = 5,
        kernel_size: int = 11,
    ):
        super().__init__()
        self.segment_model = ResNet9Segment(
            num_classes=num_classes,
            variant=variant,
            pool_size=pool_size,
            kernel_size=kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            b, s, c, t = x.shape
            if c != 1:
                raise ValueError(f"Expected channel dimension 1, got shape {tuple(x.shape)}")
            x_flat = x.reshape(b * s, c, t)
            logits = self.segment_model(x_flat)
            return logits.reshape(b, s, -1)

        # Also allow direct segment batches for debugging or standalone use.
        if x.ndim in (2, 3):
            return self.segment_model(x)

        raise ValueError(f"Expected [B, S, 1, T], [N, 1, T], or [N, T], got {tuple(x.shape)}")


# -----------------------------------------------------------------------------
# CF-ResNet-1D baseline adapted from the original paper code
# -----------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """
    1D Fourier spectral convolution.

    Original source reference:
    https://github.com/szbela87/insect_wingbeat_classification_fno/blob/main/model.py

    Original comment in the source says this layer follows the Fourier Neural
    Operator idea: FFT, learnable linear transform in the Fourier domain, then
    inverse FFT.

    MOSBNET adaptation:
    used_modes = min(requested modes, available rFFT bins)
    so the layer remains safe after repeated temporal pooling.
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        if int(modes1) <= 0:
            raise ValueError(f"modes1 must be positive, got {modes1}")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)

        self.scale = 1.0 / (self.in_channels * self.out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                self.in_channels,
                self.out_channels,
                self.modes1,
                2,
                dtype=torch.float,
            )
        )

    @staticmethod
    def compl_mul1d(input_ft: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication.

        input_ft: [batch, in_channel, modes]
        weights:  [in_channel, out_channel, modes]
        output:   [batch, out_channel, modes]
        """
        return torch.einsum("bix,iox->box", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        signal_length = x.size(-1)

        x_ft = torch.fft.rfft(x)
        n_freq = signal_length // 2 + 1
        used_modes = min(self.modes1, n_freq)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            n_freq,
            device=x.device,
            dtype=torch.cfloat,
        )

        weights = torch.view_as_complex(self.weights1)
        out_ft[:, :, :used_modes] = self.compl_mul1d(
            x_ft[:, :, :used_modes],
            weights[:, :, :used_modes],
        )

        return torch.fft.irfft(out_ft, n=signal_length)


class FourierLayer(nn.Module):
    """
    Convolutional Fourier Layer from CF-ResNet-1D.

    output = Conv1D(x) + SpectralConv1D(x)

    This mirrors the original code structure:
        x1 = self.conv1(x)
        x2 = self.conv_fno1(x)
        out = x1 + x2
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        modes: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=True,
        )
        self.conv_fno1 = SpectralConv1d(in_channels, out_channels, modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x) + self.conv_fno1(x)


class ResidualBlockFNO(nn.Module):
    """Residual block using two Convolutional Fourier Layers."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        padding: int,
        stride: int,
        modes: int,
    ):
        super().__init__()
        self.fn1 = FourierLayer(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            modes=modes,
        )
        self.fn2 = FourierLayer(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            modes=modes,
        )
        self.bn1 = nn.BatchNorm1d(num_features=channels)
        self.bn2 = nn.BatchNorm1d(num_features=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.fn1(x))
        out = self.bn1(out)
        out = F.gelu(self.fn2(out))
        out = self.bn2(out)
        return out + residual


class CFResNet1DSegment(nn.Module):
    """
    CF-ResNet-1D segment classifier.

    This is the original-style CF-ResNet classifier adapted for robust segment
    input shapes. It predicts one label per segment.

    Variants:
        small:  1 -> 32 -> 64 -> 96  -> 128
        medium: 1 -> 32 -> 64 -> 128 -> 256
        large:  1 -> 64 -> 128 -> 256 -> 512

    Input:
        [N, T] or [N, 1, T]

    Output:
        [N, num_classes]

    Notes for faithful reporting:
        The original uploaded/source code uses pool_size=5 and kernel_size=11.
        The original small FNO class default uses modes=64, while the medium
        class default uses modes=16. For the MOSBNET segment pipeline, modes=16
        is a safer default because the segment length is 4000 and repeated
        pool_size=5 reduces the final temporal dimension to about 32 samples.
        You may still pass modes=64 for a closer small-model reproduction because
        SpectralConv1d safely clamps used modes at runtime.
    """

    CHANNELS = {
        "small": (32, 64, 96, 128),
        "medium": (32, 64, 128, 256),
        "large": (64, 128, 256, 512),
    }

    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        pool_size: int = 5,
        kernel_size: int = 11,
        modes: int = 16,
    ):
        super().__init__()
        _check_odd_kernel(kernel_size)
        if variant not in self.CHANNELS:
            raise ValueError(f"variant must be one of {sorted(self.CHANNELS)}, got {variant}")

        c1, c2, c3, c4 = self.CHANNELS[variant]
        padding = kernel_size // 2

        self.variant = variant
        self.pool_size = int(pool_size)
        self.kernel_size = int(kernel_size)
        self.modes = int(modes)

        self.conv1 = FourierLayer(1, c1, kernel_size, padding, stride=1, modes=modes)
        self.bn1 = nn.BatchNorm1d(num_features=c1)

        self.conv2 = FourierLayer(c1, c2, kernel_size, padding, stride=1, modes=modes)
        self.bn2 = nn.BatchNorm1d(num_features=c2)

        self.rb1 = ResidualBlockFNO(c2, kernel_size, padding, stride=1, modes=modes)

        self.conv3 = FourierLayer(c2, c3, kernel_size, padding, stride=1, modes=modes)
        self.bn3 = nn.BatchNorm1d(num_features=c3)

        self.conv4 = FourierLayer(c3, c4, kernel_size, padding, stride=1, modes=modes)
        self.bn4 = nn.BatchNorm1d(num_features=c4)

        self.rb2 = ResidualBlockFNO(c4, kernel_size, padding, stride=1, modes=modes)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=c4, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_segment_batch(x)
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = F.gelu(x)
        x = self.bn2(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = self.rb1(x)

        x = self.conv3(x)
        x = F.gelu(x)
        x = self.bn3(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)

        x = self.conv4(x)
        x = F.gelu(x)
        x = self.bn4(x)

        x = F.avg_pool1d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = self.rb2(x)

        x = self.gap(x)
        x = x.reshape(batch_size, -1)
        return self.fc(x)


class CFResNet1DSequence(nn.Module):
    """
    MOSBNET-compatible CF-ResNet-1D wrapper.

    This is the class you should instantiate in your current training pipeline.

    Input from AudioSequenceDataset:
        [B, S, 1, T]

    Output expected by train_model/evaluate_model:
        [B, S, num_classes]

    Implementation statement for paper/code comments:
        We adapted the original CF-ResNet-1D segment classifier to our sequence
        pipeline by applying the same CF-ResNet-1D network independently to each
        0.5-second segment. The wrapper reshapes the batch from [B, S, 1, T] to
        [B*S, 1, T], performs segment-level classification, and restores the
        output to [B, S, C] so that the existing segment-level loss and metrics
        operate without changing the training loop.
    """

    def __init__(
        self,
        num_classes: int,
        variant: str = "small",
        pool_size: int = 5,
        kernel_size: int = 11,
        modes: int = 16,
    ):
        super().__init__()
        self.segment_model = CFResNet1DSegment(
            num_classes=num_classes,
            variant=variant,
            pool_size=pool_size,
            kernel_size=kernel_size,
            modes=modes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            b, s, c, t = x.shape
            if c != 1:
                raise ValueError(f"Expected channel dimension 1, got shape {tuple(x.shape)}")
            x_flat = x.reshape(b * s, c, t)
            logits = self.segment_model(x_flat)
            return logits.reshape(b, s, -1)

        # Also allow direct segment batches for debugging or standalone use.
        if x.ndim in (2, 3):
            return self.segment_model(x)

        raise ValueError(f"Expected [B, S, 1, T], [N, 1, T], or [N, T], got {tuple(x.shape)}")


# -----------------------------------------------------------------------------
# Convenience aliases for clearer experiment names
# -----------------------------------------------------------------------------

class CFResNet1DSmall(CFResNet1DSequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11, modes: int = 16):
        super().__init__(num_classes, variant="small", pool_size=pool_size, kernel_size=kernel_size, modes=modes)


class CFResNet1DMedium(CFResNet1DSequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11, modes: int = 16):
        super().__init__(num_classes, variant="medium", pool_size=pool_size, kernel_size=kernel_size, modes=modes)


class CFResNet1DLarge(CFResNet1DSequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11, modes: int = 16):
        super().__init__(num_classes, variant="large", pool_size=pool_size, kernel_size=kernel_size, modes=modes)


class ResNet9Small(ResNet9Sequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11):
        super().__init__(num_classes, variant="small", pool_size=pool_size, kernel_size=kernel_size)


class ResNet9Medium(ResNet9Sequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11):
        super().__init__(num_classes, variant="medium", pool_size=pool_size, kernel_size=kernel_size)


class ResNet9Large(ResNet9Sequence):
    def __init__(self, num_classes: int, pool_size: int = 5, kernel_size: int = 11):
        super().__init__(num_classes, variant="large", pool_size=pool_size, kernel_size=kernel_size)


# -----------------------------------------------------------------------------
# Factory helper
# -----------------------------------------------------------------------------

def build_model(
    model_name: str,
    num_classes: int,
    pool_size: int = 5,
    kernel_size: int = 11,
    modes: int = 16,
) -> nn.Module:
    """
    Build a sequence-compatible model for your current training loop.

    Examples:
        model = build_model("cfresnet1d_small", num_classes=len(label_map))
        model = build_model("cfresnet1d_medium", num_classes=len(label_map))
        model = build_model("resnet9_small", num_classes=len(label_map))

    Returns models that accept [B, S, 1, T] and return [B, S, C].
    """
    name = model_name.strip().lower()

    if name in {"cfresnet1d_small", "cf_resnet1d_small", "cfresnet_small"}:
        return CFResNet1DSequence(num_classes, "small", pool_size, kernel_size, modes)

    if name in {"cfresnet1d_medium", "cf_resnet1d_medium", "cfresnet_medium"}:
        return CFResNet1DSequence(num_classes, "medium", pool_size, kernel_size, modes)

    if name in {"cfresnet1d_large", "cf_resnet1d_large", "cfresnet_large"}:
        return CFResNet1DSequence(num_classes, "large", pool_size, kernel_size, modes)

    if name in {"resnet9_small", "resnet1d_small"}:
        return ResNet9Sequence(num_classes, "small", pool_size, kernel_size)

    if name in {"resnet9_medium", "resnet1d_medium"}:
        return ResNet9Sequence(num_classes, "medium", pool_size, kernel_size)

    if name in {"resnet9_large", "resnet1d_large"}:
        return ResNet9Sequence(num_classes, "large", pool_size, kernel_size)

    raise ValueError(
        f"Unknown model_name={model_name!r}. Supported: "
        "cfresnet1d_small, cfresnet1d_medium, cfresnet1d_large, "
        "resnet9_small, resnet9_medium, resnet9_large"
    )


__all__ = [
    "count_parameters",
    "ResidualBlock",
    "ResNet9Segment",
    "ResNet9Sequence",
    "ResNet9Small",
    "ResNet9Medium",
    "ResNet9Large",
    "SpectralConv1d",
    "FourierLayer",
    "ResidualBlockFNO",
    "CFResNet1DSegment",
    "CFResNet1DSequence",
    "CFResNet1DSmall",
    "CFResNet1DMedium",
    "CFResNet1DLarge",
    "build_model",
]


# if __name__ == "__main__":
#     # Tiny smoke test for the MOSBNET shape.
#     # B=2 recordings, S=20 segments, C=1 channel, T=4000 samples.
#     model = CFResNet1DSequence(num_classes=9, variant="small", pool_size=5, kernel_size=11, modes=16)
#     x = torch.randn(2, 20, 1, 4000)
#     y = model(x)
#     print("output shape:", tuple(y.shape))
#     print("trainable params:", count_parameters(model))
