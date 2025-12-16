
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import sys
import torch
from time import time
from torch import Tensor, nn
from einops import rearrange
from torchvision.utils import save_image

# make sure project root is importable when running from repository root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transfusion_pytorch import Transfusion, print_modality_sample
# to fix: how to relative import local transfusion_pytorch

# model definition mirrors train_mnist_with_unet_new.py


class Encoder(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "... 1 (h p1) (w p2) -> ... (p1 p2) h w", p1=2, p2=2)
        return x * 2 - 1


class Decoder(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "... (p1 p2) h w -> ... 1 (h p1) (w p2)", p1=2, p2=2, h=14)
        return ((x + 1) * 0.5).clamp(min=0.0, max=1.0)


def select_device(force: str | None = None) -> torch.device:
    if force:
        return torch.device(force)
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if mps_available:
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def build_model(device: torch.device) -> Transfusion:
    model = Transfusion(
        num_text_tokens=10,
        dim_latent=4,
        modality_default_shape=(14, 14),
        modality_encoder=Encoder(),
        modality_decoder=Decoder(),
        pre_post_transformer_enc_dec=(
            nn.Conv2d(4, 64, 3, 2, 1),
            nn.ConvTranspose2d(64, 4, 3, 2, 1, output_padding=1),
        ),
        add_pos_emb=True,
        modality_num_dim=2,
        channel_first_latent=True,
        transformer=dict(
            dim=64,
            depth=4,
            dim_head=32,
            heads=8,
        ),
    ).to(device)
    return model


def restore_model(checkpoint_path: Path, device: torch.device, use_ema: bool) -> Transfusion:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(device)
    model.load_state_dict(checkpoint["model"])

    if use_ema and "ema_model" in checkpoint:
        ema_model = model.create_ema()
        ema_model.load_state_dict(checkpoint["ema_model"])
        model = ema_model

    model.eval()
    return model


def extract_first_modality(modality_sample: Iterable) -> Tuple[int | None, Tensor | None]:
    for item in modality_sample:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Tensor):
            return item
        if isinstance(item, Tensor) and item.dtype.is_floating_point:
            return None, item
    return None, None


def save_first_image(modality_sample: Iterable, output_path: Path) -> bool:
    modality_type, image = extract_first_modality(modality_sample)
    if image is None:
        return False

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]


    # make sure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(image.detach().cpu(), output_path)
    tag = f"modality_{modality_type}" if modality_type is not None else "modality"
    print(f"saved {tag} to {output_path}")
    return True


def run_unconditional_sample(model: Transfusion, max_length: int, output_dir: Path) -> None:
    sample = model.sample(max_length=max_length)
    print_modality_sample(sample)
    maybe_label, maybe_image, *_ = sample

    digit = maybe_label[1].item()

    filename = output_dir / str(digit) / f"random_{time()}.png"
    saved = save_first_image(sample, filename)
    if not saved:
        print(f"[warn] no modality found in sample #{digit}")


def run_digit_conditioned_sample(
    model: Transfusion,
    digits: Iterable[int],
    max_length: int,
    output_dir: Path,
    device: torch.device | None = None,
) -> None:
    for i, digit in enumerate(digits):
        if digit < 0 or digit > 9:
            raise ValueError(f"digit {digit} out of range [0, 9]")

        prompt = torch.tensor([digit], device=device)
        sample = model.sample(prompt=prompt, max_length=max_length)
        print(f"digit {digit} -> sample:")
        print_modality_sample(sample)

        filename = output_dir / str(digit) / f"digit_{time()}.png"
        saved = save_first_image(sample, filename)
        if not saved:
            print(f"[warn] no modality found for digit {digit}")


def parse_args():
    parser = ArgumentParser(description="MNIST+UNet Transfusion inference")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--digits", type=int, nargs="*", default=None, help="Digits to condition on (0-9)")
    parser.add_argument("--num-random", type=int, default=2, help="Number of unconditional samples to generate")
    parser.add_argument("--max-length", type=int, default=384, help="Max autoregressive length for sampling")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_outputs/mnist_unet"))
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA if available")
    return parser.parse_args()


def main():
    args = parse_args()

    device = select_device()
    print(f"using device: {device}")

    model = restore_model(args.checkpoint, device=device, use_ema=not args.no_ema)

    timestamp_dir = args.output_dir
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.num_random):
        run_unconditional_sample(
            model, 
            max_length=args.max_length, 
            output_dir=timestamp_dir
        )

    if args.digits:
        run_digit_conditioned_sample(
            model,
            digits=args.digits,
            max_length=args.max_length,
            output_dir=timestamp_dir,
            device=device,
        )


if __name__ == "__main__":
    main()
