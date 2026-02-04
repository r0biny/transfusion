from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Tuple
from time import time
import re
import sys
import random

import torch
from torch import Tensor, nn
from einops import rearrange
from torchvision.utils import save_image

# make sure project root is importable when running from repository root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transfusion_pytorch import Transfusion, print_modality_sample


NUM_TEXT_TOKENS = 128


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
        num_text_tokens=NUM_TEXT_TOKENS,
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


def encode_text(text: str) -> Tensor:
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return torch.tensor([*map(ord, text)], dtype=torch.long)


def decode_text_tokens(tokens: Tensor) -> str:
    if tokens.ndim > 1:
        tokens = tokens.flatten()
    tokens = tokens[(tokens >= 0) & (tokens < NUM_TEXT_TOKENS)]
    return "".join([chr(int(t)) for t in tokens])


def extract_first_modality(modality_sample: Iterable) -> Tuple[int | None, Tensor | None]:
    for item in modality_sample:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Tensor):
            return item
        if isinstance(item, Tensor) and item.dtype.is_floating_point:
            return None, item
    return None, None


def extract_text_after_first_modality(modality_sample: Iterable) -> Tensor | None:
    seen_modality = False
    text_chunks: list[Tensor] = []
    for item in modality_sample:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Tensor):
            if not seen_modality:
                seen_modality = True
            continue
        if isinstance(item, Tensor) and item.dtype in (torch.int, torch.long):
            if seen_modality:
                text_chunks.append(item)
    if not text_chunks:
        return None
    return torch.cat([t.flatten() for t in text_chunks], dim=0)


def save_first_image(modality_sample: Iterable, output_path: Path) -> bool:
    modality_type, image = extract_first_modality(modality_sample)
    if image is None:
        return False

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(image.detach().cpu(), output_path)
    tag = f"modality_{modality_type}" if modality_type is not None else "modality"
    print(f"saved {tag} to {output_path}")
    return True


def slugify_prompt(prompt: str, max_len: int = 60) -> str:
    prompt = prompt.encode("ascii", errors="ignore").decode("ascii").lower()
    prompt = re.sub(r"[^a-z0-9]+", "-", prompt).strip("-")
    if not prompt:
        return "prompt"
    return prompt[:max_len].rstrip("-")


def load_prompts(prompt_args: list[str] | None, prompt_file: Path | None) -> list[str]:
    prompts: list[str] = []
    if prompt_args:
        prompts.extend(prompt_args)
    if prompt_file:
        for line in prompt_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts

def gen_random_prompts(num_samples: int, sheer_min_deg: float, sheer_max_deg: float) -> list[str]:
    prompts = []
    addition_infos = []
    for _ in range(num_samples):
        digit = random.randint(0, 9)
        shear_deg = random.uniform(sheer_min_deg, sheer_max_deg)
        direction = 'right' if random.random() > 0.5 else 'left'
        magnitude = abs(shear_deg)
        if magnitude < 8:
            intensity = 'slightly'
        elif magnitude < 14:
            intensity = 'moderately'
        else:
            intensity = 'strongly'
        prompt = f'A handwritten digit {digit} slants {intensity} to the {direction}.'
        addition_infos.append((digit, shear_deg, direction))
        prompts.append(prompt)
    return prompts, addition_infos

def run_prompted_sample(
    model: Transfusion,
    prompts: Iterable[str],
    max_length: int,
    text_temperature: float,
    output_dir: Path,
    device: torch.device | None = None,
) -> None:
    for idx, prompt in enumerate(prompts):
        tokens = encode_text(prompt).unsqueeze(0).to(device)
        sample = model.sample(
            prompt=tokens, 
            max_length=max_length,
            text_temperature=text_temperature,
        )
        print(f"prompt #{idx}: {prompt}")
        print_modality_sample(sample)

        slug = slugify_prompt(prompt)
        filename = output_dir / f"{slug}_{time()}.png"

        trailing_tokens = extract_text_after_first_modality(sample)
        if trailing_tokens is not None:
            decoded = decode_text_tokens(trailing_tokens)
            print(f"text after modality: {decoded!r}")

        modality_type, image = extract_first_modality(sample)
        if image is None:
            print(f'[warn] no modality found for prompt: {prompt}')
            continue
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        image_tensor = image.detach().cpu()

        save_image(
            image_tensor,
            filename
        )

        # saved = save_first_image(sample, filename)
        # if not saved:
            # print(f"[warn] no modality found for prompt #{idx}")


def parse_args():
    parser = ArgumentParser(description="MNIST augmentation Transfusion inference")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt text (repeatable)")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Text file with one prompt per line")
    parser.add_argument("--max-length", type=int, default=384, help="Max autoregressive length for sampling")
    parser.add_argument("--text-temperature", type=float, default=1.5, help="Text sampling temperature")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_outputs/mnist_augment"))
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA if available")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu, cuda, mps")
    return parser.parse_args()


def main():
    args = parse_args()

    sheer_min_deg = 4.0
    sheer_max_deg = 20.0

    prompts = load_prompts(args.prompt, args.prompt_file)
    if not prompts:
        prompts, _ = gen_random_prompts(10, sheer_min_deg, sheer_max_deg)
        
    device = select_device(args.device)
    print(f"using device: {device}")

    model = restore_model(args.checkpoint, device=device, use_ema=not args.no_ema)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_prompted_sample(
        model,
        prompts=prompts,
        max_length=args.max_length,
        text_temperature=args.text_temperature,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
