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


def _unwrap_transfusion_model(model: Transfusion) -> Transfusion:
    if hasattr(model, "ema_model") and model.ema_model is not None:
        return model.ema_model
    if hasattr(model, "online_model"):
        online = model.online_model
        if isinstance(online, list) and online:
            return online[0]
        return online
    return model


def decode_text_with_special(tokens: Tensor, model: Transfusion) -> str:
    model = _unwrap_transfusion_model(model)
    if tokens.ndim > 1:
        tokens = tokens.flatten()
    parts: list[str] = []
    for tok in tokens.tolist():
        if 0 <= tok < NUM_TEXT_TOKENS:
            parts.append(chr(tok))
        elif tok == model.sos_id:
            parts.append("<SOS>")
        elif tok == model.eos_id:
            parts.append("<EOS>")
        elif tok == model.meta_id:
            parts.append("<META>")
        elif tok in model.som_ids:
            parts.append(f"<SOM:{model.som_ids.index(tok)}>")
        elif tok in model.eom_ids:
            parts.append(f"<EOM:{model.eom_ids.index(tok)}>")
        elif tok >= model.meta_id + 1:
            parts.append(model.decode_chars(torch.tensor([tok], device=tokens.device)))
        else:
            parts.append(f"<UNK:{tok}>")
    return "".join(parts)


def parse_sample_parts(
    modality_sample: Iterable,
    model: Transfusion | None = None,
) -> list[dict]:
    parts: list[dict] = []
    for item in modality_sample:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Tensor):
            parts.append(
                {
                    "kind": "modality",
                    "modality_type": item[0],
                    "tensor": item[1],
                }
            )
            print(f"modality: {item[1].shape}]")
        elif isinstance(item, Tensor) and item.dtype.is_floating_point:
            parts.append(
                {
                    "kind": "modality",
                    "modality_type": None,
                    "tensor": item,
                }
            )
            print(f"modality: {item.shape}]")
        elif isinstance(item, Tensor) and item.dtype in (torch.int, torch.long):
            parts.append(
                {
                    "kind": "text",
                    "tensor": item,
                }
            )
            decoded = decode_text_with_special(item, model)
            print(f"text: {decoded!r}")
    return parts


def decode_first_image_part(parts: list[dict]) -> Tuple[int | None, Tensor | None]:
    for part in parts:
        if part["kind"] != "modality":
            continue
        image = part["tensor"]
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        return part["modality_type"], image
    return None, None


def slugify_prompt(prompt: str, max_len: int = 60) -> str:
    prompt = prompt.encode("ascii", errors="ignore").decode("ascii").lower()
    prompt = re.sub(r"[^a-z0-9]+", "-", prompt).strip("-")
    if not prompt:
        return "prompt"
    return prompt[:max_len].rstrip("-")


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
        
        print(f"Prompt #{idx}: {prompt}")
        
        print_modality_sample(sample)
        parts = parse_sample_parts(sample, model=model)
        modality_type, image = decode_first_image_part(parts)

        if image is None:
            print(f'[warn] no modality found for prompt: {prompt}')
            continue
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        image_tensor = image.detach().cpu()

        slug = slugify_prompt(prompt)
        filename = output_dir / f"{slug}_{time()}.png"
        save_image(
            image_tensor,
            filename
        )


def parse_args():
    parser = ArgumentParser(description="MNIST augmentation Transfusion inference")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--prompt", action="append", default=None, help="Prompt text (repeatable)")
    parser.add_argument("--max-length", type=int, default=384, help="Max autoregressive length for sampling")
    parser.add_argument("--prompt-num", type=int, default=10, help="Number of prompts to generate if none provided")
    parser.add_argument("--text-temperature", type=float, default=1.5, help="Text sampling temperature")
    parser.add_argument("--output-dir", type=Path, default=Path("inference_outputs/mnist_augment"))
    parser.add_argument("--no-ema", action="store_true", help="Use raw model weights instead of EMA if available")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu, cuda, mps")
    return parser.parse_args()


def main():
    args = parse_args()

    sheer_min_deg = 4.0
    sheer_max_deg = 20.0

    prompts = []
    if args.prompt:
        prompts.extend(args.prompt)
    else:
        prompts, _ = gen_random_prompts(args.prompt_num, sheer_min_deg, sheer_max_deg)
        
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
