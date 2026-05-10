# -*- coding: utf-8 -*-
"""
Default launch example:

python train_calliffusion.py \
  --dataset_root ~/Code/calligraphy_project \
  --train_csv ~/Code/calligraphy_project/dataset/train_260402.csv \
  --val_csv ~/Code/calligraphy_project/dataset/validation.csv \
  --image_column path \
  --caption_template "一幅取法{source}的{style}单字书法作品，书写内容为{character}。" \
  --image_backend vae \
  --vae_model_name_or_path stabilityai/sd-vae-ft-mse \
  --vae_downsample_factor 8 \
  --image_size 64 \
  --dim 512 \
  --depth 8 \
  --heads 8 \
  --dim_head 64 \
  --dropout 0.0 \
  --train_batch_size 256 \
  --learning_rate 3e-4 \
  --weight_decay 1e-2 \
  --gradient_accumulation_steps 1 \
  --max_grad_norm 0.5 \
  --mixed_precision bf16 \
  --num_workers 8 \
  --epochs 3000 \
  --ema_beta 0.99 \
  --sample_every 1000 \
  --sample_count 8 \
  --validation_loss_every 1000 \
  --val_batch_size 8 \
  --modality_steps 16 \
  --checkpoint_every 5000 \
  --output_root ~/Code/transfusion/output
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from einops import rearrange
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize, ToTensor
from torchvision.utils import save_image
from tqdm.auto import tqdm

from transfusion_pytorch import Transfusion

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_rel_path(rel_path: str) -> str:
    return rel_path.replace("\\", "/").strip()


def resolve_target_path(rel_path: str) -> str:
    rel_path = normalize_rel_path(rel_path)
    if rel_path.startswith("char_process/"):
        return rel_path
    if rel_path.startswith("char/"):
        return rel_path.replace("char/", "char_process/", 1)
    return rel_path


def path_tags(row: Dict[str, str], image_column: str) -> List[str]:
    rel_path = row.get(image_column, "") or row.get("path", "")
    if not rel_path:
        return []

    basename = os.path.splitext(os.path.basename(normalize_rel_path(rel_path)))[0]
    return basename.split()


def derive_raw_description(row: Dict[str, str], image_column: str = "path") -> str:
    return " ".join(path_tags(row, image_column))


def char_from_tag(tag: str) -> str:
    tag = tag.strip()
    if len(tag) > 1 and tag.endswith("字"):
        return tag[:-1]
    return ""


UNKNOWN_CHARACTER = "未知字"
UNKNOWN_STYLE = "未知书体"
UNKNOWN_SOURCE = "未知作者"


def encode_caption(caption: str) -> torch.Tensor:
    return torch.tensor(list(caption.encode("utf-8")), dtype=torch.long)


@dataclass(frozen=True)
class CaptionParts:
    raw: str
    character_text: str
    character: str
    style: str
    source: str
    source_style_descriptor: str


class PatchEncoder(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        patches = rearrange(image, "... c (h p1) (w p2) -> ... (c p1 p2) h w", p1=p, p2=p)
        return patches * 2.0 - 1.0


class PatchDecoder(nn.Module):
    def __init__(self, patch_size: int, channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        image = rearrange(
            patches,
            "... (c p1 p2) h w -> ... c (h p1) (w p2)",
            c=self.channels,
            p1=p,
            p2=p,
        )
        return ((image + 1.0) * 0.5).clamp(min=0.0, max=1.0)


class VAEEncoder(nn.Module):
    def __init__(self, vae: nn.Module, scaling_factor: float, sample_latents: bool = False):
        super().__init__()
        self.vae = vae
        self.scaling_factor = scaling_factor
        self.sample_latents = sample_latents

        self.vae.requires_grad_(False)
        self.vae.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.vae.eval()
        with torch.no_grad():
            posterior = self.vae.encode(image * 2.0 - 1.0).latent_dist
            latents = posterior.sample() if self.sample_latents else posterior.mode()
        return latents * self.scaling_factor


class VAEDecoder(nn.Module):
    def __init__(self, vae: nn.Module, scaling_factor: float):
        super().__init__()
        self.vae = vae
        self.scaling_factor = scaling_factor

        self.vae.requires_grad_(False)
        self.vae.eval()

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        self.vae.eval()
        with torch.no_grad():
            image = self.vae.decode(latents / self.scaling_factor).sample
        return (image / 2.0 + 0.5).clamp(min=0.0, max=1.0)


class CalliffusionCsvDataset(Dataset):
    DEFAULT_CAPTION_TEMPLATE = "一幅取法{source}的{style}单字书法作品，书写内容为{character}。"

    def __init__(
        self,
        rows: List[Dict[str, str]],
        dataset_root: Path,
        transform,
        *,
        image_column: str = "path",
        caption_template: str = DEFAULT_CAPTION_TEMPLATE,
    ):
        self.rows = rows
        self.dataset_root = dataset_root
        self.transform = transform
        self.image_column = image_column
        self.caption_template = caption_template
        self.captions = [self.format_caption(row) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def resolve_image_path(self, row: Dict[str, str]) -> Path:
        rel_path = row.get(self.image_column, "") or row.get("path", "")
        rel_path = resolve_target_path(rel_path)
        path = Path(rel_path)

        if path.is_absolute():
            return path

        candidates = [
            self.dataset_root / "dataset" / rel_path,
            self.dataset_root / rel_path,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def extract_caption_parts(self, row: Dict[str, str]) -> CaptionParts:
        filename_tags = path_tags(row, self.image_column)
        raw = " ".join(filename_tags)

        character_tag = filename_tags[0] if filename_tags else ""
        character_text = char_from_tag(character_tag)

        if character_text:
            style = filename_tags[1] if len(filename_tags) >= 2 else UNKNOWN_STYLE
            source = " ".join(filename_tags[2:]) if len(filename_tags) >= 3 else UNKNOWN_SOURCE
        elif len(filename_tags) == 3:
            style = filename_tags[1]
            source = filename_tags[2]
        else:
            style = filename_tags[0] if len(filename_tags) >= 1 else UNKNOWN_STYLE
            source = " ".join(filename_tags[1:]) if len(filename_tags) >= 2 else UNKNOWN_SOURCE

        source = source or UNKNOWN_SOURCE
        character = f"“{character_text}”字" if character_text else UNKNOWN_CHARACTER
        source_style_descriptor = f"取法{source}的{style}"

        return CaptionParts(
            raw=raw,
            character_text=character_text,
            character=character,
            style=style,
            source=source,
            source_style_descriptor=source_style_descriptor,
        )

    def format_caption(self, row: Dict[str, str]) -> str:
        parts = self.extract_caption_parts(row)
        return self.caption_template.format(
            raw=parts.raw,
            character_text=parts.character_text,
            character=parts.character,
            source=parts.source,
            author=parts.source,
            style=parts.style,
            source_style_descriptor=parts.source_style_descriptor,
            style_descriptor=parts.source_style_descriptor,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        image_path = self.resolve_image_path(row)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return encode_caption(self.captions[idx]), image


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_autoencoder_kl(args):
    try:
        from diffusers.models import AutoencoderKL
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Using --image_backend vae requires diffusers. Install it or run with --image_backend patch for debugging.") from exc

    load_kwargs = {}
    if args.vae_subfolder:
        load_kwargs["subfolder"] = args.vae_subfolder

    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path, **load_kwargs)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def get_vae_scaling_factor(vae: nn.Module, fallback: float) -> float:
    config = getattr(vae, "config", None)
    return float(getattr(config, "scaling_factor", fallback))


def get_latent_shape(args) -> Tuple[int, int]:
    downsample_factor = args.patch_size if args.image_backend == "patch" else args.vae_downsample_factor
    latent_size = args.image_size // downsample_factor
    return (latent_size, latent_size)


def build_model(args) -> Transfusion:
    if args.image_backend == "patch":
        latent_size = args.image_size // args.patch_size
        dim_latent = 3 * args.patch_size * args.patch_size
        modality_encoder = PatchEncoder(args.patch_size)
        modality_decoder = PatchDecoder(args.patch_size, channels=3)
    elif args.image_backend == "vae":
        vae = load_autoencoder_kl(args)
        latent_size = args.image_size // args.vae_downsample_factor
        dim_latent = int(getattr(vae.config, "latent_channels", args.vae_latent_channels))
        scaling_factor = get_vae_scaling_factor(vae, args.vae_scaling_factor)
        modality_encoder = VAEEncoder(vae, scaling_factor=scaling_factor, sample_latents=args.vae_sample_latents)
        modality_decoder = VAEDecoder(vae, scaling_factor=scaling_factor)
    else:
        raise ValueError(f"Unknown --image_backend: {args.image_backend}")

    return Transfusion(
        num_text_tokens=256,
        dim_latent=dim_latent,
        channel_first_latent=True,
        modality_default_shape=(latent_size, latent_size),
        modality_num_dim=2,
        modality_encoder=modality_encoder,
        modality_decoder=modality_decoder,
        add_pos_emb=True,
        velocity_consistency_loss_weight=args.velocity_consistency_loss_weight,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        transformer=dict(
            dim=args.dim,
            depth=args.depth,
            dim_head=args.dim_head,
            heads=args.heads,
            dropout=args.dropout,
            use_flex_attn=args.use_flex_attn,
        ),
    )


def build_image_prompt(model: Transfusion, caption_tokens: torch.Tensor, latent_shape: Tuple[int, int]) -> List[torch.Tensor]:
    device = model.device
    shape_text = ",".join(map(str, latent_shape))
    return [
        caption_tokens.to(device),
        torch.tensor([model.meta_id], device=device),
        model.char_tokenizer(shape_text, device=device),
        torch.tensor([model.som_ids[0]], device=device),
    ]


def save_validation_prompts(dataset: CalliffusionCsvDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for idx, (row, caption) in enumerate(zip(dataset.rows, dataset.captions)):
            rel_path = normalize_rel_path(row.get(dataset.image_column, row.get("path", "")))
            handle.write(f"{idx}\t{rel_path}\t{caption}\n")


@torch.no_grad()
def save_validation_samples(
    model: Transfusion,
    dataset: CalliffusionCsvDataset,
    step: int,
    samples_dir: Path,
    *,
    max_samples: int,
    latent_shape: Tuple[int, int],
    modality_steps: int,
) -> None:
    was_training = model.training
    model.eval()
    samples_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(min(max_samples, len(dataset))):
        caption_tokens, target = dataset[idx]
        prompt = build_image_prompt(model, caption_tokens, latent_shape)
        sampled = model.sample(
            prompt=prompt,
            max_length=math.prod(latent_shape) + 8,
            fixed_modality_shape=latent_shape,
            modality_steps=modality_steps,
            text_temperature=0.0,
        )

        generated = None
        for item in sampled:
            if isinstance(item, tuple) and item[0] == 0:
                generated = item[1].detach().cpu()
                break

        if generated is None:
            continue

        rows.extend([target.cpu(), generated])

    if not rows:
        logging.warning("No validation samples were generated at step %s.", step)
        if was_training:
            model.train()
        return

    grid = torch.stack(rows)
    save_image(grid, samples_dir / f"step_{step:06d}.png", nrow=2)
    if was_training:
        model.train()


def save_checkpoint(
    accelerator,
    model,
    ema_model,
    optimizer,
    checkpoint_dir: Path,
    *,
    epoch: int,
    step: int,
    args,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": vars(args),
    }

    if ema_model is not None:
        checkpoint["ema"] = ema_model.state_dict()

    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
    accelerator.save(checkpoint, checkpoint_path)

    latest_path = checkpoint_dir / "latest_checkpoint.txt"
    latest_path.write_text(str(checkpoint_path), encoding="utf-8")
    return checkpoint_path


def checkpoint_step(checkpoint_path: Path) -> int:
    match = re.search(r"step_(\d+)\.pt$", checkpoint_path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(path: Path) -> Path:
    path = path.expanduser()

    if path.is_file():
        return path

    search_root = path
    candidates = []

    if path.is_dir():
        candidates.extend(path.glob("step_*.pt"))
        candidates.extend(path.glob("checkpoints/step_*.pt"))
        candidates.extend(path.glob("*/checkpoints/step_*.pt"))
        candidates.extend(path.glob("calliffusion_transfusion/*/checkpoints/step_*.pt"))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching step_*.pt found under {search_root}.")

    return max(candidates, key=lambda candidate: (checkpoint_step(candidate), candidate.stat().st_mtime))


@torch.no_grad()
def compute_validation_loss(model, dataloader, accelerator, max_batches: int = 0) -> float:
    was_training = model.training
    model.eval()

    losses = []
    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        loss = model(batch)
        gathered = accelerator.gather_for_metrics(loss.detach().reshape(1))
        losses.append(gathered.float().cpu())

    if was_training:
        model.train()

    if not losses:
        return float("nan")

    return torch.cat(losses).mean().item()


def resolve_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default=None, help="Root directory that contains the calliffusion dataset/ folder.")
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--image_column", default="path")
    parser.add_argument("--caption_template", default=CalliffusionCsvDataset.DEFAULT_CAPTION_TEMPLATE)

    parser.add_argument("--image_backend", default="vae", choices=("vae", "patch"))
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--vae_model_name_or_path", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--vae_subfolder", default=None)
    parser.add_argument("--vae_downsample_factor", type=int, default=8)
    parser.add_argument("--vae_latent_channels", type=int, default=4)
    parser.add_argument("--vae_scaling_factor", type=float, default=0.18215)
    parser.add_argument("--vae_sample_latents", action="store_true")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_flex_attn", action="store_true")
    parser.add_argument("--reconstruction_loss_weight", type=float, default=0.0)
    parser.add_argument("--velocity_consistency_loss_weight", type=float, default=0.1)
    parser.add_argument("--use_velocity_consistency", action="store_true")

    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--mixed_precision", default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--ema_beta", type=float, default=0.99)
    parser.add_argument("--sample_every", type=int, default=1000)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--validation_loss_every", type=int, default=1000)
    parser.add_argument("--validation_loss_batches", type=int, default=0)
    parser.add_argument("--modality_steps", type=int, default=16)
    parser.add_argument("--checkpoint_every", type=int, default=2000)
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--resume_checkpoint", default=None, help="Path to a checkpoint, a run directory, an output root, or 'latest'.")
    parser.add_argument("--preview_captions", type=int, default=0)
    parser.add_argument("--preview_csvs", nargs="*", default=None)

    args = parser.parse_args()

    default_dataset_root = Path("~/Code/calligraphy_project").expanduser()
    args.dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else default_dataset_root

    dataset_dir = args.dataset_root / "dataset"
    args.train_csv = Path(args.train_csv).expanduser() if args.train_csv else dataset_dir / "train_260402.csv"

    if args.val_csv:
        args.val_csv = Path(args.val_csv).expanduser()
    else:
        validation_csv = dataset_dir / "validation.csv"
        legacy_typo_csv = dataset_dir / "validaiton.csv"
        args.val_csv = validation_csv if validation_csv.exists() else legacy_typo_csv

    if args.image_backend == "patch" and args.image_size % args.patch_size != 0:
        raise ValueError("--image_size must be divisible by --patch_size.")
    if args.image_backend == "vae" and args.image_size % args.vae_downsample_factor != 0:
        raise ValueError("--image_size must be divisible by --vae_downsample_factor.")

    return args


def main() -> None:
    args = resolve_args()
    set_seed(args.seed)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    output_base = Path(args.output_root).expanduser()
    output_root = output_base / "calliffusion_transfusion" / run_id
    samples_dir = output_root / "samples"
    checkpoints_dir = output_root / "checkpoints"
    tensorboard_dir = output_root / "tensorboard"

    transform = Compose(
        [
            Resize(args.image_size, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(args.image_size),
            ToTensor(),
        ]
    )

    train_rows = read_csv_rows(args.train_csv)
    val_rows = read_csv_rows(args.val_csv) if args.val_csv.exists() else train_rows[: max(args.sample_count, 1)]

    train_dataset = CalliffusionCsvDataset(
        train_rows,
        args.dataset_root,
        transform,
        image_column=args.image_column,
        caption_template=args.caption_template,
    )
    val_dataset = CalliffusionCsvDataset(
        val_rows,
        args.dataset_root,
        transform,
        image_column=args.image_column,
        caption_template=args.caption_template,
    )

    if args.preview_captions > 0:
        preview_csvs = args.preview_csvs or [args.train_csv]
        rng = random.Random(args.seed)
        for csv_path in preview_csvs:
            csv_path = Path(csv_path)
            rows = read_csv_rows(csv_path)
            preview_dataset = CalliffusionCsvDataset(
                rows,
                args.dataset_root,
                transform,
                image_column=args.image_column,
                caption_template=args.caption_template,
            )
            sample_count = min(args.preview_captions, len(preview_dataset))
            print(f"\nCSV: {csv_path}")
            for idx in rng.sample(range(len(preview_dataset)), sample_count):
                row = preview_dataset.rows[idx]
                print(f"[{idx}] path={normalize_rel_path(row.get(args.image_column, row.get('path', '')))}")
                print(f"    raw={derive_raw_description(row, args.image_column)}")
                print(f"    caption={preview_dataset.captions[idx]}")
        return

    from accelerate import Accelerator
    from torch.utils.tensorboard import SummaryWriter

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    writer = None
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)
        with (output_root / "args.json").open("w", encoding="utf-8") as handle:
            json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, ensure_ascii=False, indent=2)
        save_validation_prompts(val_dataset, output_root / "validation.txt")
        writer = SummaryWriter(log_dir=str(tensorboard_dir))

    model = build_model(args)
    ema_model = model.create_ema(beta=args.ema_beta) if args.ema_beta > 0.0 else None
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 0
    global_step = 0
    if args.resume_checkpoint:
        resume_path = find_latest_checkpoint(output_base if args.resume_checkpoint == "latest" else Path(args.resume_checkpoint).expanduser())
        logging.info("Loading checkpoint from %s", resume_path)
        torch.serialization.add_safe_globals([Path])
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if ema_model is not None and "ema" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        global_step = checkpoint.get("step", 0)
        if accelerator.is_main_process:
            (output_root / "resumed_from.txt").write_text(str(resume_path), encoding="utf-8")

    train_dataloader = model.create_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_dataloader = model.create_dataloader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, train_dataloader, val_dataloader)

    if ema_model is not None:
        ema_model.to(accelerator.device)

    latent_shape = get_latent_shape(args)

    if accelerator.is_main_process:
        logging.info("Training rows: %s; validation rows: %s", len(train_dataset), len(val_dataset))
        logging.info("Caption template: %s", args.caption_template)
        logging.info("Example caption: %s", train_dataset.captions[0])
        if start_epoch >= args.epochs:
            logging.warning("start_epoch=%s is not smaller than epochs=%s; no additional training epochs will run unless --epochs is increased.", start_epoch, args.epochs)

    stop_training = False
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        model.train()
        progress = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        progress.set_description(f"epoch {epoch}")

        for batch in progress:
            grad_norm = None
            with accelerator.accumulate(model):
                loss = model(
                    batch,
                    velocity_consistency_ema_model=ema_model if args.use_velocity_consistency else None,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if ema_model is not None:
                    ema_model.update()

                progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step)

                if accelerator.is_main_process and writer is not None:
                    writer.add_scalar("train/loss", loss.detach().float().item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("train/epoch", epoch, global_step)
                    if grad_norm is not None:
                        grad_norm_value = grad_norm.detach().float().item() if torch.is_tensor(grad_norm) else float(grad_norm)
                        writer.add_scalar("train/grad_norm", grad_norm_value, global_step)

                if args.validation_loss_every > 0 and global_step % args.validation_loss_every == 0:
                    accelerator.wait_for_everyone()
                    val_loss = compute_validation_loss(model, val_dataloader, accelerator, max_batches=args.validation_loss_batches)
                    if accelerator.is_main_process:
                        logging.info("step %s validation_loss %.6f", global_step, val_loss)
                        if writer is not None:
                            writer.add_scalar("validation/loss", val_loss, global_step)
                            writer.flush()

                if args.sample_every > 0 and global_step % args.sample_every == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        sample_model = ema_model.ema_model if ema_model is not None else accelerator.unwrap_model(model)
                        save_validation_samples(
                            sample_model,
                            val_dataset,
                            global_step,
                            samples_dir,
                            max_samples=args.sample_count,
                            latent_shape=latent_shape,
                            modality_steps=args.modality_steps,
                        )

                if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        checkpoint_path = save_checkpoint(
                            accelerator,
                            model,
                            ema_model,
                            optimizer,
                            checkpoints_dir,
                            epoch=epoch,
                            step=global_step,
                            args=args,
                        )
                        logging.info("Saved checkpoint to %s", checkpoint_path)
                        if writer is not None:
                            writer.add_text("checkpoint/latest", str(checkpoint_path), global_step)
                            writer.flush()

                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_training = True
                    break

        if stop_training:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        checkpoint_path = save_checkpoint(
            accelerator,
            model,
            ema_model,
            optimizer,
            checkpoints_dir,
            epoch=last_epoch,
            step=global_step,
            args=args,
        )
        logging.info("Saved final checkpoint to %s", checkpoint_path)
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
