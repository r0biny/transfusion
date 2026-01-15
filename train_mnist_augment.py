from shutil import rmtree
from pathlib import Path
from datetime import datetime
import json
import math
import random

import torch
from torch import tensor, nn, Tensor
from typing import Iterable, Tuple
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, lr_scheduler

from einops import rearrange

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

from tqdm import tqdm

from transfusion_pytorch import Transfusion, print_modality_sample


# configuration
CONFIG = dict(
    SEED = 2026,
    NUM_VAL_SAMPLES = 5,
    AUTO_RESUME = False,
    GRAD_CLIP_NORM = 5.0,
    IF_OVERFIT = False,
    BASE_LR = 1e-3,
    WARMUP_STEPS = 2_000,
    MIN_LR_MULT = 0.1,
    NUM_TRAIN_STEPS = 50_000,
    SAMPLE_EVERY = 20,
    CHECKPOINT_EVERY = 10_000,
    IMAGE_FIRST = False,
    NUM_TEXT_TOKENS = 128,
    SHEAR_MIN_DEG = 4.0,
    SHEAR_MAX_DEG = 20.0,
    RUN_NAME = f'mnist-sheer-{datetime.now().strftime("%m%d-%H%M")}',
)


random.seed(CONFIG['SEED'])
torch.manual_seed(CONFIG['SEED'])

# add support for mac mps

mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
if mps_available:
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f'Using device: {device}')

def gen_random_prompt(num_samples: int) -> list[str]:
    prompts = []
    for _ in range(num_samples):
        digit = random.randint(0, 9)
        shear_deg = random.uniform(CONFIG['SHEAR_MIN_DEG'], CONFIG['SHEAR_MAX_DEG'])
        direction = 'right' if random.random() > 0.5 else 'left'
        magnitude = abs(shear_deg)
        if magnitude < 8:
            intensity = 'slightly'
        elif magnitude < 14:
            intensity = 'moderately'
        else:
            intensity = 'strongly'
        prompt = f'A handwritten digit {digit} slants {intensity} to the {direction}.'
        prompts.append(prompt)
    return prompts

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


def find_latest_checkpoint():
    run_dirs = sorted(
        Path('checkpoints').glob(f"{CONFIG['RUN_NAME']}*"),
        key = lambda path: path.stat().st_mtime,
        reverse = True
    )

    for run_dir in run_dirs:
        ckpt_dir = run_dir
        checkpoints = sorted(
            ckpt_dir.glob('step-*.pt'),
            key = lambda path: path.stat().st_mtime,
            reverse = True
        )
        if checkpoints:
            return run_dir, checkpoints[0]

    return None, None

_, resume_checkpoint = (None, None)

if CONFIG['AUTO_RESUME']:
    _, resume_checkpoint = find_latest_checkpoint()

run_folder = Path(CONFIG['RUN_NAME'])

# 创建 logs 文件夹，项目根目录下 ./logs/run-name-0102-1312
log_folder = Path('logs') / run_folder
log_folder.mkdir(parents = True, exist_ok = True)
writer = SummaryWriter(log_dir = str(log_folder))

# 保存 config.json 文件到 log 文件夹中
try:
    config_json_path = log_folder / 'config.json'
    with open(config_json_path, 'w', encoding='utf-8') as f:
        json.dump(CONFIG, f, indent=2, sort_keys=True)
except Exception as e:
    print(f'Warning: could not write config.json: {e}')

is_resuming = resume_checkpoint is not None

val_folder = Path('results') / run_folder
val_folder.mkdir(exist_ok = True, parents = True)

checkpoints_folder = Path('checkpoints') / run_folder
checkpoints_folder.mkdir(exist_ok = True, parents = True)


# functions

def divisible_by(num, den):
    return (num % den) == 0

def save_checkpoint(step):
    checkpoint = {
        'step': step,
        'model': model.state_dict(),
        'ema_model': ema_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'device': str(device)
    }
    torch.save(checkpoint, checkpoints_folder / f'step-{step}.pt')

def lr_lambda(step):
    if CONFIG['WARMUP_STEPS'] > 0 and step <= CONFIG['WARMUP_STEPS']:
        return step / max(1, CONFIG['WARMUP_STEPS'])

    progress = (step - CONFIG['WARMUP_STEPS']) / max(1, CONFIG['NUM_TRAIN_STEPS'] - CONFIG['WARMUP_STEPS'])
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

    return CONFIG['MIN_LR_MULT'] + (1 - CONFIG['MIN_LR_MULT']) * cosine_decay

def encode_text(text):
    text = text.encode('ascii', errors = 'ignore').decode('ascii')
    return tensor([*map(ord, text)], dtype = torch.long)

def decode_text_tokens(tokens):
    if tokens.ndim > 1:
        tokens = tokens.flatten()
    tokens = tokens[(tokens >= 0) & (tokens < CONFIG['NUM_TEXT_TOKENS'])]
    return ''.join([chr(int(t)) for t in tokens])


# Encoder
# 输入：MNIST 28×28 的单通道图像，像素值范围 [0, 1]
# Patch分割：将图像分割成 2×2 的小块
#           28×28 → 分成 14×14 个 2×2 块 → 4 个通道（2×2=4）
#           输出形状：(batch, 4, 14, 14)
# 归一化：x * 2 - 1 将像素值从 [0, 1] 映射到 [-1, 1]
# 将图像转换为潜在空间表示

class Encoder(Module):
    def forward(self, x):
        x = rearrange(x, '... 1 (h p1) (w p2) -> ... (p1 p2) h w', p1 = 2, p2 = 2)
        return x * 2 - 1

class Decoder(Module):
    def forward(self, x):
        x = rearrange(x, '... (p1 p2) h w -> ... 1 (h p1) (w p2)', p1 = 2, p2 = 2, h = 14)
        return ((x + 1) * 0.5).clamp(min = 0., max = 1.)

model = Transfusion(
    num_text_tokens = CONFIG['NUM_TEXT_TOKENS'],
    dim_latent = 4,
    modality_default_shape = (14, 14),
    modality_encoder = Encoder(),
    modality_decoder = Decoder(),
    pre_post_transformer_enc_dec = (
        nn.Conv2d(4, 64, 3, 2, 1),
        nn.ConvTranspose2d(64, 4, 3, 2, 1, output_padding = 1),
    ),
    add_pos_emb = True,
    modality_num_dim = 2,
    channel_first_latent = True,
    transformer = dict(
        dim = 64,
        depth = 4,
        dim_head = 32,
        heads = 8,
    )
).to(device)

ema_model = model.create_ema()

class MnistDataset(Dataset):
    def __init__(self, train = True):
        self.mnist = torchvision.datasets.MNIST(
            './data/mnist',
            train = train,
            download = True
        )
        self.shear_min_deg = CONFIG['SHEAR_MIN_DEG']
        self.shear_max_deg = CONFIG['SHEAR_MAX_DEG']

    def _sample_shear_deg(self):
        magnitude = random.uniform(self.shear_min_deg, self.shear_max_deg)
        direction = 1.0 if random.random() > 0.5 else -1.0
        return direction * magnitude

    def _describe_shear(self, digit, shear_deg):
        magnitude = abs(shear_deg)
        if magnitude < 8:
            intensity = 'slightly'
        elif magnitude < 14:
            intensity = 'moderately'
        else:
            intensity = 'strongly'
        direction = 'right' if shear_deg > 0 else 'left'
        return f'A handwritten digit {digit} slants {intensity} to the {direction}.'

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        shear_deg = self._sample_shear_deg()
        pil = TF.affine(
            pil,
            angle = 0.0,
            translate = (0, 0),
            scale = 1.0,
            shear = (shear_deg, 0.0),
            interpolation = InterpolationMode.BILINEAR,
            fill = 0
        )
        digit_tensor = T.PILToTensor()(pil)
        description = self._describe_shear(labels, shear_deg)
        output = encode_text(description), (digit_tensor / 255).float()

        if not CONFIG['IMAGE_FIRST']:
            return output

        first, second = output
        return second, first

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

def collate_fn(data):
    data = [*map(list, data)]
    return data

dataset = MnistDataset()
dataloader = model.create_dataloader(dataset, batch_size = 32, shuffle = True)

iter_dl = cycle(dataloader)

optimizer = AdamW(model.parameters(), lr = CONFIG['BASE_LR'])
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda)

start_step = 1

if is_resuming and resume_checkpoint is not None:
    checkpoint = torch.load(resume_checkpoint, map_location = device)
    model.load_state_dict(checkpoint['model'])
    ema_model.load_state_dict(checkpoint['ema_model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    start_step = int(checkpoint.get('step', 0)) + 1

    print(f'Resuming from {resume_checkpoint} at step {start_step - 1}')

    if start_step > CONFIG['NUM_TRAIN_STEPS']:
        print('Checkpoint already covers configured NUM_TRAIN_STEPS, nothing to train.')
        writer.close()
        raise SystemExit

    checkpoint_device = checkpoint.get('device')
    if checkpoint_device and checkpoint_device != str(device):
        print(f'Checkpoint was trained on {checkpoint_device}, now loading on {device}')
else:
    print('Starting training from scratch')

# train loop

if CONFIG['IF_OVERFIT']:
    first_batch = next(iter_dl)
    print('Overfitting on a single batch:')
    for item in first_batch:
        first, second = item
        if CONFIG['IMAGE_FIRST']:
            image = first
            label = second
        else:
            label = first
            image = second
        print(f' - label: {label}')

with tqdm(
    range(start_step, CONFIG['NUM_TRAIN_STEPS'] + 1),
    desc = 'training',
    mininterval = 1.0,
    initial = start_step - 1,
    total = CONFIG['NUM_TRAIN_STEPS']
) as pbar:
    for step in pbar:
        model.train()

        if CONFIG['IF_OVERFIT']:
            loss = model(first_batch)
        else:
            loss = model(next(iter_dl))
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['GRAD_CLIP_NORM'])

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        ema_model.update()

        loss_item = loss.item()

        # lr = optimizer.param_groups[0]['lr']
        lr = scheduler.get_last_lr()[0]
        writer.add_scalar('train/loss', loss_item, step)
        writer.add_scalar('train/lr', lr, step)
        writer.add_scalar('train/grad_norm', float(grad_norm), step)

        pbar.set_postfix(loss = f'{loss_item:.3f}', lr = f'{lr:.2e}')

        # eval

        if divisible_by(step, CONFIG['SAMPLE_EVERY']):
            model.eval()
            with torch.no_grad():
                random_prompt = gen_random_prompt(num_samples = CONFIG['NUM_VAL_SAMPLES'])
                for i, prompt in enumerate(random_prompt):

                    text_tokens = encode_text(prompt).unsqueeze(0).to(device)
                    sample = ema_model.sample(
                        prompt = text_tokens,
                        max_length = 384
                    )
                    print_modality_sample(sample)

                    file_name = f'step-{step}-prompt-{prompt}.png'
                    modality_type, image = extract_first_modality(sample)
                    if image is None:
                        print(f'[warn] no modality found for prompt: {prompt}')
                        continue
                    if image.ndim == 4 and image.shape[0] == 1:
                        image = image[0]

                    save_image(
                        image.detach().cpu(),
                        str(val_folder / file_name)
                    )

                    writer.add_image(
                        'samples/validaiton',
                        image,
                        global_step = step,
                        dataformats = 'CHW'
                    )

        if divisible_by(step, CONFIG['CHECKPOINT_EVERY']):
            save_checkpoint(step)

writer.close()
