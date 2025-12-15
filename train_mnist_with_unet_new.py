from shutil import rmtree
from pathlib import Path
from datetime import datetime
import math
import random

import torch
from torch import tensor, nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, lr_scheduler

from einops import rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from tqdm import tqdm

from transfusion_pytorch import Transfusion, print_modality_sample


# constants

SEED = 2026
random.seed(SEED)
torch.manual_seed(SEED)

AUTO_RESUME = False
GRAD_CLIP_NORM = 5.0e10


BASE_LR = 3e-4

# constant lr setting
WARMUP_STEPS = 0
MIN_LR_MULT = 1

IMAGE_FIRST = False
NUM_TRAIN_STEPS = 50_000
SAMPLE_EVERY = 5_000
CHECKPOINT_EVERY = 10_000

RUN_NAME = f'run-m-overfit-{datetime.now().strftime("%m%d-%H%M")}'

# add support for mac mps

mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
if mps_available:
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f'Using device: {device}')

def find_latest_checkpoint():
    run_dirs = sorted(
        Path('.').glob(f'{RUN_NAME}-*'),
        key = lambda path: path.stat().st_mtime,
        reverse = True
    )

    for run_dir in run_dirs:
        ckpt_dir = run_dir / 'checkpoints'
        checkpoints = sorted(
            ckpt_dir.glob('step-*.pt'),
            key = lambda path: path.stat().st_mtime,
            reverse = True
        )
        if checkpoints:
            return run_dir, checkpoints[0]

    return None, None

_, resume_checkpoint = (None, None)

if AUTO_RESUME:
    _, resume_checkpoint = find_latest_checkpoint()

run_folder = Path(f'./{RUN_NAME}')
writer = SummaryWriter(log_dir = str('logs' / run_folder))

is_resuming = resume_checkpoint is not None
if not is_resuming:
    rmtree(run_folder / 'results', ignore_errors = True)

results_folder = run_folder / 'results'
results_folder.mkdir(exist_ok = True, parents = True)

checkpoints_folder = run_folder / 'checkpoints'
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

    if WARMUP_STEPS > 0 and step <= WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)

    progress = (step - WARMUP_STEPS) / max(1, NUM_TRAIN_STEPS - WARMUP_STEPS)
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

    return MIN_LR_MULT + (1 - MIN_LR_MULT) * cosine_decay

# encoder / decoder

class Encoder(Module):
    def forward(self, x):
        x = rearrange(x, '... 1 (h p1) (w p2) -> ... (p1 p2) h w', p1 = 2, p2 = 2)
        return x * 2 - 1

class Decoder(Module):
    def forward(self, x):
        x = rearrange(x, '... (p1 p2) h w -> ... 1 (h p1) (w p2)', p1 = 2, p2 = 2, h = 14)
        return ((x + 1) * 0.5).clamp(min = 0., max = 1.)

model = Transfusion(
    num_text_tokens = 10,
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

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = T.PILToTensor()(pil)
        output =  tensor(labels), (digit_tensor / 255).float()

        if not IMAGE_FIRST:
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

optimizer = AdamW(model.parameters(), lr = BASE_LR)
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

    if start_step > NUM_TRAIN_STEPS:
        print('Checkpoint already covers configured NUM_TRAIN_STEPS, nothing to train.')
        writer.close()
        raise SystemExit

    checkpoint_device = checkpoint.get('device')
    if checkpoint_device and checkpoint_device != str(device):
        print(f'Checkpoint was trained on {checkpoint_device}, now loading on {device}')
else:
    print('Starting training from scratch')

# train loop


first_batch = next(iter_dl)
print('Example batch:')
for item in first_batch:
    first, second = item
    if IMAGE_FIRST:
        image = first
        label = second
    else:
        label = first
        image = second
    print(f' - label: {label}')

with tqdm(
    range(start_step, NUM_TRAIN_STEPS + 1),
    desc = 'training',
    mininterval = 1.0,
    initial = start_step - 1,
    total = NUM_TRAIN_STEPS
) as pbar:
    for step in pbar:
        model.train()

        loss = model(first_batch)
        # loss = model(next(iter_dl))
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

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

        if divisible_by(step, SAMPLE_EVERY):
            one_multimodal_sample = ema_model.sample(max_length = 384)

            print_modality_sample(one_multimodal_sample)

            if len(one_multimodal_sample) < 2:
                continue

            if IMAGE_FIRST:
                _, maybe_image, maybe_label = one_multimodal_sample
            else:
                maybe_label, maybe_image, *_ = one_multimodal_sample

            print(f'[debug] all maybe_label: {maybe_label}')
            print(f'[debug] all rest token: {_}')
            filename = f'{step}.{maybe_label[1].item()}.png'

            image_tensor = maybe_image[1].detach().cpu()

            save_image(
                image_tensor,
                str(results_folder / filename),
            )

            writer.add_image(
                'samples/validaiton',
                image_tensor,
                global_step = step,
                dataformats = 'CHW'
            )

        if divisible_by(step, CHECKPOINT_EVERY):
            save_checkpoint(step)

writer.close()
