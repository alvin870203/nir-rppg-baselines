import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataloaders.mrnirp_indoor import MRNIRPIndoorDatasetConfig, MRNIRPIndoorDataset
from models.dummy import DummyConfig, Dummy


# -----------------------------------------------------------------------------
# default config values designed to train a dummy model on a dummy dataset
# data related
dataset = 'MR-NIRP_Indoor'
window_size = 2  # unit: frames
window_stride = 1  # unit: frames
img_size_h = 256
img_size_w = 256
video_fps = 30
ppg_fps = 60
# training related
max_epochs = 2
train_batch_size = 2
# evaluation related
eval_interval = 1  # unit: epochs
eval_batch_size = 1
# logging related
out_dir = 'out'
wandb_log = True
wandb_project = 'MR-NIRP_Indoor'
wandb_run_name = 'dummy'
log_interval = 1  # unit: epochs
# model related
out_dim = 1
bias = False # do we use bias inside Conv2D and Linear layers?
# system related
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
num_workers = 4
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


# various inits, derived attributes, I/O setup
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# dataloader
dataset_args = dict(
    dataset_root_path=os.path.join('data', dataset),
    window_size=window_size,
    window_stride=window_stride,
    img_size_h=img_size_h,
    img_size_w=img_size_w,
    video_fps=video_fps,
    ppg_fps=ppg_fps
)
match dataset:
    case 'MR-NIRP_Indoor':
        dataset_config = MRNIRPIndoorDatasetConfig(**dataset_args)
        train_dataset = MRNIRPIndoorDataset(dataset_config, 'train')
        val_dataset = MRNIRPIndoorDataset(dataset_config, 'val')
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=True, num_workers=num_workers)
        print(f"train dataset: {len(train_dataset)} samples, {len(train_dataloader)} batches")
        print(f"val dataset: {len(val_dataset)} samples, {len(val_dataloader)} batches")
        print(*[(element.shape, element.dtype, element.max(), element.min()) if isinstance(element, torch.Tensor) else element
                for element in next(iter(train_dataloader))],
              sep='\n')
    case _:
        raise ValueError(f"Unknown dataset: {dataset}")


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
epoch = 0
best_val_loss = 1e9


# model init
model_args = dict(
    img_size_h=img_size_h,
    img_size_w=img_size_w,
    out_dim=out_dim,
    bias=bias
)
# TODO: init from scratch or resume from checkpoint
print("Initializing a new model from scratch")
model_config = DummyConfig(**model_args)
model = Dummy(model_config)
model.to(device)


# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


# TODO: optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0


# TODO: learning rate decay scheduler (cosine with warmup)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# training loop
t0 = time.time()
for epoch in range(max_epochs):
    # TODO: determine and set the learning rate for this iteration


    # TODO: evaluate the loss on train/val sets and write checkpoints


    # train
    # forward backward update, using the GradScaler if data type is float16 and
    # TODO: with optional gradient accumulation to simulate larger batch size
    model.train()
    losses = torch.zeros(len(train_dataloader))
    for batch_idx, (nir_imgs, ppg_signals) in enumerate(tqdm(train_dataloader, desc=f'train epoch{epoch}')):
        # print(f'train: {epoch=}, {batch_idx=}')
        if device_type == 'cuda':
            # pin arrays nir_imgs,ppg_signals, which allows us to move them to GPU asynchronously (non_blocking=True)
            nir_imgs = nir_imgs.pin_memory().to(device, non_blocking=True)
            ppg_signals = ppg_signals.pin_memory().to(device, non_blocking=True)
        else:
            nir_imgs, ppg_signals = nir_imgs.to(device), ppg_signals.to(device)
        with ctx:
            logits, loss = model(nir_imgs, ppg_signals)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        # TODO: clip the gradient
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        # get loss as float. note: this is a CPU-GPU sync point
        losses[batch_idx] = loss.item()
    train_loss = losses.mean().item()


    # eval
    with torch.no_grad():
        model.eval()
        losses = torch.zeros(len(val_dataloader))
        for batch_idx, (nir_imgs, ppg_signals) in enumerate(tqdm(val_dataloader, desc=f'val epoch{epoch}')):
            # print(f'val: {epoch=}, {batch_idx=}')
            if device_type == 'cuda':
                # pin arrays nir_imgs,ppg_signals, which allows us to move them to GPU asynchronously (non_blocking=True)
                nir_imgs = nir_imgs.pin_memory().to(device, non_blocking=True)
                ppg_signals = ppg_signals.pin_memory().to(device, non_blocking=True)
            else:
                nir_imgs, ppg_signals = nir_imgs.to(device), ppg_signals.to(device)
            with ctx:
                logits, loss = model(nir_imgs, ppg_signals)
            losses[batch_idx] = loss.item()
        val_loss = losses.mean().item()


    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if epoch % log_interval == 0:
        # TODO: estimate mfu
        print(f"epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, time {dt*1000:.2f}ms")


