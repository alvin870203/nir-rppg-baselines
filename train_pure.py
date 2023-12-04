import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from dataloaders.mrnirp_indoor import MRNIRPIndoorDatasetConfig, MRNIRPIndoorDataset
from models.Dummy import DummyConfig, Dummy
from models.DeepPhys import DeepPhysConfig, DeepPhys


# -----------------------------------------------------------------------------
# default config values designed to train a dummy model on a dummy dataset
# data related
dataset_name = 'MR-NIRP_Indoor'
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
model_name = 'Dummy'
out_dim = 1
bias = False # do we use bias inside Conv2D and Linear layers?
# system related
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
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


# dataloader
dataset_args = dict(
    dataset_root_path=os.path.join('data', dataset_name),
    window_size=window_size,
    window_stride=window_stride,
    img_size_h=img_size_h,
    img_size_w=img_size_w,
    video_fps=video_fps,
    ppg_fps=ppg_fps
)
match dataset_name:
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
        raise ValueError(f"Unknown dataset: {dataset_name}")


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
epoch = 0
best_val_loss = 1e9


# model init
match model_name:
    case 'Dummy':
        model_args = dict(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            out_dim=out_dim,
            bias=bias
        )
        # TODO: init from scratch or resume from checkpoint
        print(f"Initializing a new {model_name} model from scratch")
        model_config = DummyConfig(**model_args)
        model = Dummy(model_config)
    case 'DeepPhys':
        model_args = dict(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            out_dim=out_dim,
            bias=bias
        )
        # TODO: init from scratch or resume from checkpoint
        print(f"Initializing a new {model_name} model from scratch")
        model_config = DeepPhysConfig(**model_args)
        model = DeepPhys(model_config, train_dataset)
    case _:
        raise ValueError(f"Unknown model: {model_name}")
model.to(device)


# TODO: optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


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
    # forward backward update
    # TODO: with optional gradient accumulation to simulate larger batch size
    model.train()
    losses = torch.zeros(len(train_dataloader))
    for batch_idx, (nir_imgs, ppg_signals) in enumerate(tqdm(train_dataloader, desc=f'train epoch{epoch}')):
        # print(f'train: {epoch=}, {batch_idx=}')
        nir_imgs, ppg_signals = nir_imgs.to(device), ppg_signals.to(device)
        logits, loss = model(nir_imgs, ppg_signals)
        # backward pass
        loss.backward()
        # TODO: clip the gradient
        # step the optimizer
        optimizer.step()

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
            nir_imgs, ppg_signals = nir_imgs.to(device), ppg_signals.to(device)
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


