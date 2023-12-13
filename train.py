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
train_list = ()
val_list = ()
test_list = ()
test_window_size = 900  # unit: frames (30 seconds)
test_window_stride = 900  # unit: frames (non-overlapping)
max_heart_rate = 250  # unit: bpm
min_heart_rate = 40  # unit: bpm
# training related
init_from = 'scratch'  # 'scratch' or 'resume'
max_iters = 2
train_batch_size = 2
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
# evaluation related
eval_interval = 1  # unit: iters; should be at least iters/epoch in practice
eval_batch_size = 1
# logging related
out_dir = 'out'
wandb_log = True
wandb_project = 'MR-NIRP_Indoor'
wandb_run_name = 'dummy'
log_interval = 1  # unit: iters
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
# model related
model_name = 'Dummy'
out_dim = 1
bias = False  # do we use bias inside Conv2D and Linear layers?
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
# optimizer related
learning_rate = 6e-4  # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = False  # whether to decay the learning rate
warmup_iters = 1  # how many steps to warm up for
lr_decay_iters = 2  # should be ~= max_iters
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10
# system related
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
num_workers = 4
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------


# various inits, derived attributes, I/O setup
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# dataloader
dataset_args = dict(
    dataset_root_path=os.path.join('data', dataset_name),
    window_size=window_size,
    window_stride=window_stride,
    img_size_h=img_size_h,
    img_size_w=img_size_w,
    video_fps=video_fps,
    ppg_fps=ppg_fps,
    train_list=train_list,
    val_list=val_list,
    test_list=test_list,
    test_window_size=test_window_size,
    test_window_stride=test_window_stride,
    max_heart_rate=max_heart_rate,
    min_heart_rate=min_heart_rate
)
match dataset_name:
    case 'MR-NIRP_Indoor':
        dataset_config = MRNIRPIndoorDatasetConfig(**dataset_args)
        train_dataset = MRNIRPIndoorDataset(dataset_config, 'train')
        val_dataset = MRNIRPIndoorDataset(dataset_config, 'val')
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=train_batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=eval_batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True)
        print(f"train dataset: {len(train_dataset)} samples, {len(train_dataloader)} batches")
        print(f"val dataset: {len(val_dataset)} samples, {len(val_dataloader)} batches")
        print(*[(element.shape, element.dtype, element.max(), element.min()) if isinstance(element, torch.Tensor) else element
                for element in next(iter(train_dataloader))],
              sep='\n')
    case _:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


# model init
if init_from == 'scratch':
    # init a new model from scratch
    print(f"Initializing a new {model_name} model from scratch")
elif init_from == 'resume':
    print(f"Resuming training {model_name} from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    assert model_name == checkpoint['config']['model_name'], "model_name mismatch"
    assert dataset_name == checkpoint['config']['dataset_name'], "dataset_name mismatch"
else:
    pass  # FUTURE: init from pretrained

match model_name:
    case 'Dummy':
        model_args = dict(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            out_dim=out_dim,
            bias=bias
        )  # start with model_args from command line
        if init_from == 'scratch':
            model_config = DummyConfig(**model_args)
            model = Dummy(model_config)
        elif init_from == 'resume':
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['img_size_h', 'img_size_w', 'out_dim', 'bias']:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            model_config = DummyConfig(**model_args)
            model = Dummy(model_config, train_dataset)
    case 'DeepPhys':
        model_args = dict(
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            out_dim=out_dim,
            bias=bias,
            dropout=dropout
        )  # start with model_args from command line
        if init_from == 'scratch':
            # init a new model from scratch
            print(f"Initializing a new {model_name} model from scratch")
            model_config = DeepPhysConfig(**model_args)
            model = DeepPhys(model_config, train_dataset)
        elif init_from == 'resume':
            # force these config attributes to be equal otherwise we can't even resume training
            # the rest of the attributes (e.g. dropout) can stay as desired from command line
            for k in ['img_size_h', 'img_size_w', 'out_dim', 'bias']:
                model_args[k] = checkpoint_model_args[k]
            # create the model
            model_config = DeepPhysConfig(**model_args)
            model = DeepPhys(model_config, train_dataset)
        else:
            pass  # FUTURE: init from pretrained
    case _:
        raise ValueError(f"Unknown model: {model_name}")

if init_from == 'resume':
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)


# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))


# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory


# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in zip(['train', 'val'], [train_dataloader, val_dataloader]):
        losses = torch.zeros(len(loader))
        for batch_idx, (nir_imgs, ppg_labels) in enumerate(loader):
            if device_type == 'cuda':
                # pin arrays nir_imgs,ppg_labels, which allows us to move them to GPU asynchronously (non_blocking=True)
                nir_imgs = nir_imgs.to(device, non_blocking=True)
                ppg_labels = ppg_labels.to(device, non_blocking=True)
            else:
                nir_imgs, ppg_labels = nir_imgs.to(device), ppg_labels.to(device)
            with ctx:
                logits, loss = model(nir_imgs, ppg_labels)
            losses[batch_idx] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# generate heart rate & spectrum results
def generate_result():
    results = {}
    for split, dataset in zip(['train', 'val'], [train_dataset, val_dataset]):
        result = model.generate(dataset, device)
        results[split] = result
        np.savez(os.path.join(out_dir, f'{split}_result.npz'), **result)
    return results


# learning rate decay scheduler (cosine with warmup)
def get_lr(iter):
    # 1) linear warmup for warmup_iters steps
    if iter < warmup_iters:
        return learning_rate * iter / warmup_iters
    # 2) if ep > lr_decay_iters, return min learning rate
    if iter > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# training loop
train_dataiter = iter(train_dataloader)
nir_imgs, ppg_labels = next(train_dataiter)  # fetch the very first batch
if device_type == 'cuda':
    # pin arrays nir_imgs,ppg_labels, which allows us to move them to GPU asynchronously (non_blocking=True)
    nir_imgs = nir_imgs.to(device, non_blocking=True)
    ppg_labels = ppg_labels.to(device, non_blocking=True)
else:
    nir_imgs, ppg_labels = nir_imgs.to(device), ppg_labels.to(device)
t0 = time.time()
local_iter_num = 0  # number of iters in the lifetime of this process
pbar = tqdm(total=max_iters, initial=iter_num, dynamic_ncols=True)
while True:


    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        tqdm.write(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                # "mfu": FUTURE: estimate mfu and convert it to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                tqdm.write(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                results = generate_result()
                for split, result in results.items():
                    tqdm.write(f"results on {split} set:")
                    for subject_name, subject_result in result.items():
                        tqdm.write(f"  {subject_name}: BPM predict={subject_result['bpm_predicts']}, label={subject_result['bpm_labels']}")
    if iter_num == 0 and eval_only:
        break


    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(nir_imgs, ppg_labels)
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # and re-init iterator if needed
        try:
            nir_imgs, ppg_labels = next(train_dataiter)
        except StopIteration:
            train_dataiter = iter(train_dataloader)
            nir_imgs, ppg_labels = next(train_dataiter)
        if device_type == 'cuda':
            # pin arrays nir_imgs,ppg_labels, which allows us to move them to GPU asynchronously (non_blocking=True)
            nir_imgs = nir_imgs.to(device, non_blocking=True)
            ppg_labels = ppg_labels.to(device, non_blocking=True)
        else:
            nir_imgs, ppg_labels = nir_imgs.to(device), ppg_labels.to(device)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)


    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        # FUTURE: estimate mfu
        tqdm.write(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1
    pbar.update(1)

    # termination conditions
    if iter_num > max_iters:
        break

pbar.close()
