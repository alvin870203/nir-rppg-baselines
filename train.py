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
# training related
init_from = 'scratch'  # 'scratch' or 'resume'
max_epochs = 2
train_batch_size = 2
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
# evaluation related
eval_interval = 1  # unit: epochs
eval_batch_size = 1
# logging related
out_dir = 'out'
wandb_log = True
wandb_project = 'MR-NIRP_Indoor'
wandb_run_name = 'dummy'
log_interval = 1  # unit: epochs
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
warmup_epochs = 1  # how many steps to warm up for
lr_decay_epochs = 2  # should be ~= max_epochs
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
    test_list=test_list
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
epoch_num = 0
best_val_loss = 1e9


# model init
if init_from == 'scratch':
    # init a new model from scratch
    print(f"Initializing a new {model_name} model from scratch")
elif init_from == 'resume':
    print(f"Resuming training {model_name} from {out_dir}")
    # TODO: assert model_name matches the checkpoint
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
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
    epoch_num = checkpoint['epoch_num']
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


# learning rate decay scheduler (cosine with warmup)
def get_lr(epoch):
    # 1) linear warmup for warmup_iters steps
    if epoch < warmup_epochs:
        return learning_rate * epoch / warmup_epochs
    # 2) if ep > lr_decay_epochs, return min learning rate
    if epoch > lr_decay_epochs:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (epoch - warmup_epochs) / (lr_decay_epochs - warmup_epochs)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# training loop
t0 = time.time()
local_epoch_num = 0  # number of epochs in the lifetime of this process
while True:


    # determine and set the learning rate for this iteration
    lr = get_lr(epoch_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


    # evaluate the loss on train/val sets and write checkpoints
    if epoch_num % eval_interval == 0:
        losses = {'train': 0.0, 'val': 0.0}  # TODO: implement this
        print(f"step {epoch_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "epoch": epoch_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                # "mfu": TODO: estimate mfu and convert it to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if epoch_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'epoch_num': epoch_num,
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if epoch_num == 0 and eval_only:
        break


    # train
    # forward backward update, using the GradScaler if data type is float16
    # and with optional gradient accumulation to simulate larger batch size
    model.train()
    losses = []
    for batch_idx, (nir_imgs, ppg_signals) in enumerate(tqdm(train_dataloader, desc=f'train epoch{epoch_num}')):

        if device_type == 'cuda':
            # pin arrays nir_imgs,ppg_signals, which allows us to move them to GPU asynchronously (non_blocking=True)
            nir_imgs = nir_imgs.to(device, non_blocking=True)
            ppg_signals = ppg_signals.to(device, non_blocking=True)
        else:
            nir_imgs, ppg_signals = nir_imgs.to(device), ppg_signals.to(device)
        with ctx:
            logits, loss = model(nir_imgs, ppg_signals)
            # FIXME: Do we need to scale the loss differently if the remaining steps are less than gradient_accumulation_steps?
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        if ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_dataloader)):
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            # FIXME: Do we need to scale the loss differently if the remaining steps are less than gradient_accumulation_steps?
            lossf = loss.item() * gradient_accumulation_steps
            losses.append(lossf)
    train_loss = np.mean(losses)


    # eval
    with torch.no_grad():
        model.eval()
        losses = torch.zeros(len(val_dataloader))
        for batch_idx, (nir_imgs, ppg_signals) in enumerate(tqdm(val_dataloader, desc=f'val epoch{epoch_num}')):
            # print(f'val: {epoch=}, {batch_idx=}')
            if device_type == 'cuda':
                # pin arrays nir_imgs,ppg_signals, which allows us to move them to GPU asynchronously (non_blocking=True)
                nir_imgs = nir_imgs.to(device, non_blocking=True)
                ppg_signals = ppg_signals.to(device, non_blocking=True)
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
    if epoch_num % log_interval == 0:
        # TODO: estimate mfu
        print(f"epoch {epoch_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, time {dt*1000:.2f}ms")
    epoch_num += 1
    local_epoch_num += 1

    # termination conditions
    if epoch_num > max_epochs:
        break
