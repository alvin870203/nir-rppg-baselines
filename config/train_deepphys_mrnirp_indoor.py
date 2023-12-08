# Config for training a dummy model on MR-NIRP Indoor dataset

# data related
dataset_name = 'MR-NIRP_Indoor'
window_size = 2  # unit: frames
window_stride = 1  # unit: frames
img_size_h = 36
img_size_w = 36
video_fps = 30.
ppg_fps = 60.

# training related
max_epochs = 50
train_batch_size = 128

# evaluation related
eval_interval = 1  # unit: epochs
eval_batch_size = 128

# logging related
wandb_log = False
wandb_project = 'MR-NIRP_Indoor'
wandb_run_name = 'DeepPhys'
log_interval = 1  # unit: epochs

# model related
model_name = 'DeepPhys'
out_dim = 1
bias = False

# system related
compile = False
