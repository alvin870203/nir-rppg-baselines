# Config for training a dummy model on MR-NIRP Indoor dataset

# data related
dataset_name = 'MR-NIRP_Indoor'
window_size = 2  # unit: frames
window_stride = 1  # unit: frames
img_size_h = 36
img_size_w = 36
video_fps = 30.
ppg_fps = 60.
train_list = (                       'Subject1_still_940', 'Subject2_motion_940', 'Subject2_still_940',
              'Subject3_motion_940', 'Subject3_still_940', 'Subject4_motion_940', 'Subject4_still_940',
                                     'Subject5_still_940', 'Subject6_motion_940', 'Subject6_still_940',
              'Subject7_motion_940', 'Subject7_still_940', 'Subject8_motion_940', 'Subject8_still_940')
val_list = ('Subject1_motion_940')

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
dropout = 0.50

# optimizer related
learning_rate = 6e-4  # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = True  # whether to decay the learning rate
warmup_epochs = 5  # how many steps to warm up for
lr_decay_epochs = 50  # should be ~= max_epochs
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10

# system related
compile = False
