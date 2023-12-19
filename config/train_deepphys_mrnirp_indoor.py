# Config for training a dummy model on MR-NIRP Indoor dataset

import time

# data related
dataset_name = 'MR-NIRP_Indoor'
window_size = 2  # unit: frames
window_stride = 1  # unit: frames
img_h = 36  # input image height of the model
img_w = 36  # input image width of the model
video_fps = 30.
ppg_fps = 60.
train_list = (                       'Subject1_still_940', 'Subject2_motion_940', 'Subject2_still_940',
              'Subject3_motion_940', 'Subject3_still_940', 'Subject4_motion_940', 'Subject4_still_940',
                                     'Subject5_still_940', 'Subject6_motion_940', 'Subject6_still_940',
              'Subject7_motion_940', 'Subject7_still_940', 'Subject8_motion_940', 'Subject8_still_940')
val_list = ('Subject1_motion_940')
crop_face_type = 'video_first'  # 'no', 'video_fist', 'window_first', 'every'
bbox_scale = 1.6

# transform related
video_freq_scale_range = (1.0, 1.0)  # augmented freq ~= freq * random.uniform(min, max), e.g., (0.7, 1.4)
video_freq_scale_p = 0.0  # probability of applying random video freq scale
window_hflip_p = 0.0
frame_shift = 0.0  # augmented bbox center_{x or y} = center_{x or y} + bbox_{w or h} * random.uniform(-max, max))
frame_shift_p = 0.0  # probability of applying random bbox shift
frame_scale_range = (1.0, 1.0)  # augmented bbox_scale = bbox_scale * random.uniform(min, max)
frame_scale_p = 0.0  # probability of applying random bbox scale

# training related
# the number of examples per iter:
# 128 batch_size * 1 grad_accum = 128 clips/iter
# MR-NIRP_Indoor has 58,718 clips (when window size=2, stride=1), so 1 epoch ~= 458.7 iters
max_iters = 25000
gradient_accumulation_steps = 1
train_batch_size = 128

# evaluation related
eval_interval = 500  # unit: iters; keep frequent because we'll overfit
eval_batch_size = 128

# logging related
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
out_dir = 'out/DeepPhys-MR-NIRP_Indoor/' + timestamp
wandb_log = False
wandb_project = 'MR-NIRP_Indoor'
wandb_run_name = 'DeepPhys-' + timestamp
log_interval = 100  # unit: iters; don't print too often
always_save_checkpoint = False  # we expect to overfit on this small dataset, so only save when val improves

# model related
model_name = 'DeepPhys'
out_dim = 1
bias = True
dropout = 0.50

# optimizer related
learning_rate = 6e-4  # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.0  # clip gradients at this value, or disable if == 0.0
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2500  # how many steps to warm up for
lr_decay_iters = 25000  # should be ~= max_iters
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10

# system related
compile = False
# num_workers = 1
