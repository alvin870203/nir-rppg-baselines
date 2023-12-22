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
val_list = ('Subject1_motion_940',)
rppg_labels_diff_std = 6.969092845916748

# training related
# the number of examples per iter:
# 128 batch_size * 1 grad_accum = 128 clips/iter
# MR-NIRP_Indoor has 58,718 clips (when window size=2, stride=1), so 1 epoch ~= 458.7 iters
max_iters = 50000
gradient_accumulation_steps = 1
train_batch_size = 128

# evaluation related
eval_interval = 500  # unit: iters; keep frequent because we'll overfit
eval_batch_size = 128

# logging related
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
out_dir = 'out/Mean-MR-NIRP_Indoor/' + timestamp
wandb_log = True
wandb_project = 'MR-NIRP_Indoor'
wandb_run_name = 'Mean-' + timestamp
log_interval = 100  # unit: iters; don't print too often
always_save_checkpoint = False  # we expect to overfit on this small dataset, so only save when val improves

# model related
model_name = 'Mean'
out_dim = 1

# system related
# num_workers = 1
