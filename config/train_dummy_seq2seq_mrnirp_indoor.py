# Config for training a dummy model on MR-NIRP Indoor dataset

# data related
dataset_name = 'MR-NIRP_Indoor'
window_size = 30  # unit: frames
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
max_epochs = 10
train_batch_size = 32

# evaluation related
eval_interval = 1  # unit: epochs
eval_batch_size = 32

# logging related
wandb_log = False
wandb_project = 'MR-NIRP_Indoor'
wandb_run_name = 'Dummy'
log_interval = 1  # unit: epochs

# model related
model_name = 'Dummy'
out_dim = 30
bias = False
