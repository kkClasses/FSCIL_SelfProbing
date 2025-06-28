python train.py teen  -project teen  -dataset cub200 -dataroot CUB_DATA_DIR  -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.25 \
    -lr_base 0.004 -lr_new 0.1 -decay 0.0005 -epochs_base 400 schedule Milestone -milestones 50 100 150 200 250 300 \
    -gpu '3'  -temperature 32 -batch_size_base 128  -softmax_t 16  -shift_weight 0.5

python train.py teen  -project teen  -dataset cub200 -dataroot CUB_DATA_DIR  -base_mode 'ft_cos' -new_mode 'avg_cos' -gamma 0.25 \
    -lr_base 0.004 -lr_new 0.1 -decay 0.0005 -epochs_base 100 schedule Milestone -milestones 50 70 \
    -gpu '3'  -temperature 32 -batch_size_base 128  -softmax_t 16  -shift_weight 0.5

test
python train.py teen  -project teen -dataset cub200 -dataroot CUB_DATA_DIR  -base_mode 'ft_cos'  -new_mode 'avg_cos'  -gpu '0' -temperature 16 -softmax_t 16  -shift_weight 0.1 -trainTest 'test' -lr_base 0.0001  -decay 0.0005  -epochs_new 5 -batch_size_new 0 -incremental_train 0
-model_dir 'checkpoint/cub200/teen/ft_cos-avg_cos-data_init-start_0/1220-10-26-09-611-Epo_50-Bs_128-sgd-Lr_0.004-decay0.0005-Mom_0.9-MS_30_40-Gam_0.25-NormF-T_16.00-tw_16.0-0.5-soft_proto/session0_max_acc.pth'

#selfProbing

python train.py selfProb  -project selfProb -dataset cub200 -dataroot dataSet/CUB_DATA_DIR  -base_mode 'ft_cos'  -new_mode 'avg_cos'  -gpu '0' -temperature 16
-trainTest 'train' -lr_base 0.0001 -decay 0.0005 -batch_size_new 128  -epochs_base 3 -schedule Cosine  -tmax  15  -alpha 0.5
-model_dir 'checkpoint/cub200/teen/ft_cos-avg_cos-data_init-start_0/1224-16-06-47-682-Epo_90-Bs_128-sgd-Lr_0.004-decay0.0005-Mom_0.9-MS_30_50_70-Gam_0.25-NormF-T_16.00-tw_16.0-0.5-soft_proto/session0_max_acc.pth'
