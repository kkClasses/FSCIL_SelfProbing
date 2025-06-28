python train.py teen  -project teen -dataset cifar100 -dataroot CIFAR_DATA_DIR  -base_mode 'ft_cos'  -new_mode 'avg_cos' -lr_base 0.1
-decay 0.0005  -epochs_base 400 -batch_size_base 256  -schedule Cosine -tmax 400  -gpu '0' -temperature 16  -softmax_t 16  -shift_weight 0.1


python train.py teen  -project teen -dataset cifar100 -dataroot CIFAR_DATA_DIR  -base_mode 'ft_cos'  -new_mode 'avg_cos'  -gpu '0' -temperature 16 -softmax_t 16  -shift_weight 0.1 -trainTest 'test' -model_dir 'checkpoint/cifar100/teen/ft_cos-avg_cos-data_init-start_0/1017-13-25-36-435-Epo_400-Bs_256-sgd-Lr_0.1-decay0.0005-Mom_0.9-Max_400-NormF-T_16.00-tw_16.0-0.1-soft_proto/session0_max_acc.pth'

python train.py teen  -project teen -dataset cifar100 -dataroot CIFAR_DATA_DIR  -base_mode 'ft_cos'  -new_mode 'avg_cos'  -gpu '0' -temperature 16 -softmax_t 16  -shift_weight 0.1 -trainTest 'test' -lr_base 0.0001  -decay 0.0005  -epochs_new 5 -batch_size_new 0 -incremen
tal_train 1 -model_dir 'checkpoint/cifar100/teen/ft_cos-avg_cos-data_init-start_0/1017-13-25-36-435-Epo_400-Bs_256-sgd-Lr_0.1-decay0.0005-Mom_0.9-Max_400-NormF-T_16.00-tw_16.0-0.1-soft_proto/session0_max_acc.pth'

 python train.py teen  -project teen -dataset cifar100 -dataroot CIFAR_DATA_DIR  -base_mode 'ft_cos'  -new_mode 'avg_cos'  -g
pu '0' -temperature 16 -softmax_t 16  -shift_weight 0.1 -trainTest 'test' -lr_base 0.01 -lr_new 0.01  -decay 0.0005  -epochs_new 100 -batch_size_new 0 -incremental_train 0 -model_dir 'checkpoint/cifar100/teen/ft_cos-avg_cos-data_init-start_0/1023-13-19-09-808-Epo_400-Bs_256-sgd-Lr_0.1-decay0.0005-Mom_0.9-Max_600-NormF-T_16.00-tw_16.0-0.1-soft_proto/session0_max_acc.pth'


#SelfProbing
python train.py selfProb  -project selfProb  -dataset cifar100 -dataroot dataSet/CIFAR_DATA_DIR  -base_mode 'ft_cos'  -new_mode 'avg_cos'  -gpu '0' -temperature 16
 -trainTest 'train' -lr_base 0.0001 -decay 0.0005 -batch_size_new 256  -epochs_base 10 -schedule Cosine  -tmax  15 -alpha 0.9
-model_dir 'checkpoint/cifar100/teen/ft_cos-avg_cos-data_init-start_0/1223-18-20-41-423-Epo_600-Bs_256-sgd-Lr_0.1-decay0.0005-Mom_0.9-Max_600-NormF-T_16.00-tw_16.0-0.1-soft_proto/session0_max_acc.pth'