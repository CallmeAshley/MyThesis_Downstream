########## 저장 경로 이름 순서: loss_aug

########## loss change  choose one of dice, ce, wce, ce+dice, wce+dice


############ 학습

# pretrain='/mnt/LSH/spine/mae_patch4_0.75/Best_acc.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_patch4_0.75_Best_acc/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
############# infer

resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp8/model'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp8/'
gpu='2'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --gpus $gpu

# ------------------------------------------------------------------------------------------------------
############ 학습

# pretrain='/mnt/LSH/spine/mae_patch4_0.75/Epoch300.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_patch4_0.75_Epoch300/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
############# infer

resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp9/model/'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp9/'
gpu='2'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --gpus $gpu
# ------------------------------------------------------------------------------------------------------

resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp10/model/'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp10/'
gpu='2'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --gpus $gpu
# ------------------------------------------------------------------------------------------------------

resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp11/model/'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp11/'
gpu='2'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --gpus $gpu
# ------------------------------------------------------------------------------------------------------
############ 학습

# pretrain='/mnt/LSH/spine/mae_patch4_0.75/Epoch400.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_patch4_0.75_Epoch400/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
############# infer

resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp12/model/'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp12/'
gpu='2'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --gpus $gpu

# ------------------------------------------------------------------------------------------------------
############ 학습

# pretrain='/mnt/LSH/spine/mae_patch4_0.75/Epoch500.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_patch4_0.75_Epoch500/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp4/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ori_jig_exp4/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.50/Epoch70.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch70/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch70/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch70/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.50/Epoch80.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch80/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch80/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch80/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.50/Epoch90.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch90/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch90/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch90/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.50/Epoch100.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch100/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch100/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.50_Epoch100/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir


# pretrain='/mnt/LSH/spine/maemasking_0.25/Best_acc.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Best_acc/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Best_acc/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Best_acc/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir 

# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.25/Epoch40.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch40/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch40/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch40/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir
# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.25/Epoch25.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch50/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch25/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch50/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.25/Epoch60.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch60/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch60/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch60/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.25/Epoch70.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch70/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch70/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch70/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.25/Epoch80.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch80/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch80/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch80/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.25/Epoch90.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch90/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch90/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch90/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# # ------------------------------------------------------------------------------------------------------
# ############ 학습

# pretrain='/mnt/LSH/spine/maemasking_0.25/Epoch100.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch100/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 
# ############# infer

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch100/model/'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/maemasking_0.25_Epoch100/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir








# loss_type="ce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_contrast/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_contrast/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_contrast/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################


# loss_type="dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_plain/1000_400_9'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_plain/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_plain/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_plain/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# # #########################################################


# loss_type="ce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_plain/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_plain/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_plain/1500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_plain/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################


# loss_type="wce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_plain/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_plain/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_plain/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_plain/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# # #########################################################


# loss_type="ce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_plain/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_plain/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_plain/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_plain/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# # #########################################################


# loss_type="wce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_plain/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_plain/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_plain/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_plain/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# # #########################################################


# loss_type="dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_rot/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_rot/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_rot/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_rot/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# # #########################################################


# loss_type="ce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_rot/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_rot/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_rot/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_rot/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# # #########################################################


# loss_type="wce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_rot/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_rot/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_rot/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_rot/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# # #########################################################


# loss_type="ce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_rot/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_rot/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_rot/1500_400_9'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_rot/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################


# loss_type="wce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_rot/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_rot/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_rot/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_rot/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################


# loss_type="dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_contrast/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_contrast/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed



# loss_type="dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_contrast/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed



# loss_type="dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_contrast/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# #########################################################


# loss_type="wce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_contrast/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_contrast/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_contrast/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_contrast/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################


# loss_type="ce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_contrast/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_contrast/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_contrast/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="ce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_contrast/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed
# #########################################################


# loss_type="wce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_contrast/1000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_contrast/1000_500_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed



# loss_type="wce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_contrast/1500_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# loss_type="wce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_contrast/2000_400_9/'
# resume=$output_dir"model/"
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --resume $resume --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# #########################################################