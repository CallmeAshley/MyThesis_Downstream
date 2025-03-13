########## 저장 경로 이름 순서: loss_aug

########## loss change  choose one of dice, ce, wce, ce+dice, wce+dice
pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/exp/puzzle9/Best_acc.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/puzzle9/'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir

pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/exp/puzzle16/Best_acc.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/puzzle16/'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir

# pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_LSPPX_0.75_pretrain/Epoch100.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_LSPPX_0.75_Epoch100/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir

# pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_LSPPX_0.75_pretrain/Epoch200.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_LSPPX_0.75_Epoch200/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir


# pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_LSPPX_0.75_pretrain/Epoch300.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_LSPPX_0.75_Epoch300/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir

# pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_LSPPX_0.75_pretrain/Epoch400.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_LSPPX_0.75_Epoch400/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir

# pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_LSPPX_0.75_pretrain/Epoch500.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_LSPPX_0.75_Epoch500/'
# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 


# pretrain="/mnt/LSH/Jigsaw/exp/rad_spine_noaug/Best_acc.pth"
# loss_type="ce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/rad_spine_noaug/'
# random_seed=6343

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="ce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_contrast/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="ce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_contrast/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="ce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_contrast/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_plain/1000_400_9'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_plain/1000_500_9'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_plain/1500_400_9'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_plain/2000_400_9'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="ce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_plain/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="ce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_plain/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="ce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_plain/1500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="ce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_plain/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

#########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="wce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_plain/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="wce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_plain/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="wce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_plain/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="wce"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_plain/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_plain/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_plain/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_plain/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_plain/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_plain/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_plain/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_plain/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='plain'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_plain/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_rot/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_rot/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_rot/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_rot/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="ce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_rot/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="ce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_rot/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="ce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_rot/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="ce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce_rot/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="wce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_rot/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="wce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_rot/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="wce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_rot/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="wce"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_rot/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_rot/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_rot/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_rot/1500_400_9'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_rot/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

#########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_rot/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_rot/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_rot/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='rot'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_rot/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

#########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_contrast/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_contrast/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_contrast/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/dice_contrast/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="wce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_contrast/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="wce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_contrast/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="wce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_contrast/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="wce"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce_contrast/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_contrast/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_contrast/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_contrast/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="ce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/ce+dice_contrast/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed
# #########################################################

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_400_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_contrast/1000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1000_500_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_contrast/1000_500_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/1500_400_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_contrast/1500_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed

# pretrain="/mai_nas/LSH/MRSpineSeg_Challenge_SMU/JigsawPuzzlesPytorch/2000_400_9/Best_acc.pth"
# loss_type="wce+dice"
# aug_type='contrast'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/wce+dice_contrast/2000_400_9/'
# random_seed=6395

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --loss_type $loss_type --aug_type $aug_type --random_seed $random_seed


# #########################################################
