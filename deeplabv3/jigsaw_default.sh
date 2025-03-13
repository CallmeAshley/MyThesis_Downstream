



# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1300.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1300_seed7723/'
# seed=7723

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1300_seed7723/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1300_seed7723/'
# seed=7723

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed


# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch750.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_750_seed7723/'
# seed=7723

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_750_seed7723/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_750_seed7723/'
# seed=7723

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed


# pretrain='/mai_nas/LSH/SparK/exp/real_mae_jig_a1b05/resnet50_epoch1450.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b05_1450_seed7723/'
# seed=7723

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b05_1450_seed7723/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b05_1450_seed7723/'
# seed=7723

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed




# pretrain='/mai_nas/LSH/SparK/exp/mae_jig_a1b05/resnet50_withdecoder_epoch750.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_750_seed6343/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --output_dir $output_dir 

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_750_seed6343/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_750_seed6343/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir


# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1300.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1300_seed7723/'
# seed=7723

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1300_seed7723/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1300_seed7723/'
# seed=7723

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed


# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1300.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1300_seed9876/'
# seed=9876

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1300_seed9876/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1300_seed9876/'
# seed=9876

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed


pretrain='/mai_nas/LSH/RotNet/Mine/angles2/model/Last_model.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/RN_angles2/'

python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir 

resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/RN_angles2/model/Best_DSC.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/RN_angles2/'

python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir 



pretrain='/mai_nas/LSH/RotNet/Mine/angles16/model/Last_model.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/RN_angles16/'


python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir

resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/RN_angles16/model/Best_DSC.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/RN_angles16/'


python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir









#-----------------------------seed 실험


# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1200.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed1234/'
# seed=1234

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed1234/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed1234/'
# seed=1234

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed

# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1200.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed5678/'
# seed=5678

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed5678/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed5678/'
# seed=5678

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed

# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1200.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed8303/'
# seed=8303

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed8303/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed8303/'
# seed=8303

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed

# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1200.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed4190/'
# seed=4190

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed4190/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed4190/'
# seed=4190

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed


# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1200.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed9619/'
# seed=9619

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed4190/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed9619/'
# seed=9619

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed

# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1200.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed7723/'
# seed=7723

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed7723/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed7723/'
# seed=7723

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed

# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1200.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed1213/'
# seed=1213

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed1213/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed1213/'
# seed=1213

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed

# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1200.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed1016/'
# seed=1016

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed1016/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed1016/'
# seed=1016

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed

# pretrain='/mai_nas/LSH/SparK/exp/mae_jig/resnet50_epoch1200.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed2018/'
# seed=2018

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --random_seed $seed

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed2018/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b1_1200_seed2018/'
# seed=2018

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --random_seed $seed


#----------------------------------------------------------------------------------------------------------------------------------------------


# pretrain='/mai_nas/LSH/SparK/exp/basic/resnet50_epoch1550.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1550_batch128/'
# batch_size=128

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --batch_size $batch_size

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1550_batch128/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1550_batch128/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir


# pretrain='/mai_nas/LSH/SparK/exp/basic/resnet50_epoch1300.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1300_batch256/'
# batch_size=256

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --batch_size $batch_size

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1300_batch256/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1300_batch256/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir


# pretrain='/mai_nas/LSH/SparK/exp/basic/resnet50_epoch1000.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1000/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1000/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1000/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir


# pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/RandomGapNPuzzlueN/exp/default_exp_label4000/final.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/default_label4000/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/default_label4000/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/default_label4000/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir