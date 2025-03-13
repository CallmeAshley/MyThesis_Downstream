# pretrain='/mai_nas/LSH/SparK/exp/real_mae_jig_a1b05/resnet50_epoch1300.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b05_1300/'
# gpus=0,1
# workers=32


# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --gpus $gpus --workers $workers

resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b05_1300/model/Best_DSC.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/mae_jig_a1b05_1300/'
gpus=0
workers=32

python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --gpus $gpus --workers $workers



# pretrain='/mai_nas/LSH/SparK/exp/basic/resnet50_epoch1300.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1300_batch64/'
# batch_size=64

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --batch_size $batch_size

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1300_batch64/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1300_batch64/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir


# pretrain='/mai_nas/LSH/SparK/exp/basic/resnet50_epoch1300.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1300_batch128/'
# batch_size=128

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --batch_size $batch_size

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1300_batch128/model/Best_DSC.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/spark_resnet50_epoch1300_batch128/'

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