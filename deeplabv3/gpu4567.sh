pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/RandomGapNPuzzlueN/exp/exp/J_R_exp1/epoch_100.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_100/'
gpus='4,5,6,7'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --gpus $gpus
resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_100/model/Best_DSC.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_100/'
gpus='4,5,6,7'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --gpus $gpus



pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/RandomGapNPuzzlueN/exp/exp/J_R_exp1/epoch_200.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_200/'
gpus='4,5,6,7'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --gpus $gpus
resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_200/model/Best_DSC.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_200/'
gpus='4,5,6,7'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --gpus $gpus



pretrain='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/RandomGapNPuzzlueN/exp/exp/J_R_exp1/epoch_400.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_400/'
gpus='4,5,6,7'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/train_ce.py --pretrain $pretrain --output_dir $output_dir --gpus $gpus
resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_400/model/Best_DSC.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_400/'
gpus='4,5,6,7'
python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir --gpus $gpus




