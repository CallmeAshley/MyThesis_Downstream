resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp2/epoch_100/model/Best_DSC.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp2/epoch_100/'

python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp2/epoch_200/model/Best_DSC.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp2/epoch_200/'

python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp2/epoch_300/model/Best_DSC.pth'
output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp2/epoch_300/'

python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/RandomGapNPuzzlueN/exp/exp/J_R_exp1/epoch_400.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_400/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir

# resume='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/RandomGapNPuzzlueN/exp/exp/J_R_exp1/epoch_500.pth'
# output_dir='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/J_R_exp1/epoch_500/'

# python /mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/infer.py --resume $resume --output_dir $output_dir





