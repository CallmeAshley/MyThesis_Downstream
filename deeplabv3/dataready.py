import numpy as np
import os
import SimpleITK as sitk



trainMR_path = '/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/train/MR/'
test1MR_path = '/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/test1/MR/'
test2MR_path = '/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/test2/MR/'

trainMR_list = os.listdir(trainMR_path)
test1MR_list = os.listdir(test1MR_path)
test2MR_list = os.listdir(test2MR_path)


trainMask_path = '/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/train/Mask/'
test1Mask_path = '/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/test1/Mask/'
test2Mask_path = '/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/test2/Mask/'

trainMask_list = os.listdir(trainMask_path)
test1Mask_list = os.listdir(test1Mask_path)
test2Mask_list = os.listdir(test2Mask_path)



for case in trainMR_list:
    
    img = sitk.ReadImage(trainMR_path+case)
    img = sitk.GetArrayFromImage(img)
    # a,_,_ = img.shape
    
    for i in range(5,9):
        np.save('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal/MR/'+case[:-7]+'_'+str(i)+'.npy', img[i,:,:])
        

for case in trainMask_list:
    
    img = sitk.ReadImage(trainMask_path+case)
    img = sitk.GetArrayFromImage(img)
    # a,_,_ = img.shape
    
    for i in range(5,9):
        np.save('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal/Mask/'+case[:-7]+'_'+str(i)+'.npy', img[i,:,:])
        
        
for case in test1MR_list:
    
    img = sitk.ReadImage(test1MR_path+case)
    img = sitk.GetArrayFromImage(img)
    # a,_,_ = img.shape
    
    for i in range(5,9):
        np.save('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal/MR/'+case[:-7]+'_'+str(i)+'.npy', img[i,:,:])
        
        
for case in test2MR_list:
    
    img = sitk.ReadImage(test2MR_path+case)
    img = sitk.GetArrayFromImage(img)
    # a,_,_ = img.shape
    
    for i in range(5,9):
        np.save('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test2_ideal/MR/'+case[:-7]+'_'+str(i)+'.npy', img[i,:,:])
        
        
        
for case in test1Mask_list:
    
    img = sitk.ReadImage(test1Mask_path+case)
    img = sitk.GetArrayFromImage(img)
    # a,_,_ = img.shape
    
    for i in range(5,9):
        np.save('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal/Mask/'+case[:-7]+'_'+str(i)+'.npy', img[i,:,:])
        
for case in test2Mask_list:
    
    img = sitk.ReadImage(test2Mask_path+case)
    img = sitk.GetArrayFromImage(img)
    # a,_,_ = img.shape
    
    for i in range(5,9):
        np.save('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test2_ideal/Mask/'+case[:-7]+'_'+str(i)+'.npy', img[i,:,:])


