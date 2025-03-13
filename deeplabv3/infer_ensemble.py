from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

import os
import random
import time
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import matplotlib.pyplot as plt

import pandas as pd
import openpyxl

from model_ce import UNet
from dataset_ce import SpineDataset

import scipy.io as sio

from monai.transforms import *
from monai import transforms
from monai.losses import DiceLoss
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
import torch.backends.cudnn as cudnn
from utils import compute_dice_score

from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

from swin_unet import SwinUnet
import math

# import segmentation_models_pytorch as smp

def parse_args():
    parser = argparse.ArgumentParser(description='Spine Segmentation by Soohyun Lee in MAI-LAB')

    parser.add_argument('--resume', type=str, default='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/best/model/Best_DSC.pth', help='load pretrained model path')
    parser.add_argument('--resume2', type=str, default='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/RN_exp1_100epoch/model/Best_DSC.pth', help='load pretrained model path')
    parser.add_argument('--pretrain', type=str, default=None, help='load pretrained model path')
     
    parser.add_argument('--output_dir', type=str, default='./ensemble/',help='Directory name to save the model, log, config')


    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--val_batch_size', type=int, default=40, help='The size of validation batch')
    parser.add_argument('--training_size', default=(512,512))    
     
    parser.add_argument('--mode', type=str, default='infer')

    parser.add_argument('--loss_type', default='dice', type=str, help='ce, dice')    
    parser.add_argument('--aug_type', default='contrast', type=str, help='plain, rot, contrast')
    parser.add_argument('--lr_sch', default=True, type=bool, help='쓸지 말지')    


    parser.add_argument('--gpus',default='0,1' ,type=str, help='gpu num')
    parser.add_argument('--workers', default=32, type=int, help='number of workers for dataloader')
    parser.add_argument('--random_seed', default=6343, type=int, help='default 12345')
    
    args = parser.parse_args()
    
    return args

args = parse_args()

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed) # if use multi-GPU
np.random.seed(args.random_seed)
random.seed(args.random_seed)

if not os.path.exists(args.output_dir):
    
        os.makedirs(args.output_dir)
        os.makedirs(args.output_dir+'model')
        os.makedirs(args.output_dir+'test/matfile')
        os.makedirs(args.output_dir+'val')
        os.makedirs(args.output_dir+'log')
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model = UNet()
model = deeplabv3_resnet50(weights=None, num_classes=2)
model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


net = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=2)
net.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


if (args.resume is None) and (args.pretrain is None):
    model = nn.DataParallel(model)



if (args.resume is not None) or (args.pretrain):
    if args.resume is not None:
        state_dict = torch.load(args.resume)
    else:
        state_dict = torch.load(args.pretrain)


    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k if k.startswith('module.') else 'module.'+k
    #     new_state_dict[name] = v

    model = nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=False)
    
    
if args.resume2 is not None:
    state_dict = torch.load(args.resume2)        
    net = nn.DataParallel(net)
    net.load_state_dict(state_dict, strict=False)       
    
    
    
    
        
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))    


if args.loss_type == 'ce':
    loss_ce = nn.CrossEntropyLoss().to(device)
elif args.loss_type == 'dice':
    loss_dice = DiceLoss(include_background=False).to(device)




if args.lr_sch:
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)




if args.aug_type == 'plain': 
    transform_train = transforms.Compose([   
                                        Resized(keys=["img"], spatial_size=args.training_size, mode="bilinear"),
                                        Resized(keys=["label"], spatial_size=args.training_size, mode="nearest"),
                                        
                                        ToTensord(keys=["img",'label'])
                                        ])
    
elif args.aug_type == 'rot':
    transform_train = transforms.Compose([   
                                        Resized(keys=["img"], spatial_size=args.training_size, mode="bilinear"),
                                        Resized(keys=["label"], spatial_size=args.training_size, mode="nearest"),
                                        
                                        RandRotated(keys=["img", "label"], range_x=(-30,30), prob=0.5, mode=["bilinear", "nearest"], padding_mode="border"),
                                        ToTensord(keys=["img",'label'])
                                        ])
    
elif args.aug_type == 'contrast':
    
    transform_train = transforms.Compose([   
                                        Resized(keys=["img"], spatial_size=args.training_size, mode="bilinear"),
                                        Resized(keys=["label"], spatial_size=args.training_size, mode="nearest"),
                                        
                                        RandGaussianNoised(keys=['img'], mean=0, std=0.1, prob=0.2),
                                        RandGaussianSmoothd(keys=['img'], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=0.2),
                                        RandScaleIntensityd(keys=["img"], factors=0.25, prob=0.2),
                                        RandAdjustContrastd(keys=['img'], gamma=(0.75, 1.25), prob=0.2),
                                        ToTensord(keys=["img",'label'])
                                        ])


        
transform_valandtest = transforms.Compose([   
                                        Resized(keys=["img"], spatial_size=args.training_size, mode="bilinear"),
                                        # Resized(keys=["label"], spatial_size=args.training_size, mode="nearest"),
                                        
                                        ToTensord(keys=["img"])
                                        ])


transform_train.set_random_state(seed=args.random_seed)
transform_valandtest.set_random_state(seed=args.random_seed)


dataloader_train = DataLoader(SpineDataset(args, is_valid=False, is_test=False, transform=transform_train), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
dataloader_valid = DataLoader(SpineDataset(args, is_valid=True, is_test=False, transform=transform_valandtest), batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
dataloader_test = DataLoader(SpineDataset(args, is_valid=False, is_test=True, transform=transform_valandtest), batch_size=1, shuffle=False, pin_memory=True, num_workers=args.workers)


import torch.nn.functional as F

def resize_output_to_cropped_size(output, cropped_shape, is_label=None):
    """
    PyTorch를 사용하여 모델 출력을 크롭된 이미지 크기로 리사이즈하는 함수. 배치 차원을 고려합니다.

    :param output: 딥러닝 모델의 출력, 텐서 형식 (N, C, H, W)
    :param cropped_shape: 크롭된 이미지의 차원, 튜플 형식 (C, H', W')
    :return: 리사이즈된 모델 출력, 텐서 형식
    """
    # PyTorch의 interpolate 함수 사용 (리사이즈)
    # cropped_shape는 배치 차원을 포함하지 않으므로, output.size(0)을 사용하여 배치 크기를 유지
    if is_label==False:
        resized_output = F.interpolate(output, size=cropped_shape[1:], mode='bilinear', align_corners=False)

        return resized_output
    
    else:
        resized_output = F.interpolate(output, size=cropped_shape[1:], mode='nearest')

        return resized_output

def pad_to_original_size(resized_output, original_shape, bbox):
    """
    PyTorch를 사용하여 리사이즈된 모델 출력을 원본 이미지 크기로 zero padding하여 복원하는 함수. 배치 차원을 고려합니다.

    :param resized_output: 리사이즈된 모델 출력, 텐서 형식 (N, C, H', W')
    :param original_shape: 원본 이미지의 차원, 튜플 형식 (N, C, H, W)
    :param bbox: 크롭에 사용된 경계 상자 정보
    :return: 원본 차원으로 복원된 이미지, 텐서 형식
    """
    
    # 원본 크기의 빈 텐서 생성
    padded_output = torch.zeros(original_shape, dtype=resized_output.dtype, device=resized_output.device)
    
    # 리사이즈된 출력을 적절한 위치에 할당
    # 여기서는 모든 이미지에 대해 동일한 bbox를 적용하고 있습니다.
    for i in range(resized_output.size(0)):  # 배치 내의 각 이미지에 대해 반복
        padded_output[i, :, bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = resized_output[i]

    return padded_output

HD = HausdorffDistanceMetric(include_background=False, percentile=95)
SD = SurfaceDistanceMetric(include_background=False)

before = 0
after = 0
DSC_test_final = 0
a = 0
b = 0
c = 1
thre0 = 0
thre1 = 0
# HD_tst_per_epoch = 0
# SD_tst_per_epoch = 0

test_names = []
dsc_scores = []
precision_arr = []
recall_arr = []
specificity_arr = []
HD_arr = []
SD_arr = []

model.to(device)
net.to(device)

with torch.no_grad():
    model.eval()
    net.eval()

    for t, batch in enumerate(dataloader_test):
        
        print("")
        
        test_data, original_shape, cropped_shape, bbox, ori_img = batch
        # test_data = batch
        
        original_shape.insert(0, torch.tensor([2]))
        original_shape.insert(0, torch.tensor([1]))
        
        
        test_input = test_data['img'].to(device)
        test_label = test_data['label'].to(device)
        test_name = test_data['file_name']


        test_output = model(test_input)
        test_output_en = test_output['out']
        test_output_softmax = F.softmax(test_output_en, dim=1)
        
        test_output2 = net(test_input)
        test_output2_en = test_output2['out']
        test_output2_softmax = F.softmax(test_output2_en, dim=1)
        
        resized_output = resize_output_to_cropped_size(test_output_softmax, cropped_shape, is_label=False)
        restored_output = pad_to_original_size(resized_output, original_shape, bbox)
        
        resized_output2 = resize_output_to_cropped_size(test_output2_softmax, cropped_shape, is_label=False)
        restored_output2 = pad_to_original_size(resized_output2, original_shape, bbox)
        
        resized_label = resize_output_to_cropped_size(test_label, cropped_shape, is_label=True)
        restored_label = pad_to_original_size(resized_label, original_shape, bbox)


        test_output = restored_output
        test_output2 = restored_output2
        test_label= restored_label
        # test_output = test_output[:,1,:,:]
        # test_output2 = test_output2[:,1,:,:]
        # test_label=test_label[:,1,:,:]
        
        
        test_output[test_output>=0.5] = 1
        test_output[test_output<0.5] = 0
        
        test_output2[test_output2>=0.5] = 1
        test_output2[test_output2<0.5] = 0
        
        # sio.savemat(args.output_dir+'test/matfile/'+test_name[0]+'_output.mat',mdict={'output':restored_output[0,0,:,:].detach().cpu().numpy()})
        # sio.savemat(args.output_dir+'test/matfile/'+test_name[0]+'_input.mat',mdict={'input':ori_img[0,0,:,:].detach().cpu().numpy()})
        # sio.savemat(args.output_dir+'test/matfile/'+test_name[0]+'_label.mat',mdict={'label':restored_label[0,0,:,:].detach().cpu().numpy()})
        
        # test_output_flatten = test_output.flatten()
        # test_label_flatten = test_label.flatten()
                
        # cm = confusion_matrix(test_label_flatten.cpu().detach().numpy(), test_output_flatten.cpu().detach().numpy())   
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot()
        # plt.savefig(args.output_dir+'test/'+test_name[0]+'_confusion_matrix.png')
        # tn, fp, fn, tp = cm.ravel() 
        
        # tn, fp, fn, tp = cm.ravel() 
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # specificity = tn/(tn+fp)
        
        # precision_arr.append(precision)
        # recall_arr.append(recall)
        # specificity_arr.append(specificity)
        

        dsc_test = compute_dice_score(test_output[:,1,:,:], test_label[:,1,:,:])
        dsc_test = dsc_test.mean()
        
        dsc_test2 = compute_dice_score(test_output2[:,1,:,:], test_label[:,1,:,:])
        dsc_test2 = dsc_test2.mean()
        
        
        weight1 = math.exp(dsc_test) / (math.exp(dsc_test) + math.exp(dsc_test2))
        weight2 = math.exp(dsc_test2) / (math.exp(dsc_test2) + math.exp(dsc_test))
        
        
        weighted_output = weight1 * test_output_en + weight2 * test_output2_en
        weighted_output = F.softmax(weighted_output, dim=1)
        weighted_resized_output = resize_output_to_cropped_size(weighted_output, cropped_shape, is_label=False)
        weighted_restored_output = pad_to_original_size(weighted_resized_output, original_shape, bbox)
        
        weighted_restored_output[weighted_restored_output>=0.5] = 1
        weighted_restored_output[weighted_restored_output<0.5] = 0
        
        dsc_test3 = compute_dice_score(weighted_restored_output[:,1,:,:], test_label[:,1,:,:])
        dsc_test3 = dsc_test3.mean()
        

        
        
    
        # HD_tst = HD(test_output, test_label)
        # SD_tst = SD(test_output, test_label)
        

        # print("[Test set name: %s, DSC: %.4f, , HD: %.4f, SD: %.4f, Precision: %.4f, Recall: %.4f, Specificity: %.5f]"%(test_name[0], dsc_test3.item(),
        #                                                                                                                 HD_tst.item(), SD_tst.item(), precision, recall, specificity))
        print("[Test set name: %s, DSC: %.4f]"%(test_name[0], dsc_test3.item()))
        
        
        
        test_names.append(test_name[0])
        dsc_scores.append(dsc_test3.item())
        # HD_arr.append(HD_tst.item())
        # SD_arr.append(SD_tst.item())
        
        # for q in range(19):
        fig = plt.figure(figsize=(20,6))
        plt.subplot(1,3,1)
        plt.title('input')
        plt.imshow(ori_img[0,0,:,:].cpu().detach().numpy(),cmap='gray')
        plt.subplot(1,3,2)
        plt.title('output')
        plt.imshow(weighted_restored_output[0,1,:].cpu().detach().numpy(),cmap='gray')
        plt.subplot(1,3,3)
        plt.title('label')
        plt.imshow(test_label[0,1,:].cpu().detach().numpy(), cmap='gray')
        plt.savefig(args.output_dir+'test/'+test_name[0]+'.png')
        plt.close()
        
        plt.imsave(args.output_dir+'test/'+test_name[0]+'_1_in_img.png',ori_img[0,0,:,:].cpu().detach().numpy(), cmap='gray')
        plt.imsave(args.output_dir+'test/'+test_name[0]+'_2_out_img.png',weighted_restored_output[0,1,:].cpu().detach().numpy(),cmap='gray')
        plt.imsave(args.output_dir+'test/'+test_name[0]+'_3_label_img.png', test_label[0,1,:].cpu().detach().numpy(), cmap='gray')
        
        
        
        
        
    # df = pd.DataFrame({'Test Set Name': test_names, 'DSC': dsc_scores, 'HD': HD_arr, 'SD': SD_arr, 'Precision': precision_arr, 'Recall': recall_arr, 'Specificity': specificity_arr})
    df = pd.DataFrame({'Test Set Name': test_names, 'DSC': dsc_scores})
    df.to_excel(args.output_dir+'test/'+'test_results.xlsx', index=False)