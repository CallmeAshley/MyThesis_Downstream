################################################  
# torch.load_state_dict('sdfsdf',strict=False)  후에 jigsaw pretrain 불러올 때 쓸 코드.



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

# from monai.losses.dice import DiceLoss
# from monai.metrics import DiceMetric
import pandas as pd
import openpyxl

from model_bce import UNet
from dataset_bce import SpineDataset

import scipy.io as sio

from monai.transforms import *
from monai import transforms
from monai.losses import DiceLoss

import torch.backends.cudnn as cudnn
from utils import compute_dice_score, weightedCE_loss, sensivity_specifity_cutoff


def parse_args():
    parser = argparse.ArgumentParser(description='Spine Segmentation by Soohyun Lee in MAI-LAB')

    parser.add_argument('--resume', type=str, default=None, help='load pretrained model path')
    parser.add_argument('--pretrain', type=str, default=None, help='load pretrained model path')
    
    parser.add_argument('--output_dir', type=str, default='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/unet_bce_only_rot/',help='Directory name to save the model, log, config')


    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
    parser.add_argument('--training_size', default=(512,512))    
    

    parser.add_argument('--loss_type', default='wce', type=str, help='dice, bce, wce, bce+dice, wce+dice')    
    parser.add_argument('--aug_type', default='rot', type=str, help='plain, rot, contrast')
    parser.add_argument('--lr_sch', default=True, type=bool, help='쓸지 말지')    


    parser.add_argument('--gpus',default='4,5,6,7' ,type=str, help='gpu num')
    parser.add_argument('--workers', default=32, type=int, help='number of workers for dataloader')
    parser.add_argument('--random_seed', default=1213, type=int, help='default 12345')
    
    
    args = parser.parse_args()
    
    return args
    
    
args = parse_args()

# cudnn.benchmark = False
# cudnn.deterministic = True
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
        
        
    
writer = SummaryWriter(log_dir=(args.output_dir+'log'))   


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = UNet()


if args.resume is None:
    model = nn.DataParallel(model)



if (args.resume is not None) or (args.pretrain):
    if args.resume:
        state_dict = torch.load(args.resume)
    else:
        state_dict = torch.load(args.pretrain)

    # 'module.' 접두어가 있는지 확인하고, 없으면 추가
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k  # 'module.' 접두어 추가
        new_state_dict[name] = v

    # 수정된 state_dict를 모델에 적용
    model = nn.DataParallel(model)
    model.load_state_dict(new_state_dict, strict=False)
    
               
        
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

if args.loss_type == 'bce':
    loss_bce = nn.BCEWithLogitsLoss().to(device)
    
elif args.loss_type == 'bce+dice':
    loss_bce = nn.BCEWithLogitsLoss().to(device)
    loss_dice = DiceLoss(include_background=True, sigmoid=True).to(device)
    
elif args.loss_type == 'wce+dice':
    loss_dice = DiceLoss(include_background=True, sigmoid=True).to(device)
    
elif args.loss_type == 'dice':
    loss_dice = DiceLoss(include_background=True, sigmoid=True).to(device)
    



if args.lr_sch:
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0.0001)


# dice_metric_train = DiceMetric(include_background=False, reduction="mean")
# dice_metric_val = DiceMetric(include_background=False, reduction="mean")
# dice_metric_test = DiceMetric(include_background=False, reduction="mean")


if args.aug_type == 'plain': 
    transform_train = transforms.Compose([   
                                        Resized(keys=["img"], spatial_size=args.training_size, mode="bilinear"),
                                        Resized(keys=["label"], spatial_size=args.training_size, mode="nearest"),
                                        Resized(keys=["w_map"], spatial_size=args.training_size, mode="nearest"),
                                        ToTensord(keys=["img",'label', "w_map"])
                                        ])
    
elif args.aug_type == 'rot':
    transform_train = transforms.Compose([   
                                        Resized(keys=["img"], spatial_size=args.training_size, mode="bilinear"),
                                        Resized(keys=["label"], spatial_size=args.training_size, mode="nearest"),
                                        Resized(keys=["w_map"], spatial_size=args.training_size, mode="nearest"),
                                        RandRotated(keys=["img", "label", "w_map"], range_x=(-30,30), prob=0.5, mode=["bilinear", "nearest", "nearest"], padding_mode="border"),
                                        ToTensord(keys=["img",'label', 'w_map'])
                                        ])
    
elif args.aug_type == 'contrast':
    
    transform_train = transforms.Compose([   
                                        Resized(keys=["img"], spatial_size=args.training_size, mode="bilinear"),
                                        Resized(keys=["label"], spatial_size=args.training_size, mode="nearest"),
                                        Resized(keys=["w_map"], spatial_size=args.training_size, mode="nearest"),
                                        RandGaussianNoised(keys=['img'], mean=0, std=0.1, prob=0.2),
                                        RandGaussianSmoothd(keys=['img'], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=0.2),
                                        RandScaleIntensityd(keys=["img"], factors=0.25, prob=0.2),
                                        RandAdjustContrastd(keys=['img'], gamma=(0.75, 1.25), prob=0.2),
                                        ToTensord(keys=["img",'label', "w_map"])
                                        ])


        
transform_valandtest = transforms.Compose([   
                                        Resized(keys=["img"], spatial_size=args.training_size, mode="bilinear"),
                                        Resized(keys=["label"], spatial_size=args.training_size, mode="nearest"),
                                        Resized(keys=["w_map"], spatial_size=args.training_size, mode="nearest"),
                                        ToTensord(keys=["img",'label',"w_map"])
                                        ])


transform_train.set_random_state(seed=args.random_seed)
transform_valandtest.set_random_state(seed=args.random_seed)


dataloader_train = DataLoader(SpineDataset(args, is_valid=False, is_test=False, transform=transform_train), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
dataloader_valid = DataLoader(SpineDataset(args, is_valid=True, is_test=False, transform=transform_valandtest), batch_size=80, shuffle=False, pin_memory=True, num_workers=args.workers)
dataloader_test = DataLoader(SpineDataset(args, is_valid=False, is_test=True, transform=transform_valandtest), batch_size=1, shuffle=False, pin_memory=True, num_workers=args.workers)

before = 0
after = 0
thre0 = 0
thre1 = 0
DSC_test_final = 0
a = 0
b = 0
c = 1

if args.resume is None:
    for epoch in range(args.epoch):
        
        loss_train_per_epoch = 0
        loss_val_per_epoch = 0
        DSC_train_per_epoch = 0
        DSC_val_per_epoch = 0    

        model.to(device)
        model.train()
        
        
        for i, train_data in enumerate(dataloader_train):
            
            train_input = train_data['img'].to(device)
            train_label = train_data['label'].to(device)
            train_weight = train_data['w_map'].to(device)
            train_name = train_data['file_name']
            
            optimizer.zero_grad()

            train_output = model(train_input)
            
            if args.loss_type == 'bce':
                loss_train = loss_bce(train_output, train_label)
            elif args.loss_type == 'bce+dice':
                loss_train = loss_dice(train_output, train_label) + loss_bce(train_output, train_label)
            elif args.loss_type == 'wce':
                loss_train = weightedCE_loss(train_output, train_label, train_weight)
            elif args.loss_type == 'wce+dice':
                loss_train = weightedCE_loss(train_output, train_label, train_weight) + loss_dice(train_output, train_label)
            elif args.loss_type == 'dice':
                loss_train = loss_dice(train_output, train_label)

            
            loss_train.backward()
            optimizer.step()
            
            
            loss_train_per_epoch += loss_train
            
            # softmax = nn.Softmax(dim=1)
            train_output = torch.sigmoid(train_output) 
            
            thre = sensivity_specifity_cutoff(train_label.detach().cpu().numpy().flatten(), train_output.detach().cpu().numpy().flatten())
            
            train_output[train_output>=thre] = 1
            train_output[train_output<thre] = 0
            
            
            dsc_train = compute_dice_score(train_output[:,0,:,:], train_label[:,0,:,:])
            dsc_train = dsc_train.mean()
            dsc_train = dsc_train.detach()
            DSC_train_per_epoch += dsc_train
            
            
            print("[Epoch %3d/%3d, Batch %4d/%4d, Train loss: %.4f, Best_thre: %.8f]"
                    % (epoch+1, args.epoch, i+1, len(dataloader_train), loss_train.item(), thre))
                


        
        print("[Epoch %3d/%3d, Train loss: %.4f,  Average DSC: %.4f]"
                    % (epoch+1, args.epoch, loss_train_per_epoch.item()/len(dataloader_train), DSC_train_per_epoch.item()/len(dataloader_train)))
        
        writer.add_scalar("Loss_train", loss_train_per_epoch.item()/len(dataloader_train), epoch+1)
        writer.add_scalar("DSC_train", DSC_train_per_epoch.item()/len(dataloader_train), epoch+1)
        
        lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning Rate', lr, epoch+1)
        

        scheduler.step()
        
        
        with torch.no_grad():
            model.eval()
            
            for v, valid_data in enumerate(dataloader_valid):
                
                valid_input = valid_data['img'].to(device)
                valid_label = valid_data['label'].to(device)
                valid_weight = valid_data['w_map'].to(device)
                valid_name = valid_data['file_name']
                

                valid_output = model(valid_input)
            
                if args.loss_type == 'bce':
                    loss_val = loss_bce(valid_output, valid_label)
                elif args.loss_type == 'bce+dice':
                    
                    loss_val = loss_dice(valid_output, valid_label) + loss_bce(valid_output, valid_label)
                    
                elif args.loss_type == 'wce':
                    loss_val = weightedCE_loss(valid_output, valid_label, valid_weight)
                elif args.loss_type == 'wce+dice':
                    
                    loss_val = weightedCE_loss(valid_output, valid_label, valid_weight) + loss_dice(valid_output, valid_label)
                    
                elif args.loss_type == 'dice':
                    
                    loss_val = loss_dice(valid_output, valid_label)
                    
                
                
                loss_val_per_epoch += loss_val
                
                # softmax = nn.Softmax(dim=1)
                # valid_output = softmax(valid_output) 
                
                valid_output = torch.sigmoid(valid_output) 
                
                thre_val = sensivity_specifity_cutoff(valid_label.detach().cpu().numpy().flatten(), valid_output.detach().cpu().numpy().flatten())    #flatten 방식과 순서가 같은지 확인
                
                valid_output[valid_output>=thre_val] = 1
                valid_output[valid_output<thre_val] = 0
                
                dsc_val = compute_dice_score(valid_output[:,0,:,:], valid_label[:,0,:,:])
                dsc_val = dsc_val.mean()
                dsc_val = dsc_val.detach()
                DSC_val_per_epoch += dsc_val
            
            print("[Epoch %3d/%3d, Valid loss: %.4f,  Average DSC: %.4f]"
                    % (epoch+1, args.epoch, loss_val_per_epoch.item()/len(dataloader_valid), DSC_val_per_epoch.item()/len(dataloader_valid)))
            
            
            after = DSC_val_per_epoch.item()/len(dataloader_valid)
            thre1 = thre_val
            
            
            if before < after:
                
                torch.save(model.module.state_dict(), args.output_dir+'model/Best_DSC.pth') 
                
                before = after
                thre0 = thre1
                with open(args.output_dir+"model/Best_thre.txt", "w") as file:
                    file.write(str(thre0))
                    
                for k in range(80):
                    fig = plt.figure(figsize=(20,6))
                    plt.subplot(1,3,1)
                    plt.title('input')
                    plt.imshow(valid_input[k,0,:,:].cpu().detach().numpy(),cmap='gray')
                    plt.subplot(1,3,2)
                    plt.title('output')
                    plt.imshow(valid_output[k,0,:,:].cpu().detach().numpy(),cmap='gray')
                    plt.subplot(1,3,3)
                    plt.title('label')
                    plt.imshow(valid_label[k,0,:,:].cpu().detach().numpy(), cmap='gray')
                    plt.savefig(args.output_dir+'val/'+str(k)+'_'+valid_name[0]+'.png')
                    plt.close()
                    
                    
            
            writer.add_scalar("Loss_valid", loss_val_per_epoch.item()/len(dataloader_valid), epoch+1)
            writer.add_scalar("DSC_valid", DSC_val_per_epoch.item()/len(dataloader_valid), epoch+1)
            
    writer.close()      
            
      
        
        
        

else:
    test_names = []
    dsc_scores = []
    
    model.to(device)
    
    for t, test_data in enumerate(dataloader_test):
        test_input = test_data['img'].to(device)
        test_label = test_data['label'].to(device)
        test_name = test_data['file_name']


        test_output = model(test_input)

        sio.savemat(args.output_dir+'test/matfile/'+test_name[0]+'.mat',mdict={'output':test_output[0,0,:,:].detach().cpu().numpy()})

        # softmax = nn.Softmax(dim=1)
        # test_output = softmax(test_output) 
        test_output = torch.sigmoid(test_output) 
        
        with open(args.output_dir+"model/Best_thre.txt", "r") as file:
            content = file.read()
            
        best_thre = float(content)
        
        test_output[test_output>=best_thre] = 1
        test_output[test_output<best_thre] = 0


        dsc_test = compute_dice_score(test_output[:,0,:,:], test_label[:,0,:,:])
        dsc_test = dsc_test.mean()
        dsc_test = dsc_test.detach()

        print("[Test set name: %s, DSC: %.4f]"
            % (test_name[0], dsc_test.item()))
        
        
        test_names.append(test_name[0])
        dsc_scores.append(dsc_test.item())

        

        fig = plt.figure(figsize=(20,6))
        plt.subplot(1,3,1)
        plt.title('input')
        plt.imshow(test_input[0,0,:,:].cpu().detach().numpy(),cmap='gray')
        plt.subplot(1,3,2)
        plt.title('output')
        plt.imshow(test_output[0,0,:,:].cpu().detach().numpy(),cmap='gray')
        plt.subplot(1,3,3)
        plt.title('label')
        plt.imshow(test_label[0,0,:,:].cpu().detach().numpy(), cmap='gray')
        plt.savefig(args.output_dir+'test/'+test_name[0]+'.png')
        plt.close()
        
        
        
    df = pd.DataFrame({'Test Set Name': test_names, 'DSC': dsc_scores})
    df.to_excel(args.output_dir+'test/'+'test_results.xlsx', index=False)
            
            
            
       
            
            
  