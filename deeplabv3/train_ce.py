################################################  
# torch.load_state_dict('sdfsdf',strict=False)  후에 jigsaw pretrain 불러올 때 쓸 코드.


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse

import os
import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import matplotlib.pyplot as plt

from model_ce import UNet
from dataset_ce import SpineDataset

from monai.transforms import *
from monai import transforms
from monai.losses import DiceLoss
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric

import torch.backends.cudnn as cudnn
from utils import compute_dice_score

# from swin_unet import SwinUnet
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.resnet import ResNet50_Weights

import torch.nn.functional as F

# import segmentation_models_pytorch as smp


def parse_args():
    parser = argparse.ArgumentParser(description='Spine Segmentation by Soohyun Lee in MAI-LAB')

    parser.add_argument('--resume', type=str, default=None, help='load pretrained model path')
    parser.add_argument('--pretrain', type=str, default='/mai_nas/LSH/SparK/exp/basic/resnet50_epoch1000.pth', help='load pretrained model path')
     
    parser.add_argument('--output_dir', type=str, default='/mai_nas/LSH/MRSpineSeg_Challenge_SMU/unet/try/',help='Directory name to save the model, log, config')

    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32, help='The size of batch')
    parser.add_argument('--val_batch_size', type=int, default=80, help='The size of validation batch')
    parser.add_argument('--training_size', default=(512,512))    
    
    # parser.add_argument('--num_train_data', type=int, default=100)
    # parser.add_argument('--num_val_data', type=int, default=11)
     
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--freeze', type=str, default='true', help='true or false')
    
    parser.add_argument('--loss_type', default='dice', type=str, help='ce, dfice')    
    parser.add_argument('--aug_type', default='contrast', type=str, help='plain, rot, contrast')        
    parser.add_argument('--lr_sch', default=True, type=bool, help='쓸지 말지')    


    parser.add_argument('--gpus',default='2,3' ,type=str, help='gpu num')
    parser.add_argument('--workers', default=8, type=int, help='number of workers for dataloader')
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
        os.makedirs(args.output_dir+'/model')
        os.makedirs(args.output_dir+'/test/matfile')
        os.makedirs(args.output_dir+'/val')
        os.makedirs(args.output_dir+'/log')
        
        
    
writer = SummaryWriter(log_dir=(args.output_dir+'/log'))   


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model = UNet()

# model = SwinUnet()
model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=2)
model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# model = deeplabv3_resnet101(weights=None, num_classes=2)
# model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


if (args.resume is None) and (args.pretrain is None):
    model = nn.DataParallel(model)



if (args.resume is not None) or (args.pretrain):
    if args.resume is not None:
        state_dict = torch.load(os.path.join(args.resume))
        model.load_state_dict(state_dict, strict=False)
    else:
        state_dict = torch.load(args.pretrain)
    
        # 새로운 state_dict를 생성합니다
        new_state_dict = {}
        
        
        # # 'module' 키워드를 'backbone'으로 변경합니다
        # for key, value in state_dict.items():
        #     new_key = key.replace('module', 'backbone')  # 'module'을 'backbone'으로 교체
        #     new_state_dict[new_key] = value
        
        # elif args.pretrain == '/mai_nas/LSH/SparK/exp/basic/resnet50_epoch1300.pth':
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'backbone.'+k
            new_state_dict[name] = v

        # 변경된 state_dict를 모델에 로드합니다
        model.load_state_dict(new_state_dict, strict=False)
        
        
        
    
    if args.freeze == 'true':
    
        for param in model.backbone.parameters():
            param.requires_grad = False
        
        for param in model.classifier.parameters():
            param.requires_grad = True
            
        
    model = nn.DataParallel(model)

    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k if k.startswith('module.') else 'module.'+k
    #     new_state_dict[name] = v

    

    
               
if args.freeze == 'false':        
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))    
elif args.freeze == 'true':
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)




if args.loss_type == 'ce':
    loss_ce = nn.CrossEntropyLoss().to(device)
elif args.loss_type == 'dice':
    loss_dice = DiceLoss(include_background=True).to(device)




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
                                        RandGaussianNoised(keys=['img'], mean=0, std=0.1, prob=1),
                                        RandGaussianSmoothd(keys=['img'], sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), prob=1),
                                        RandScaleIntensityd(keys=["img"], factors=0.25, prob=1),
                                        RandAdjustContrastd(keys=['img'], gamma=(0.75, 1.25), prob=1),
                                        ToTensord(keys=["img",'label'])
                                        ])


        
transform_valandtest = transforms.Compose([   
                                        Resized(keys=["img"], spatial_size=args.training_size, mode="bilinear"),
                                        Resized(keys=["label"], spatial_size=args.training_size, mode="nearest"),
                                        
                                        ToTensord(keys=["img",'label'])
                                        ])


transform_train.set_random_state(seed=args.random_seed)
transform_valandtest.set_random_state(seed=args.random_seed)


dataloader_train = DataLoader(SpineDataset(args, is_valid=False, is_test=False, transform=transform_train), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
dataloader_valid = DataLoader(SpineDataset(args, is_valid=True, is_test=False, transform=transform_valandtest), batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
dataloader_test = DataLoader(SpineDataset(args, is_valid=False, is_test=True, transform=transform_valandtest), batch_size=1, shuffle=False, pin_memory=True, num_workers=args.workers)

before = 0
after = 0
DSC_test_final = 0
a = 0
b = 0
c = 1
thre0 = 0
thre1 = 0

# HD = HausdorffDistanceMetric(include_background=True, percentile=95)
# SD = SurfaceDistanceMetric(include_background=True)

# if args.resume is None:
for epoch in range(args.epoch):
    print("*******************%s+%s train*******************"%(args.loss_type, args.aug_type))
    
    loss_train_per_epoch = 0
    loss_val_per_epoch = 0
    DSC_train_per_epoch = 0
    DSC_val_per_epoch = 0   
    HD_train_per_epoch = 0
    SD_train_per_epoch = 0
    HD_val_per_epoch = 0
    SD_val_per_epoch = 0
    
    

    model.to(device)
    model.train()
    
    
    for i, train_data in enumerate(dataloader_train):
        
        dsc_train = 0 
        
        train_input = train_data['img'].to(device)
        train_label = train_data['label'].to(device)
        train_name = train_data['file_name']

        
        optimizer.zero_grad()

        train_output = model(train_input)
        train_output = train_output['out']
        
        if args.loss_type == 'ce':
            loss_train = loss_ce(train_output, train_label)
        elif args.loss_type == 'dice':
            train_output = F.softmax(train_output, dim=1)
            loss_train = loss_dice(train_output, train_label)
        

        loss_train.backward()

        optimizer.step()
        
        
        loss_train_per_epoch += loss_train


        
        
        train_output[train_output>=0.5] = 1
        train_output[train_output<0.5] = 0  
        

        dsc_train += compute_dice_score(train_output[:,1,:,:], train_label[:,1,:,:])
            
        dsc_train_mean = dsc_train.mean()
        DSC_train_per_epoch += dsc_train_mean
        
        
        # HD_train = HD(train_output[:,1,:,:], train_label[:,1,:,:])
        # SD_train = SD(train_output[:,1,:,:], train_label[:,1,:,:])
        
        # HD_train_per_epoch += HD_train
        # SD_train_per_epoch += SD_train
        
        
        
        print("[Epoch %3d/%3d, Batch %4d/%4d, Train loss: %.4f]"
                % (epoch+1, args.epoch, i+1, len(dataloader_train), loss_train.item()))
        


    
    # print("[Epoch %3d/%3d, Train loss: %.4f,  Average DSC: %.4f, Average HD: %.4f, Average SD: %.4f]"
    #             % (epoch+1, args.epoch, loss_train_per_epoch.item()/len(dataloader_train), DSC_train_per_epoch.item()/len(dataloader_train), 
    #                HD_train_per_epoch.item()/len(dataloader_train), SD_train_per_epoch.item()/len(dataloader_train)))
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
            dsc_val=0
            valid_input = valid_data['img'].to(device)
            valid_label = valid_data['label'].to(device)
            valid_name = valid_data['file_name']

            

            valid_output = model(valid_input)
            valid_output = valid_output['out']
        
            if args.loss_type == 'ce':
                loss_val = loss_ce(valid_output, valid_label)
            elif args.loss_type == 'dice':
                valid_output = F.softmax(valid_output, dim=1)
                loss_val = loss_dice(valid_output, valid_label)
            
            
            loss_val_per_epoch += loss_val
            
            valid_output = valid_output[:,1,:,:]
            valid_label = valid_label[:,1,:,:]

            
            valid_output[valid_output>=0.5] = 1
            valid_output[valid_output<0.5] = 0      
            

            dsc_val += compute_dice_score(valid_output, valid_label)
            dsc_val_mean = dsc_val.mean()
            DSC_val_per_epoch += dsc_val_mean
            
            valid_output_flatten = valid_output.flatten()
            valid_label_flatten = valid_label.flatten()
            
            cm = confusion_matrix(valid_label_flatten.cpu().detach().numpy(), valid_output_flatten.cpu().detach().numpy())
            
            
            tn, fp, fn, tp = cm.ravel() 
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            specificity = tn/(tn+fp)
            
            # HD_val = HD(valid_output, valid_label)
            # SD_val = SD(valid_output, valid_label)
            
            # HD_val_per_epoch += HD_val
            # SD_val_per_epoch += SD_val


        # print("[Epoch %3d/%3d, Valid loss: %.4f,  Average DSC: %.4f, Average HD: %.4f, Average SD: %.4f, Precision: %.10f, Recall: %.10f, Specificity: %.5f]"
        #         % (epoch+1, args.epoch, loss_val_per_epoch.item()/len(dataloader_valid), DSC_val_per_epoch.item()/len(dataloader_valid), 
        #            HD_val_per_epoch.item()/len(dataloader_valid), SD_val_per_epoch.item()/len(dataloader_valid), precision, recall, specificity))
        
        print("[Epoch %3d/%3d, Valid loss: %.4f,  Average DSC: %.4f, Precision: %.10f, Recall: %.10f, Specificity: %.5f]"
                % (epoch+1, args.epoch, loss_val_per_epoch.item()/len(dataloader_valid), DSC_val_per_epoch.item()/len(dataloader_valid), 
                   precision, recall, specificity))
        
        
        after = DSC_val_per_epoch.item()/len(dataloader_valid)

        
        if before < after:                                                                                                                                                                           

            torch.save(model.state_dict(), args.output_dir+'model/Best_DSC.pth') 
            
            before = after

                
            for k in range(args.val_batch_size):

                    fig = plt.figure(figsize=(20,6))
                    plt.subplot(1,3,1)
                    plt.title('input')
                    plt.imshow(valid_input[k,0,:].cpu().detach().numpy(),cmap='gray')
                    plt.subplot(1,3,2)
                    plt.title('output')
                    plt.imshow(valid_output[k,:,:].detach().cpu().numpy(),cmap='gray')
                    plt.subplot(1,3,3)
                    plt.title('label')
                    plt.imshow(valid_label[k,:,:].detach().cpu().numpy(), cmap='gray')

                    plt.savefig(args.output_dir+'val/'+str(k)+'_'+valid_name[0]+'.png')
                    plt.close()
        
        
        writer.add_scalar("Loss_valid", loss_val_per_epoch.item()/len(dataloader_valid), epoch+1)
        writer.add_scalar("DSC_valid", DSC_val_per_epoch.item()/len(dataloader_valid), epoch+1)
        
writer.close()      
            
      
        
        
# else:
#     test_names = []
#     dsc_scores = []
#     fnr = []
#     fpr = []
    
#     model.to(device)

#     for t, test_data in enumerate(dataloader_test):
        
#         print("")
        
#         test_input = test_data['img'].to(device)
#         test_label = test_data['label'].to(device)
#         test_name = test_data['file_name']


#         test_output = model(test_input)
        
        

#         sio.savemat(args.output_dir+'test/matfile/'+test_name[0]+'.mat',mdict={'output':test_output[0,1,:,:].detach().cpu().numpy()})

#         test_output = test_output[:,1,:,:]
#         test_label=test_label[:,1,:,:]
        
        
#         test_output[test_output>=0.5] = 1
#         test_output[test_output<0.5] = 0
        
#         test_output_flatten = test_output.flatten()
#         test_label_flatten = test_label.flatten()
                
#         cm = confusion_matrix(test_label_flatten.cpu().detach().numpy(), test_output_flatten.cpu().detach().numpy())   
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#         disp.plot()
#         plt.savefig(args.output_dir+'test/'+test_name[0]+'_confusion_matrix.png')
#         tn, fp, fn, tp = cm.ravel() 
        
#         FNR = fn / (fn + tp)
#         FPR = fp / (fp + tn)
        
#         fnr.append(FNR)
#         fpr.append(FPR)
        

#         dsc_test = compute_dice_score(test_output, test_label)
#         dsc_test = dsc_test.mean()

#         print("[Test set name: %s, DSC: %.4f, FNR: %.4f, FPR: %.4f]"%(test_name[0], dsc_test.item(), FNR, FPR))
        
        
#         test_names.append(test_name[0])
#         dsc_scores.append(dsc_test.item())

        
#         # for q in range(19):
#         fig = plt.figure(figsize=(20,6))
#         plt.subplot(1,3,1)
#         plt.title('input')
#         plt.imshow(test_input[0,0,:,:].cpu().detach().numpy(),cmap='gray')
#         plt.subplot(1,3,2)
#         plt.title('output')
#         plt.imshow(test_output[0,:].cpu().detach().numpy(),cmap='gray')
#         plt.subplot(1,3,3)
#         plt.title('label')
#         plt.imshow(test_label[0,:].cpu().detach().numpy(), cmap='gray')
#         plt.savefig(args.output_dir+'test/'+test_name[0]+'.png')
        
        
#     df = pd.DataFrame({'Test Set Name': test_names, 'DSC': dsc_scores, 'FNR': fnr, 'FPR': fpr})
#     df.to_excel(args.output_dir+'test/'+'test_results.xlsx', index=False)
            
            
            
       
            
            
  
