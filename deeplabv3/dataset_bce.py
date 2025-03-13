from glob import glob
import numpy as np


import torch
from torch.utils.data import Dataset

from utils import weight_map


def normalize_images(t1):
    
    t1 = (t1 - t1.mean()) / max(t1.std(), 1e-8)
    
    return t1


class SpineDataset(Dataset):
    def __init__(self, args, is_valid=False, is_test=False, transform=None):
        
        self.args = args   
        self.is_valid = is_valid
        self.is_test = is_test
        self.transform = transform
        
        if self.is_valid:
            self.img_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal/MR/*.npy'))
            self.label_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal/Mask/*.npy'))
        
        elif self.is_test:
            self.img_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test2_ideal/MR/*.npy'))
            self.label_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test2_ideal/Mask/*.npy'))
            
        else:
            self.img_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal/MR/*.npy'))
            self.label_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal/Mask/*.npy'))
            
    def __len__(self):
        
        return len(self.img_dir)
    
    def __getitem__(self, idx):
        
        img = np.flipud(np.load(self.img_dir[idx]).astype(np.float32))
        label = np.flipud(np.load(self.label_dir[idx]).astype(np.uint8))
        
        img = normalize_images(img)
        
        
        label[label>10] = 0
        label[label!=0] = 1 
        
        w_map = weight_map(label)
        
        
        img = np.expand_dims(img, axis=0)
        label = np.expand_dims(label, axis=0)    # single class로만 학습 시에 이용
        w_map = np.expand_dims(w_map, axis=0)
        
        
        data_dict = {'img': img, 'label': label, 'w_map': w_map, 'file_name': self.img_dir[idx].split('/')[-1][:-4]}

        if self.transform is not None:
            data_dict = self.transform(data_dict)
    
        
        return data_dict
        
        
        
        
        
        