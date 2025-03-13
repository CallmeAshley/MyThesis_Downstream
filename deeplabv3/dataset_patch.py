from glob import glob
import numpy as np
import cv2
from utils import weight_map
import torch
from torch.utils.data import Dataset
import random


def normalize_images(t1):    # slice 별 z-score
    
    t1 = (t1 - t1.mean()) / max(t1.std(), 1e-8)
    
    return t1

def min_max_norm(x):    # slice 별 min max
    x = (x - x.min()) / (x.max() - x.min())
    
    return x


class SpineDataset(Dataset):
    def __init__(self, args, mode='plain', is_valid=False, is_test=False, transform=None):
        
        self.args = args   
        self.mode = mode
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
        if self.mode == 'plain':
            img = np.flipud(np.load(self.img_dir[idx]).astype(np.float32))
            label = np.flipud(np.load(self.label_dir[idx]).astype(np.uint8))
            
            img = normalize_images(img)
            
            
            label[label>10] = 0
            label[label!=0] = 1
            
            w_map = weight_map(label)
            
            img = np.expand_dims(img, axis=0)
            w_map = np.expand_dims(w_map, axis=0)

            label = np.eye(2)[label].astype('uint8')     # 2-class로 만들고 싶으면 사용
            label = np.transpose(label, (2, 0, 1))
            
            
            data_dict = {'img': img, 'label': label, 'w_map': w_map, 'file_name': self.img_dir[idx].split('/')[-1][:-4]}


            
            if self.transform is not None:
                data_dict = self.transform(data_dict)
            
            
            return data_dict
        
        else:
            img = np.flipud(np.load(self.img_dir[idx]).astype(np.float32))
            img = normalize_images(img)
            img = cv2.resize(img, (225, 225), cv2.INTER_LINEAR)
            img = np.expand_dims(img, axis=0)
        
            seg = np.flipud(np.load(self.label_dir[idx]).astype(np.uint8))
            seg[seg>10] = 0
            seg[seg!=0] = 1
            seg = cv2.resize(seg, (225, 225), cv2.INTER_NEAREST_EXACT)
            seg = np.eye(2)[seg].astype('uint8')
            seg = np.transpose(seg, (2, 0, 1))
            
            
        
        
            segclips = []
            oriclips = []
        
            for i in range(3):
                for j in range(3):
                    ori_clip = img[:, i * 75: (i + 1) * 75, j * 75: (j + 1) * 75]   # seg용 img
                    seg_clip = seg[:, i * 75: (i + 1) * 75, j * 75: (j + 1) * 75]   # seg용 label
                    segclips.append(seg_clip)
                    oriclips.append(ori_clip)
                    
            segclips = np.array(segclips)
            oriclips = np.array(oriclips)
            
            
            data_dict = {'img': oriclips, 'label': segclips, 'file_name': self.img_dir[idx].split('/')[-1][:-4]}


            
            # if self.transform is not None:
            #     data_dict = self.transform(data_dict)
            
            return data_dict
        
        
        
        
class FoldDataset(Dataset):

    def __init__(self, permutations, n_classes, in_channels=1, is_valid=False):
        super(FoldDataset, self).__init__()
        
        self.is_valid = is_valid
        self.n_classes = n_classes

        if self.is_valid:
            self.img_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal/MR/*.npy'))
            self.seg_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal/Mask/*.npy'))
        else:
            self.img_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal/MR/*.npy'))
            self.seg_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal/Mask/*.npy'))
        
        
        self.in_channels = in_channels
        self.permutations = permutations # list

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx): 
        
        img = np.flipud(np.load(self.img_dir[idx]).astype(np.float32))
        img = normalize_images(img)
        img = cv2.resize(img, (225, 225), cv2.INTER_LINEAR)  # INTER_CUBIC
        img = np.expand_dims(img, axis=0)
        
        seg = np.flipud(np.load(self.seg_dir[idx]).astype(np.uint8))
        seg[seg>10] = 0
        seg[seg!=0] = 1
        seg = cv2.resize(seg, (225, 225), cv2.INTER_NEAREST_EXACT)
        seg = np.eye(2)[seg].astype('uint8')
        seg = np.transpose(seg, (2, 0, 1))
    
        
        label = random.randint(0, self.n_classes-1)     # 그 label이 아니라, permutation에서 무작위로 뽑아오는 역할을 하는 label.
        
        
        imgclips = []
        segclips = []
        oriclips = []
        
        for i in range(3):
            for j in range(3):
                img_clip = img[:, i * 75: (i + 1) * 75, j * 75: (j + 1) * 75]   # jigsaw용 img
                ori_clip = img[:, i * 75: (i + 1) * 75, j * 75: (j + 1) * 75]   # seg용 img
                seg_clip = seg[:, i * 75: (i + 1) * 75, j * 75: (j + 1) * 75]   # seg용 label
                randomx = random.randint(0, 10)
                randomy = random.randint(0, 10)
                img_clip = img_clip[:, randomx: randomx+64, randomy:randomy+64]
                # seg_clip = seg_clip[:, randomx: randomx+64, randomy:randomy+64]       # 이게 gap 주는 코드 같음.... 
 
                imgclips.append(img_clip)
                segclips.append(seg_clip)
                oriclips.append(ori_clip)

        imgclips = [imgclips[item] for item in self.permutations[label]]
        segclips = [segclips[item] for item in self.permutations[label]]
        oriclips = [oriclips[item] for item in self.permutations[label]]
        imgclips = np.array(imgclips)
        segclips = np.array(segclips)
        oriclips = np.array(oriclips)



        return img, torch.from_numpy(imgclips), torch.tensor(label), torch.from_numpy(segclips), torch.from_numpy(oriclips)      # imgclips: (9, 1, 64, 64) --> patch size가 imgclips
                                                                                                    # segclips: (9, 2, 64, 64)