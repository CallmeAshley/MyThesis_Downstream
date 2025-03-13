from glob import glob
import numpy as np
import random
from utils import weight_map
import torch
from torch.utils.data import Dataset


def normalize_images(t1):    # slice 별 z-score
    
    t1 = (t1 - t1.mean()) / max(t1.std(), 1e-8)
    
    return t1


# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, crop_to_bbox, bounding_box_to_slice


def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    from scipy.ndimage import binary_fill_holes
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        this_mask = data[c] != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)   # nonzero인 부분 bbox

    slicer = bounding_box_to_slice(bbox)
    data = data[tuple([slice(None), *slicer])]

    if seg is not None:
        seg = seg[:, :, bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]


    return data, seg, bbox


class SpineDataset(Dataset):
    def __init__(self, args, is_valid=False, is_test=False, transform=None):
        
        self.args = args   
        self.is_valid = is_valid
        self.is_test = is_test
        self.transform = transform
        
        if self.is_valid:
            self.img_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal/MR/*.npy'))
            self.label_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal/Mask/*.npy'))
            
            # n = self.args.num_val_data  # 추출하고자 하는 샘플 수
            # indices = random.sample(range(len(self.img_dir)), n)
            # self.img_dir = [self.img_dir[i] for i in indices]
            # self.label_dir = [self.label_dir[i] for i in indices]
        
        elif self.is_test:
            self.img_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test2_ideal/MR/*.npy'))
            self.label_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test2_ideal/Mask/*.npy'))
            
        else:
            self.img_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal/MR/*.npy'))
            self.label_dir = sorted(glob('/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/train_ideal/Mask/*.npy'))
            
            # n = self.args.num_train_data  # 추출하고자 하는 샘플 수
            # indices = random.sample(range(len(self.img_dir)), n)
            # self.img_dir = [self.img_dir[i] for i in indices]
            # self.label_dir = [self.label_dir[i] for i in indices]
            
    def __len__(self):
        
        return len(self.img_dir)
    
    def __getitem__(self, idx):
        if self.args.mode == 'train':
            img = np.flipud(np.load(self.img_dir[idx]).astype(np.float32))
            label = np.flipud(np.load(self.label_dir[idx]).astype(np.uint8))
            
            # original_shape = img.shape
            
            label[label>10] = 0
            label[label!=0] = 1
            
            
            img = np.expand_dims(img, axis=0)
            
            # ori_img = img
            # ori_img = normalize_images(ori_img)
            
            img = np.expand_dims(img, axis=0)

            
            label = np.eye(2)[label].astype('uint8')     # 2-class로 만들고 싶으면 사용
            label = np.transpose(label, (2, 0, 1))
            label = np.expand_dims(label, axis=0)
            
            
            img, label, bbox=crop_to_nonzero(data=img, seg=label)
            
            img = np.squeeze(img, axis=0)
            label = np.squeeze(label, axis=0)
            
            # cropped_shape = img.shape
            
            img = normalize_images(img)
            
            
            
            data_dict = {'img': img, 'label': label, 'file_name': self.img_dir[idx].split('/')[-1][:-4]}


            
            if self.transform is not None:
                data_dict = self.transform(data_dict)
            
            
            return data_dict
            # return data_dict, original_shape, cropped_shape, bbox, ori_img
        
        
        elif self.args.mode == 'infer':
            img = np.flipud(np.load(self.img_dir[idx]).astype(np.float32))
            label = np.flipud(np.load(self.label_dir[idx]).astype(np.uint8))
            
            original_shape = img.shape
            
            label[label>10] = 0
            label[label!=0] = 1
            
            
            img = np.expand_dims(img, axis=0)
            
            ori_img = img
            ori_img = normalize_images(ori_img)
            
            img = np.expand_dims(img, axis=0)

            
            label = np.eye(2)[label].astype('uint8')     # 2-class로 만들고 싶으면 사용
            label = np.transpose(label, (2, 0, 1))
            label = np.expand_dims(label, axis=0)
            
            
            img, label, bbox=crop_to_nonzero(data=img, seg=label)
            
            img = np.squeeze(img, axis=0)
            label = np.squeeze(label, axis=0)
            
            cropped_shape = img.shape
            
            img = normalize_images(img)
            
            
            
            data_dict = {'img': img, 'label': label, 'file_name': self.img_dir[idx].split('/')[-1][:-4]}


            
            if self.transform is not None:
                data_dict = self.transform(data_dict)
            
            
            # return data_dict
            return data_dict, original_shape, cropped_shape, bbox, ori_img
        
        
        
        