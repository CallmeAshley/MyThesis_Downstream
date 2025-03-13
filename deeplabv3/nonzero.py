import numpy as np


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
        seg = seg[tuple([slice(None), *slicer])]

    # nonzero_mask = nonzero_mask[slicer][None]
    # if seg is not None:
    #     seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    # else:
    #     nonzero_mask = nonzero_mask.astype(np.int8)
    #     nonzero_mask[nonzero_mask == 0] = nonzero_label
    #     nonzero_mask[nonzero_mask > 0] = 0
    #     seg = nonzero_mask
    return data, seg, bbox



if __name__ == '__main__':
    img_path  = '/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal/MR/Case99_7.npy'
    img = np.load(img_path)
    seg_path = '/mai_nas/LSH/MRSpineSeg_Challenge_SMU/dataset/2D/test1_ideal/Mask/mask_case99_7.npy'
    seg=np.load(seg_path)
    img = np.expand_dims(img, axis=0)
    img = np.concatenate((img,img,img), axis=0)
    img = np.expand_dims(img, axis=0)  # (1, 3, 880, 880)
    seg[seg>10] = 0
    seg[seg!=0] = 1
    seg = np.eye(2)[seg].astype('uint8')
    seg = np.transpose(seg, (2, 0, 1))
    seg = np.expand_dims(seg, axis=0)    # (1, 2, 880, 880)

    img,seg,_=crop_to_nonzero(data=img, seg=seg)
    
    print(str(img.shape))