import torch.nn as nn
import cv2
import numpy as np
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve

def compute_dice_score(pred, gt):
    intersection = (pred * gt).sum(dim=(1, 2))
    epsilon = 1e-6
    # intersection = (pred * gt).sum(1).sum(1)
    pred_area = pred.sum(dim=(1, 2))
    gt_area = gt.sum(dim=(1, 2))
    # pred_area = pred.sum(1).sum(1)
    # gt_area = gt.sum(1).sum(1)
    dice = (2.0 * intersection) / (pred_area + gt_area + epsilon)
    return dice



# def weightedCE_loss(pred, label, weight):
#     BCE_loss = nn.BCEWithLogitsLoss(reduction='none')
#     loss_=  BCE_loss(pred, label) * weight

#     return loss_.mean()


def weightedCE_loss(pred, label, weight):
    CE_loss = nn.CrossEntropyLoss(reduction='none')
    loss_=  CE_loss(pred, label) * weight

    return loss_.mean()



def weight_map(mask, w0 = 10, sigma = 10, background_class = 0):
    
    # Fix mask datatype (should be unsigned 8 bit)
    if mask.dtype != 'uint8': 
        mask = mask.astype('uint8')
    
    # Weight values to balance classs frequencies
    wc = _class_weights(mask)
    
    # Assign a different label to each connected region of the image
    _, regions = cv2.connectedComponents(mask)
    
    # Get total no. of connected regions in the image and sort them excluding background
    region_ids = sorted(np.unique(regions))
    region_ids = [region_id for region_id in region_ids if region_id != background_class]
        
    if len(region_ids) > 1: # More than one connected regions

        # Initialise distance matrix (dimensions: H x W x no.regions)
        distances = np.zeros((mask.shape[0], mask.shape[1], len(region_ids)))

        # For each region
        for i, region_id in enumerate(region_ids):

            # Mask all pixels belonging to a different region
            m = (regions != region_id).astype(np.uint8)# * 255
        
            # Compute Euclidean distance for all pixels belongind to a different region
            distances[:, :, i] = cv2.distanceTransform(m, distanceType = cv2.DIST_L2, maskSize = 0)

        # Sort distances w.r.t region for every pixel
        distances = np.sort(distances, axis = 2)

        # Grab distance to the border of nearest region
        d1, d2 = distances[:, :, 0], distances[:, :, 1]

        # Compute RHS of weight map and mask background pixels
        w = w0 * np.exp(-1 / (2 * sigma ** 2)  * (d1 + d2) ** 2) * (regions == background_class)

    else: # Only a single region present in the image
        w = np.zeros_like(mask)

    # Instantiate a matrix to hold class weights
    wc_x = np.zeros_like(mask)
    
    # Compute class weights for each pixel class (background, etc.)
    for pixel_class, weight in wc.items():
    
        wc_x[mask == pixel_class] = weight
    
    # Add them to the weight map
    w = w + wc_x
    
    return w

def _class_weights(mask):
    ''' Create a dictionary containing the classes in a mask,
        and their corresponding weights to balance their occurence
    '''
    wc = defaultdict()

    # Grab classes and their corresponding counts
    unique, counts = np.unique(mask, return_counts = True)

    # Convert counts to frequencies
    counts = counts / np.product(mask.shape)

    # Get max. counts
    max_count = max(counts)

    for val, count in zip(unique, counts):
        wc[val] = max_count / count
    
    return wc




def sensivity_specifity_cutoff(y_true, y_score):

    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr - fpr)
    return thresholds[idx]

    '''Find data-driven cut-off for classification
    
    y_true : array, shape = [n_samples]
        True binary labels.
        
    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned by
        “decision_function” on some classifiers).
        
    '''
    
    
    
    