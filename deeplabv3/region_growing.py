import cv2
import numpy as np



def region_growing(binary_img):
    """
    Extract the largest connected component from a binary image.
    """
    num_labels, labels = cv2.connectedComponents(binary_img)
    unique, counts = np.unique(labels, return_counts=True)
    largest_label = unique[np.argmax(counts[1:]) + 1]
    largest_component = np.zeros_like(binary_img)
    largest_component[labels == largest_label] = 255
    return largest_component

def process_images(image_path):
    medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    grown_region = region_growing(binary)

    closed = cv2.morphologyEx(grown_region, cv2.MORPH_CLOSE, medium_kernel)
    closed[closed>0] = 1
    return closed





