"""Python module which provides functions to calculate echo intensity of US images."""

import cv2 
import numpy as np


def calculate_echo_int(imagepath: str, mask):
    """Calculates echo intensity (mean grey value) of pixels within 
       given region. 

    Arguments:
        Path to image,
        predicted mask of the respective image as binary np.array.

    Returns: 
        Echo intensity value of predicted muscle area.

    Example:
        >>>calculate_echo_int(C:/Desktop/Test, C:/Desktop/Test/Img1.tif,
                          pred_apo_t)
        65.728
    """
    img = cv2.imread(imagepath, 0)
    mask = mask.astype(np.uint8)

    # Find contours in binary mask image
    conts = cv2.findContours(mask, cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_NONE)

    # Grab countours
    conts = conts[0][0]

    cv2.fillPoly(mask, conts, (255))
    res = cv2.bitwise_and(img, img, mask=mask)  # Crop mask region from img
    rect = cv2.boundingRect(conts)  # Returns (x,y,w,h) of the bounding rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    pixel = cropped.ravel()  # 1D Array of pixels in cropped

    # Count pixel with value > 0
    vals=[]
    for pix in pixel:
        if pix > 1:
            vals.append(pix)

    # Calculate echo intensity
    echo_int = round(np.mean(vals), 3)
    return echo_int
