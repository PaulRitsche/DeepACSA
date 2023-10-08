"""
Description
-----------
This module contains a function to automatically calculate the echo intensity
(mean grayscale value) of the predicted mask. The correctness of the calculated
echo intensity depends on the correctess of the predicton.

Functions scope
---------------
calculate_echo_int
    Function to calculate the echo intensity of muscle area.
"""
import cv2
import numpy as np


def calculate_echo_int(img_copy: np.ndarray, mask: np.ndarray):
    """
    Function to calculatae the echo intensity (mean grey value) of pixels within
    given region.

    The function calculates the echo intensity based on the previously segmented/
    predicted mask. The mask is overlayd and the for the section of the predicted
    muscle area, the echo intensity is calculated on the copy of the original
    image.

    Parameters
    ----------
    img_copy : np.ndarray
        Copy of original US image input image.
    mask : np.ndarray
        Predicted ACSA of the respective image as binary np.array.

    Returns
    -------
    echo_int : float
        Float variable containint the mean grey scale value of
        the predicted muscle area.

    Notes
    -----
    - If no contours are found in the provided mask, the function returns None.
    - The echo intensity is calculated as the mean grey value of the pixels within the predicted muscle area.
    - The function uses the OpenCV library for contour finding and image manipulation.

    Example
    -------
    >>> calculate_echo_int(C:/Desktop/Test, C:/Desktop/Test/Img1.tif, pred_apo_t)
    65.728
    """
    img = img_copy
    img = img.astype(np.uint8)
    mask = mask.astype(np.uint8)

    # Find contours in binary mask image
    conts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Check for contours
    if len(conts[0]) < 1:
        return None
    # Grab contours
    conts = conts[0][0]

    cv2.fillPoly(mask, conts, (255))
    res = cv2.bitwise_and(img, img, mask=mask)  # Crop mask region from img
    rect = cv2.boundingRect(conts)  # Returns (x,y,w,h) of the bounding rect
    cropped = res[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
    pixel = cropped.ravel()  # 1D Array of pixels in cropped

    # Count pixel with value > 0
    vals = []
    for pix in pixel:
        if pix > 1:
            vals.append(pix)

    # Calculate echo intensity
    echo_int = round(np.mean(vals), 3)

    return echo_int
