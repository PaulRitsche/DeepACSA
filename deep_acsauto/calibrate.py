"""Python module which provides functions to calibrate US images."""

import math
import numpy as np
import pandas as pd
import cv2

class ScalinglineError(Exception):
    pass

class StaticScalingError(Exception):
    pass


mlocs = []


def region_of_interest(img, vertices):
    """Defines region of interest where ridges are searched.

    Arguments:
        Processed image containing edges,
        numpy array of regions of interest vertices / coordinates.

    Returns:
        Masked image of ROI containing only ridges detected by preprocessing.

    Example:
        >>>region_of_interest(preprocessed_image,
        np.array([(0,1), (0,2), (4,2), (4,7)], np.int32),)
    """
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def mclick(event, x_val, y_val, flags, param):
    """Detect mouse clicks for purpose of image calibration.

    Arguments:

    Returns:
        List of y coordinates of clicked points.
    """
    global mlocs
    # if the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        mlocs.append(y_val)


def draw_the_lines(img, lines):
    """Draws lines along the detected ridges.

    Arguments:
        Original image,
        numpy array of detected lines by Hough-Transform.

    Returns:
        Original images containing lines.

    Example:
        >>>draw_the_lines(Image1.tif,
        np.array([[0 738 200 539]))
    """
    img = np.copy(img)
    # Creating empty image to draw lines on
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x_1, y_1, x_2, y_2 in line:
            cv2.line(blank_image, (x_1, y_1), (x_2, y_2), (0, 255, 0),
                     thickness=1)

    # Overlay image with lines on original images (only needed for plotting)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def calibrate_distance_efov(path_to_image: str, arg_muscle: str):
    """Calculates scalingline length of image based computed
        length of detected rigdes.

        Arguments:
            Path to image that should be analyzed.

        Returns:
            Length of scalingline (pixel).

        Example:
            >>>calibrate_distance_efov(C:/Desktop/Test/Img1.tif)
            571
    """
    image = cv2.imread(path_to_image)
    # Transform BGR Image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    # Define ROI with scaling lines
    region_of_interest_vertices = [
        (150, height),
        (150, 80),
        (1100, 80),
        (1100, height)
    ]
    # Transform RGB to greyscale for edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Edge detecition
    canny_image = cv2.Canny(gray_image, 400, 600)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices],
                                                np.int32),)

    # For RF
    muscle = arg_muscle
    if muscle == "RF":
        lines = cv2.HoughLinesP(cropped_image,
                                rho=1,
                                theta=np.pi/180,
                                threshold=50,
                                lines=np.array([]),
                                minLineLength=350,
                                maxLineGap=1)
        image_with_lines = draw_the_lines(image, lines)

    # For VL
    if muscle == "VL":
        lines = cv2.HoughLinesP(cropped_image,
                                rho=1,  # Distance of pixels in accumulator
                                theta=np.pi/180,  # Angle resolution
                                threshold=50,  # Only lines with higher vote
                                lines=np.array([]),
                                minLineLength=200,
                                maxLineGap=3)  # Gap between lines
        image_with_lines = draw_the_lines(image, lines)

    # For GM / GL
    if muscle == "GL" or "GM":
        lines = cv2.HoughLinesP(cropped_image,
                                rho=1,  # Distance of pixels in accumulator
                                theta=np.pi / 180,  # Angle resolution
                                threshold=50,  # Only lines with higher vote
                                lines=np.array([]),
                                minLineLength=250,
                                maxLineGap=3)
        image_with_lines = draw_the_lines(image, lines)
    
    if lines is None: 
        raise ScalinglineError(f"Scalingline not found in {path_to_image}")

    # Calculate length of the scaling line   
    scalingline = lines[0][0]
    point1 = [scalingline[0], scalingline[1]]
    point2 = [scalingline[2], scalingline[3]]
    scalingline_length = math.sqrt(((point1[0] - point2[0])**2)
                                   + ((point1[1] - point2[1])**2))
    
    return scalingline_length, image_with_lines


def calibrate_distance_static(nonflipped_img, path_to_image: str, spacing: int, 
                              depth: float, flip: int):
    """Calculates scalingline length of image based computed
        distance between two points on image and image depth.

    Arguments:
        Original(nonflipped) image with scaling lines on right border,
        Path to image that should be analyzed,
        distance between scaling points (mm),
        US scanning depth (cm), 
        flip flag of image.

    Returns:
        Length of scaling line (pixel).

    Example:
        >>>calibrate_distance_manually(Image, 5, 4.5, 0)
        5 mm corresponds to 95 pixels
    """
    # calibrate according to scale at the right border of image
    if flip == 1:
        nonflipped_img = np.fliplr(nonflipped_img)
    img2 = np.uint8(nonflipped_img)
    imgscale = img2[70:, 1100:1115]
    # search for rows with white pixels, calculate median of distance
    calib_dist = np.median(np.diff(np.argwhere(imgscale.sum(axis=1) > 200),
                                   axis=0))
   
    if pd.isnull(calib_dist) is True: 
        raise StaticScalingError(f"Spacing not found in {path_to_image}")

    scalingline_length = depth * calib_dist

    print(str(spacing) + ' mm corresponds to ' + str(calib_dist) + ' pixels')

    return float(scalingline_length)


def calibrate_distance_manually(nonflipped_img, spacing: int, depth: float):
    """Calculates scalingline length of image based on manual specified
        distance between two points on image and image depth.

    Arguments:
        Original(nonflipped) image,
        distance between scaling points (mm),
        US scanning depth (cm).

    Returns:
        Length of scaling line (pixel).

    Example:
        >>>calibrate_distance_manually(Image, 5, 4.5)
        5 mm corresponds to 99 pixels
    """
    img2 = np.uint8(nonflipped_img)

    # display the image and wait for a keypress
    cv2.imshow("image", img2)
    cv2.setMouseCallback("image", mclick)
    key = cv2.waitKey(0)

    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()

    calib_dist = np.abs(mlocs[0] - mlocs[1])
    scalingline_length = depth * calib_dist

    print(str(spacing) + ' mm corresponds to ' + str(calib_dist) + ' pixels')

    return float(scalingline_length)
