"""Python module to automatically calcuate muscle area in ultrasound images"""

# Import necessary packages
from __future__ import division
import os
import math
import glob
import pandas as pd
import numpy as np
# import openpyxl
from skimage.transform import resize
# from skimage.morphology import skeletonize
# from scipy.signal import resample, savgol_filter, butter, filtfilt
# from PIL import Image, ImageDraw
import cv2
# from cv2 import EVENT_LBUTTONDOWN
# import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model  # Model
from keras.preprocessing.image import img_to_array, load_img  # array_to_img, ImageDataGenerator
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use("ggplot")

# GLOBAL VARIABLES #
mlocs = []


def get_list_of_files(pathname):
    """Get a list of all files in the directory.

    Arguments:
        One path to root directory.

    Returns:
        List of all files in root directory.

    Example:
        >>> get_list_of_files(C:/Desktop/Test)
        ["C:/Desktop/Test/Img1.tif", "C:/Desktop/Test/Img2.tif",
        "C:/Desktop/Test/Flip.txt"]
    """
    return glob.glob(pathname)


def import_image_efov(path_to_image):
    """Define the image to analyse, import and reshape the image.

    Arguments:
        Path to image that should be analyzed.

    Returns:
        Filename, image, image height, image width

    Example:
        >>>import_image(C:/Desktop/Test/Img1.tif)
        (Img1.tif, array[[[[...]]]], 864, 1152)
    """

    image_add = path_to_image
    filename = os.path.splitext(os.path.basename(image_add))[0]
    img = load_img(image_add, color_mode='grayscale')

    # print("Loaded image at " + path_to_image)
    img = img_to_array(img)
    height = img.shape[0]
    weight = img.shape[1]
    img = np.reshape(img, [-1, height, weight, 1])
    img = resize(img, (1, 256, 256, 1), mode='constant', preserve_range=True)
    img = img/255.0

    return filename, img,  height, weight


def import_image(path_to_image):
    """Define the image to analyse, import and reshape the image.

    Arguments:
        Path to image that should be analyzed.

    Returns:
        Filename, image, copy of image, image not flipped,
        image height, image width

    Example:
        >>>import_image(C:/Desktop/Test/Img1.tif)
        (Img1.tif, array[[[[...]]]],
        <PIL.Image.Image image mode=L size=1152x864 at 0x1FF843A2550>,
        <PIL.Image.Image image mode=L size=1152x864 at 0x1FF843A2550>,
        864, 1152)
    """

    image_add = path_to_image
    filename = os.path.splitext(os.path.basename(image_add))[0]
    img = load_img(image_add, color_mode='grayscale')

    # print("Loaded image at " + path_to_image)
    nonflipped_img = img
    img = img_to_array(img)
    height = img.shape[0]
    weight = img.shape[1]
    img = np.reshape(img, [-1, height, weight, 1])
    img = resize(img, (1, 256, 256, 1), mode='constant', preserve_range=True)
    img = img/255.0

    return filename, img, nonflipped_img, height, weight


def get_flip_flags_list(pathname):
    """Define the path to text file including flipping flags.

    Arguments:
        Path to Flip.txt file.

    Returns:
        List of flipping flags.

    Example:
        >>>get(C:/Desktop/Test)
        ["0", "1"]
    """
    flip_flags = []
    file = open(pathname, 'r')
    for line in file:
        for digit in line:
            if digit.isdigit():
                flip_flags.append(digit)
    return flip_flags


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


def calibrate_distance_manually(nonflipped_img, spacing, depth):
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

    return scalingline_length


def calibrate_distance_static(nonflipped_img, spacing, depth):
    """Calculates scalingline length of image based computed
        distance between two points on image and image depth.

    Arguments:
        Original(nonflipped) image,
        distance between scaling points (mm),
        US scanning depth (cm).

    Returns:
        Length of scaling line (pixel).

    Example:
        >>>calibrate_distance_manually(Image, 5, 4.5)
        5 mm corresponds to 95 pixels
    """
    # calibrate according to scale at the right border of image
    img2 = np.uint8(nonflipped_img)
    imgscale = img2[70:, 1100:1115]
    # search for rows with white pixels, calculate median of distance
    calib_dist = np.median(np.diff(np.argwhere(imgscale.sum(axis=1) > 200),
                                   axis=0))
    scalingline_length = depth * calib_dist

    print(str(spacing) + ' mm corresponds to ' + str(calib_dist) + ' pixels')

    return scalingline_length


# Optional, just for plotting
def plot_image(image):
    """Plots image with detected ridges/scalingline.

    Arguments:
        Image containing detected ridges.

    Returns:
        Plot of image containing detected Ridges.
    """
    img = image
    fig, (ax1) = plt.subplots(1, 1, figsize=(15, 15))
    ax1.imshow(img, cmap="gray")
    ax1.grid(False)
    plt.savefig("Ridge_test_1.tif")


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


def calibrate_distance_efov(path_to_image, arg_muscle):
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
                                minLineLength=400,
                                maxLineGap=1)
        # image_with_lines = draw_the_lines(image, lines)

    # For VL
    if muscle == "VL":
        lines = cv2.HoughLinesP(cropped_image,
                                rho=1,  # Distance of pixels in accumulator
                                theta=np.pi/180,  # Angle resolution
                                threshold=50,  # Only lines with higher vote
                                lines=np.array([]),
                                minLineLength=200,
                                maxLineGap=3)  # Gap between lines
        # image_with_lines = draw_the_lines(image, lines)

    # Calculate length of the scaling line
    scalingline = lines[0][0]
    point1 = [scalingline[0], scalingline[1]]
    point2 = [scalingline[2], scalingline[3]]
    scalingline_length = math.sqrt(((point1[0] - point2[0])**2)
                                   + ((point1[1] - point2[1])**2))
    # plot_image(image_with_lines)
    return scalingline_length


def IoU(y_true, y_pred, smooth=1):
    """Computes intersection over union (IoU), a measure of labelling accuracy.

    Arguments:

    Returns:
        Intersection over union scores.

    Example:
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def do_predictions(img, height, weight, modelpath):
    """Get preditions of model for the image.

    Arguments: Original image,
               image height (int),
               image width (int),
               filename (str) (only used when plotted seperately),
               path to model used for prediction.

    Returns:
        Image of thresholded, binary muscle areaprediction,
        figure containing original image and model prediction.

    Example:
        >>>do_predictions(image, 1152, 856, Image1.tif, model)
    """
    # Set threshold with minimal confidence to make binary
    apo_threshold = 0.5
    model_apo = load_model(modelpath, custom_objects={'IoU': IoU})
    pred_apo = model_apo.predict(img)
    # Makes binary using integer between 0 and 255 -> np.uint8
    pred_apo_t = (pred_apo > apo_threshold).astype(np.uint8)
    img = resize(img, (1, height, weight, 1))
    img = np.reshape(img, (height, weight))
    pred_apo = resize(pred_apo, (1, height, weight, 1))
    pred_apo = np.reshape(pred_apo, (height, weight))
    pred_apo_t = resize(pred_apo_t, (1, height, weight, 1))
    pred_apo_t = np.reshape(pred_apo_t, (height, weight))

    fig = plt.figure(figsize=(20, 20))
    # Fist is n rows in grid, second n columns, third is position
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img.squeeze(), cmap='gray')
    ax1.grid(False)
    ax1.set_title('Original image')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(pred_apo_t.squeeze(), cmap="gray")
    ax2.grid(False)
    ax2.set_title('Aponeuroses')

    return pred_apo_t, fig


def calc_area(depth, scalingline_length, img):
    """Calculates predicted muscle aread.

    Arguments:
        Scanning depth (cm),
        Scalingline length (pixel),
        thresholded binary model prediction.

    Returns:
        Predicted muscle area (cm²).

    Example:
        >>>calc_area(float(4.5), int(571), Image1.tif)
        3.813
    """
    pix_per_cm = scalingline_length / depth
    # Counts pixels with values != 0
    pred_muscle_area = cv2.countNonZero(img) / pix_per_cm**2
    # print(pred_muscle_area)
    return pred_muscle_area


def compile_save_results(rootpath, filename, muscle, area):
    """Saves analysis results to excel and pdf files.

    Arguments:
        Path to root directory of files,
        filename (str),
        defined muscle for analysis (str),
        predicted muscle area (int)

    Returns:
        Excel file containing filename, muscle and predicted area,
        pdf file containing pairs of original images and model predictions.

    Example:
    >>>compile_save_results(C:/Desktop/Test, C:/Desktop/Test/Img1.tif,
                            Image1.tif, str("RF"), float(3.813))
    """
    excelpath = rootpath + '/Results.xlsx'
    if os.path.exists(excelpath):
        with pd.ExcelWriter(excelpath, mode='a') as writer:
            data = pd.DataFrame({'Image_ID': filename,
                                'Muscle': muscle, 'Area_cm²': area},
                                index=[0])
            data.to_excel(writer)
    else:
        with pd.ExcelWriter(excelpath, mode='w') as writer:
            data = pd.DataFrame({'Image_ID': filename,
                                'Muscle': muscle, 'Area_cm²': area},
                                index=[0])
            data.to_excel(writer)


def calculate_batch_efov(rootpath, modelpath, depth, muscle):
    """Calculates area predictions for batches of EFOV US images
        containing continous scaling line.

    Arguments:
        Path to root directory of images,
        path to model used for predictions,
        ultrasound scanning depth,
        analyzed muscle.
    """
    list_of_files = glob.glob(rootpath + '/**/*.tif', recursive=True)

    with PdfPages(rootpath + '/Analyzed_images.pdf') as pdf:

        for imagepath in list_of_files:

            filename, img, height, weight = import_image_efov(imagepath)
            scalingline_length = calibrate_distance_efov(imagepath, muscle)

            img, fig = do_predictions(img, height, weight, modelpath)
            area = calc_area(depth, scalingline_length, img)
            compile_save_results(rootpath, filename, muscle, area)
            pdf.savefig(fig)
            plt.close(fig)


def calculate_batch(rootpath, flip_file_path, modelpath, depth,
                    spacing, muscle, scaling):
    """Calculates area predictions for batches of (EFOV) US images
        not containing a continous scaling line.

        Arguments:
            Path to root directory of images,
            path to txt file containing flipping information for images,
            path to model used for predictions,
            ultrasound scanning depth,
            distance between (vertical) scaling lines (mm),
            analyzed muscle,
            scaling type.
    """
    list_of_files = glob.glob(rootpath + '/**/*.tif', recursive=True)
    flip_flags = get_flip_flags_list(flip_file_path)

    with PdfPages(rootpath + '/Analyzed_images.pdf') as pdf:

        if len(list_of_files) == len(flip_flags):

            for imagepath in list_of_files:

                # flip = flip_flags.pop(0)
                filename, img, nonflipped_img, height, weight = import_image(imagepath)

                if scaling == "Static":

                    scalingline_length = calibrate_distance_static(nonflipped_img, spacing, depth)

                else:

                    scalingline_length = calibrate_distance_manually(nonflipped_img, spacing, depth)

                img, fig = do_predictions(img, height, weight, modelpath)
                area = calc_area(depth, scalingline_length, img)
                compile_save_results(rootpath, filename, muscle, area)
                pdf.savefig(fig)
                plt.close(fig)
        else:
            print("Warning: number of flipFlags and images doesn\'t match! " +
                  "Calculations aborted.")
