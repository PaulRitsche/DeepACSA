"""Python module to automatically calcuate muscle area in ultrasound images"""

# Import necessary packages
from apo_model import ApoModel
from calibrate import calibrate_distance_efov
from calibrate import calibrate_distance_manually
from calibrate import calibrate_distance_static

import os

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
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
# from keras.preprocessing.image import array_to_img
# from keras.preprocessing.image import ImageDataGenerator
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use("ggplot")


def get_list_of_files(pathname: str):
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


def import_image_efov(path_to_image: str):
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
    img = cv2.imread(path_to_image, 0)
    org_img = img.copy()

    # print("Loaded image at " + path_to_image)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20, 20))
    img = clahe.apply(img)
    img = img_to_array(img)
    height = img.shape[0]
    weight = img.shape[1]
    img = np.reshape(img, [-1, height, weight, 1])
    img = resize(img, (1, 256, 256, 1), mode='constant', preserve_range=True)
    img = img/255.0

    return filename, org_img, img,  height, weight


def import_image(path_to_image: str):
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
    img = cv2.imread(path_to_image, 0)
    org_img = img.copy()

    # print("Loaded image at " + path_to_image)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20, 20))
    img = clahe.apply(img)
    img = img_to_array(img)
    height = img.shape[0]
    weight = img.shape[1]
    img = np.reshape(img, [-1, height, weight, 1])
    img = resize(img, (1, 256, 256, 1), mode='constant', preserve_range=True)
    img = img/255.0

    return filename, img, nonflipped_img, height, weight


def get_flip_flags_list(pathname: str):
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


def calc_area(depth: int, scalingline_length: int, img: np.ndarray):
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


def compile_save_results(rootpath: str, dataframe: pd.DataFrame):
    """Saves analysis results to excel and pdf files.

    Arguments:
        Path to root directory of files,
        filename (str),
        dataframe (pd.DataFrame) containing filename, muscle
        and predicted area

    Returns:
        Excel file containing filename, muscle and predicted area.

    Example:
    >>>compile_save_results(C:/Desktop/Test, C:/Desktop/Test/Img1.tif,
                            dataframe)
    """
    excelpath = rootpath + '/Results.xlsx'
    if os.path.exists(excelpath):
        with pd.ExcelWriter(excelpath, mode='a') as writer:
            data = dataframe
            data.to_excel(writer, sheet_name="Results")
    else:
        with pd.ExcelWriter(excelpath, mode='w') as writer:
            data = dataframe
            data.to_excel(writer, sheet_name="Results")


def calculate_batch_efov(rootpath: str, modelpath: str, depth: int,
                         muscle: str):
    """Calculates area predictions for batches of EFOV US images
        containing continous scaling line.

    Arguments:
        Path to root directory of images,
        path to model used for predictions,
        ultrasound scanning depth,
        analyzed muscle.
    """
    list_of_files = glob.glob(rootpath + '/**/*.tif', recursive=True)

    apo_model = ApoModel(modelpath)

    with PdfPages(rootpath + '/Analyzed_images.pdf') as pdf:

        dataframe = pd.DataFrame(columns=["File", "Muscle", "Area_cm²"])
        for imagepath in list_of_files:

            # load image
            filename, org_img, img, height, width = import_image_efov(imagepath)

            # find length of the scalingline
            scalingline_length = calibrate_distance_efov(imagepath, muscle)

            # predict area
            pred_apo_t, fig = apo_model.predict_t(org_img, img, width, height)
            area = calc_area(depth, scalingline_length, pred_apo_t)

            # append results to dataframe
            dataframe = dataframe.append({"File": filename,
                                          "Muscle": muscle,
                                          "Area_cm²": area},
                                         ignore_index=True)

            # save figures
            pdf.savefig(fig)
            plt.close(fig)

        # save predicted area values
        compile_save_results(rootpath, dataframe)


def calculate_batch(rootpath: str, flip_file_path: str, modelpath: str,
                    depth: int, spacing: int, muscle: str, scaling: str):
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

    apo_model = ApoModel(modelpath)
    dataframe = pd.DataFrame(columns=["File", "Muscle", "Area_cm²"])

    with PdfPages(rootpath + '/Analyzed_images.pdf') as pdf:

        if len(list_of_files) == len(flip_flags):

            for imagepath in list_of_files:

                # load image
                # flip = flip_flags.pop(0)
                imported = import_image(imagepath)
                filename, img, nonflipped_img, height, width = imported

                if scaling == "Static":
                    calibrate_fn = calibrate_distance_static
                else:
                    calibrate_fn = calibrate_distance_manually
                # find length of the scaling line
                scalingline_length = calibrate_fn(
                    nonflipped_img, spacing, depth
                )

                # predict area
                pred_apo_t, fig = apo_model.predict_t(img, width, height)
                area = calc_area(depth, scalingline_length, pred_apo_t)

                # append results to dataframe
                dataframe = dataframe.append({"File": filename,
                                              "Muscle": muscle,
                                              "Area_cm²": area},
                                             ignore_index=True)

                # save figures
                pdf.savefig(fig)
                plt.close(fig)

            # save predicted area results
            compile_save_results(rootpath, filename, dataframe)

        else:
            print("Warning: number of flipFlags and images doesn\'t match! " +
                  "Calculations aborted.")
