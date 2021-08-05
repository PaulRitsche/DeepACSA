"""Python module to automatically calcuate muscle area in US images"""

# Import necessary packages
import os
import glob
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from apo_model import ApoModel
from calibrate import calibrate_distance_efov
from calibrate import calibrate_distance_manually
from calibrate import calibrate_distance_static
from echo_int import calculate_echo_int
from skimage.transform import resize
from keras.preprocessing.image import img_to_array
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use("ggplot")
plt.switch_backend("agg")


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


def import_image_efov(path_to_image: str, muscle: str):
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
    rows,cols = img.shape
    img = img[75:rows,20:cols-10]
    img_copy = img.copy()


    # print("Loaded image at " + path_to_image)
    if muscle == "RF":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    if muscle == "VL":
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20, 20))
    if muscle == "GM":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    if muscle == "GL":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    img = clahe.apply(img)
    img = img_to_array(img)
    height = img.shape[0]
    weight = img.shape[1]
    img = np.reshape(img, [-1, height, weight, 1])
    img = resize(img, (1, 256, 256, 1), mode='constant', preserve_range=True)
    img = img/255.0

    return filename, img_copy, img,  height, weight


def import_image(path_to_image: str, muscle: str):
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
    nonflipped_img = img.copy()

    # print("Loaded image at " + path_to_image)
    if muscle == "RF":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    if muscle == "VL":
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20, 20))
    if muscle == "GM":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    if muscle == "GL":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    img = clahe.apply(img)
    img = img_to_array(img)
    height = img.shape[0]
    weight = img.shape[1]
    img = np.reshape(img, [-1, height, weight, 1])
    img = resize(img, (1, 256, 256, 1), mode='constant', preserve_range=True)
    img = img/255.0

    return filename, img, nonflipped_img, height, weight


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


def calc_area(depth: float, scalingline_length: float, img: np.ndarray):
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
    pix_per_cm = scalingline_length / float(depth)
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


def calculate_batch_efov(rootpath: str, filetype: str, modelpath: str,
                         depth: float, muscle: str, gui):
    """Calculates area predictions for batches of EFOV US images
        containing continous scaling line.

    Arguments:
        Path to root directory of images,
        type of image files,
        path to model used for predictions,
        ultrasound scanning depth,
        analyzed muscle.
    """
    list_of_files = glob.glob(rootpath + filetype, recursive=True)

    apo_model = ApoModel(modelpath)

    with PdfPages(rootpath + '/Analyzed_images.pdf') as pdf:

        try:
            dataframe = pd.DataFrame(columns=["File", "Muscle", "Area_cm²"])

            for imagepath in list_of_files:

                if gui.should_stop:
                    # there was an input to stop the calculations
                    break

                # load image
                imported = import_image_efov(imagepath, muscle)
                filename, img_copy, img, height, width = imported

                calibrate_efov = calibrate_distance_efov
                # find length of the scalingline
                scalingline_length, img_lines = calibrate_efov(imagepath, muscle)

                # predict area
                pred_apo_t, fig = apo_model.predict_e(img, img_lines,
                                                          width, height)
                echo = calculate_echo_int(img_copy, pred_apo_t)
                area = calc_area(depth, scalingline_length, pred_apo_t)

                # append results to dataframe
                dataframe = dataframe.append({"File": filename,
                                              "Muscle": muscle,
                                              "Area_cm²": area,
                                              "Echo_intensity": echo},
                                             ignore_index=True)

                # save figures
                pdf.savefig(fig)
                plt.close(fig)

        except:
            pass

        finally:
            # save predicted area values
            compile_save_results(rootpath, dataframe)
            # clean up
            gui.should_stop = False
            gui.is_running = False


def calculate_batch(rootpath: str, filetype: str, modelpath: str,
                    depth: float, spacing: int, muscle: str,
                    scaling: str, gui):
    """Calculates area predictions for batches of (EFOV) US images
        not containing a continous scaling line.

        Arguments:
            Path to root directory of images,
            type of image files,
            path to txt file containing flipping information for images,
            path to model used for predictions,
            ultrasound scanning depth,
            distance between (vertical) scaling lines (mm),
            analyzed muscle,
            scaling type.
    """
    list_of_files = glob.glob(rootpath + filetype, recursive=True)

    apo_model = ApoModel(modelpath)
    dataframe = pd.DataFrame(columns=["File", "Muscle", "Area_cm²"])

    with PdfPages(rootpath + '/Analyzed_images.pdf') as pdf:

        try:

            for imagepath in list_of_files:

                if gui.should_stop:
                    # there was an input to stop the calculations
                    break

                # load image
                imported = import_image(imagepath, muscle)
                filename, img, nonflipped_img, height, width = imported

                if scaling == "Bar":
                    calibrate_fn = calibrate_distance_static
                    # find length of the scaling line
                    scalingline_length, imgscale, dist = calibrate_fn(
                    nonflipped_img, imagepath, spacing, depth
                    )
                else:
                    calibrate_fn = calibrate_distance_manually
                    scalingline_length = calibrate_fn(
                    nonflipped_img, spacing, depth
                    )

                # predict area
                pred_apo_t, fig = apo_model.predict_s(img, imgscale, dist,
                                                     width, height)
                echo = calculate_echo_int(nonflipped_img, pred_apo_t)
                area = calc_area(depth, scalingline_length, pred_apo_t)

                # append results to dataframe
                dataframe = dataframe.append({"File": filename,
                                              "Muscle": muscle,
                                              "Area_cm²": area,
                                              "Echo_intensity": echo},
                                              ignore_index=True)

                # save figures
                pdf.savefig(fig)
                plt.close(fig)

        except:
            pass

        finally:
            # save predicted area results
            compile_save_results(rootpath, dataframe)
            # clean up
            gui.should_stop = False
            gui.is_running = False
