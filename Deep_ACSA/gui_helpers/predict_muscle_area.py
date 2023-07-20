"""Python module to automatically calcuate muscle area in US images

Description
-----------
This Python module provides a collection of functions to automatically calculate muscle area in
ultrasound (US) images. It includes functions for importing and preprocessing images, predicting
aponeurosis areas, calculating echo intensity, and optionally estimating muscle volume based on the predicted areas.
The module also offers various calibration methods, including continuous scaling lines (EFOV images)
and scaling bars or manual calibration for regular ultrasound images. The results are saved in a
Pandas DataFrame and exported to an Excel file. The module is designed to streamline the analysis
of muscle area in large batches of ultrasound images for research and medical purposes.

Functions scope
---------------
get_list_of_files
    Gets a list of all files in the directory that match the specified pattern.
import_image_efov
    Imports and preprocesses an EFOV image, returning the filename, preprocessed image,
    original image, height, and width.
import_image
    Imports and preprocesses an image, returning the filename, preprocessed image,
    original image, height, and width.
calc_area_efov
    Calculates the predicted muscle area in the region of interest (ROI) of an EFOV ultrasound image.
calc_area_efov
    Calculates the predicted muscle area in an ultrasound image using a known calibration distance.
compile_save_results
    Saves analysis results to an Excel file.
calculate_batch_efov
    Calculates area predictions for batches of EFOV US images containing a continuous scaling line,
    including echo intensity and optional muscle volume calculations.
calculate_batch
    Calculates area predictions for batches of (EFOV) US images not containing a continuous scaling line,
    including echo intensity and optional muscle volume calculations.
"""

import glob
import os
import time
import tkinter as tk
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from skimage.transform import resize
from tensorflow.keras.utils import img_to_array

from Deep_ACSA.gui_helpers.apo_model import ApoModel
from Deep_ACSA.gui_helpers.calculate_muscle_volume import muscle_volume_calculation
from Deep_ACSA.gui_helpers.calibrate import (
    calibrate_distance_efov,
    calibrate_distance_manually,
    calibrate_distance_static,
)
from Deep_ACSA.gui_helpers.echo_int import calculate_echo_int

# from keras.preprocessing.image import img_to_array, load_img


plt.style.use("ggplot")
plt.switch_backend("agg")


def get_list_of_files(pathname: str):
    """Get a list of all files in the directory.
    
    Parameters
    ----------
    pathname : str
        The pathname pattern to match against file paths.

    Returns
    -------
    List[str]
        A list of file paths that match the specified pathname pattern.

    Example
    -------
        >>> get_list_of_files(C:/Desktop/Test)
        ["C:/Desktop/Test/Img1.tif", "C:/Desktop/Test/Img2.tif",
        "C:/Desktop/Test/Flip.txt"]
    """
    return glob.glob(pathname)


def import_image_efov(path_to_image: str):
    """Define the image to analyse, import and reshape the image.

    Parameters
    ----------
    path_to_image : str
        The file path to the EFOV image.

    Returns
    -------
    Tuple[str, np.ndarray, np.ndarray, int, int]
        A tuple containing the following elements:
        - filename (str): The filename of the EFOV image without the extension.
        - img_copy (np.ndarray): A copy of the original EFOV image as a NumPy array.
        - img (np.ndarray): The preprocessed EFOV image as a NumPy array with shape (1, 256, 256, 3).
        - height (int): The height of the original EFOV image.
        - width (int): The width of the original EFOV image.

    Notes
    -----
    - The function reads the EFOV image using OpenCV (cv2) library.
    - The function crops the image to remove unwanted margins (75 pixels from the top, 20 pixels from the left,
      and 10 pixels from the right).
    - The function then reshapes the image and resizes it to a fixed size of (256, 256).
    - The final preprocessed image is normalized to have values in the range [0, 1].

    Example
    -------
        >>>import_image(C:/Desktop/Test/Img1.tif)
        (Img1.tif, array[[[[...]]]], 864, 1152)
    """
    image_add = path_to_image
    filename = os.path.splitext(os.path.basename(image_add))[0]
    img = cv2.imread(path_to_image, 1)
    rows, cols, channels = img.shape
    img = img[75:rows, 20 : cols - 10]
    img_copy = img.copy()

    img = img_to_array(img)
    height = img.shape[0]
    weight = img.shape[1]
    img = np.reshape(img, [-1, height, weight, 3])
    img = resize(img, (1, 256, 256, 3), mode="constant", preserve_range=True)
    img = img / 255.0

    return filename, img_copy, img, height, weight


def import_image(path_to_image: str):
    """Define the image to analyse, import and reshape the image.

    Parameters
    ----------
    path_to_image : str
        The file path to the image that should be analyzed.

    Returns
    -------
    Tuple[str, np.ndarray, np.ndarray, int, int]
        A tuple containing the following elements:
        - filename (str): The filename of the image without the extension.
        - img (np.ndarray): The preprocessed image as a NumPy array with shape (1, 256, 256, 3).
        - img_copy (np.ndarray): A copy of the original image as a NumPy array.
        - height (int): The height of the original image.
        - width (int): The width of the original image.

    Example
    -------
        >>>import_image(C:/Desktop/Test/Img1.tif)
        (Img1.tif, array[[[[...]]]],
        <PIL.Image.Image image mode=L size=1152x864 at 0x1FF843A2550>,
        <PIL.Image.Image image mode=L size=1152x864 at 0x1FF843A2550>,
        864, 1152)
    """
    image_add = path_to_image
    filename = os.path.splitext(os.path.basename(image_add))[0]
    img = cv2.imread(path_to_image, 1)
    nonflipped_img = img.copy()

    img = img_to_array(img)
    height = img.shape[0]
    weight = img.shape[1]
    img = np.reshape(img, [-1, height, weight, 3])
    img = resize(img, (1, 256, 256, 3), mode="constant", preserve_range=True)
    img = img / 255.0

    return filename, img, nonflipped_img, height, weight


def calc_area_efov(depth: float, scalingline_length: int, img: np.ndarray):
    """Calculates predicted muscle aread.

    Parameters
    ----------
    depth : float
        The depth of the region of interest (ROI) in centimeters.
    scalingline_length : int
        The length of the scaling line in pixels.
    img : np.ndarray
        The eFOV ultrasound image as a NumPy array.

    Returns
    -------
    pred_muscle_area : float
        The predicted muscle area in the ROI based on the eFOV ultrasound image.

    Example
    -------
        >>>calc_area(float(5), int(254), Image1.tif)
        3.813
    """
    pix_per_cm = scalingline_length / depth
    # Counts pixels with values != 0
    pred_muscle_area = cv2.countNonZero(img) / (pix_per_cm**2)
    return pred_muscle_area


def calc_area(calib_dist: float, img: np.ndarray):
    """Calculates predicted muscle aread.

    Parameters
    ----------
    calib_dist : float
        The calibration distance in centimeters, representing the known distance in the image.
    img : np.ndarray
        The ultrasound image as a NumPy array.

    Returns
    -------
    pred_muscle_area : float
        The predicted muscle area in square centimeters based on the calibration distance.

    Examples
    --------
    >>>calc_area(int(54), Image1.tif)
    3.813
    """
    pix_per_cm = calib_dist

    # Counts pixels with values != 0
    pred_muscle_area = cv2.countNonZero(img) / (pix_per_cm**2)
    return pred_muscle_area


def compile_save_results(rootpath: str, dataframe: pd.DataFrame):
    """Saves analysis results to excel and pdf files.

    Parameters
    ----------
    rootpath : str
        The root path where the Excel file will be saved.
    dataframe : pd.DataFrame
        The Pandas DataFrame containing the results.

    Returns
    -------
        Excel file containing filename, muscle and predicted area.

    Example
    -------
    >>>compile_save_results(C:/Desktop/Test, dataframe)
    """
    excelpath = rootpath + "/Results.xlsx"
    with pd.ExcelWriter(excelpath, mode="w") as writer:
        data = dataframe
        data.to_excel(writer, sheet_name="Results")


def calculate_batch_efov(
    rootpath: str,
    filetype: str,
    modelpath: str,
    loss_function: str,
    depth: float,
    muscle: str,
    volume_wanted: str,
    distance_acsa: float,
    gui,
):
    """Calculates area predictions for batches of EFOV US images
        containing continous scaling line.
        This function takes a batch of eFOV ultrasound images, predicts the aponeurosis area,
        calculates echo intensity, and optionally calculates muscle volume based on the predicted areas.
        The results are compiled into a Pandas DataFrame and saved to an Excel file.

    Parameters
    ----------
    rootpath : str
        The root path where the eFOV images are located.
    filetype : str
        The file extension or pattern to match eFOV image files (e.g., "*.tif").
    modelpath : str
        The path to the pre-trained aponeurosis detection model.
    loss_function : str
        The loss function used during model training (e.g., "BCE" for binary cross-entropy).
    depth : float
        The depth (in centimeters) of the eFOV ultrasound image.
    muscle : str
        The name or type of muscle being analyzed.
    volume_wanted : str
        Whether to calculate muscle volume based on the predicted aponeurosis areas ("Yes" or "No").
    distance_acsa : float
        The distance (in centimeters) between adjacent aponeurosis areas for volume calculation.
    gui : tkinter.Tk
        The Tkinter root window to interact with the graphical user interface.

    Returns
    -------
    None

    Notes
    -----
    - The function uses a pre-trained aponeurosis detection model to predict aponeurosis areas in each eFOV image.
    - The function calculates the echo intensity (mean grey value) of the predicted muscle area in each image.
    - If volume_wanted is set to "Yes", the function calculates the muscle volume based on the predicted aponeurosis areas.
    - The function saves the results in a Pandas DataFrame and exports them to an Excel file in the rootpath.
    - The function also saves the analyzed images as a multi-page PDF file named "Analyzed_images.pdf" in the rootpath.
    - If any image processing or calculation error occurs, the function will record the failed files in "failed_images.txt".

    Example
    -------
    >>> rootpath = "/path/to/directory"
    >>> filetype = "*.tif"
    >>> modelpath = "/path/to/pretrained_model.h5"
    >>> loss_function = "BCE"
    >>> depth = 4.5
    >>> muscle = "Quadriceps"
    >>> volume_wanted = "Yes"
    >>> distance_acsa = 2.0
    >>> gui = ...  # Your Tkinter root window
    >>> calculate_batch_efov(rootpath, filetype, modelpath, loss_function, depth, muscle, volume_wanted, distance_acsa, gui)
    # The function will process the eFOV images in the specified directory, predict the aponeurosis areas,
    # calculate echo intensity, and optionally calculate muscle volume based on the predicted areas.
    # The results will be compiled into a DataFrame and saved to an Excel file.
    """
    list_of_files = glob.glob(rootpath + filetype, recursive=True)

    if len(list_of_files) == 0:
        tk.messagebox.showerror(
            "Information",
            "No image files found." + "\nPotential error source: Unmatched filetype",
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    apo_model = ApoModel(gui, model_path=modelpath, loss_function=loss_function)

    dataframe = pd.DataFrame(
        columns=[
            "File",
            "Muscle",
            "Area_cm2",
            "Echo_intensity",
            "A_C_ratio",
            "Circumference",
            "Volume_cm3",
        ]
    )
    failed_files = []

    calculated_areas = []

    with PdfPages(rootpath + "/Analyzed_images.pdf") as pdf:

        try:
            start_time = time.time()

            for imagepath in list_of_files:

                if gui.should_stop:
                    # there was an input to stop the calculations
                    break

                # load image
                imported = import_image_efov(imagepath)
                filename, img_copy, img, height, width = imported

                calibrate_efov = calibrate_distance_efov
                # find length of the scalingline
                scalingline_length, img_lines = calibrate_efov(imagepath, muscle)
                # check for ScalinglineError
                if scalingline_length is None:
                    fail = f"Scalingline not found in {imagepath}"
                    failed_files.append(fail)
                    warnings.warn("Image fails with ScalinglineError")
                    continue

                # predict area
                circum, pred_apo_t, fig = apo_model.predict_e(
                    gui, img, img_lines, filename, width, height
                )
                echo = calculate_echo_int(img_copy, pred_apo_t)
                if echo is None:
                    warnings.warn("Image fails with EchoIntensityError")
                    continue

                # calculate area
                area = calc_area_efov(depth, scalingline_length, pred_apo_t)
                area_circum_ratio = (area / circum) * 100

                # add the area to the list for volume calculation
                calculated_areas.append(area)

                # append results to dataframe
                dataframe = dataframe.append(
                    {
                        "File": filename,
                        "Muscle": muscle,
                        "Area_cm2": area,
                        "Echo_intensity": echo,
                        "A_C_ratio": area_circum_ratio,
                        "Circumference": circum,
                        "Volume_cm3": "",
                    },
                    ignore_index=True,
                )

                # save figures
                pdf.savefig(fig)
                plt.close(fig)
                # time duration of analysis of single image
                duration = time.time() - start_time
                print(f"duration: {duration}")

            # musclevolume calculation with the calculated areas
            if volume_wanted == "Yes":

                muscle_volume = muscle_volume_calculation(
                    calculated_areas, distance_acsa
                )

                # append musclevolume result to dataframe
                dataframe = dataframe.append(
                    {
                        "File": "",
                        "Muscle": "",
                        "Area_cm2": "",
                        "Echo_intensity": "",
                        "A_C_ratio": "",
                        "Circumference": "",
                        "Volume_cm3": muscle_volume,
                    },
                    ignore_index=True,
                )
            else:
                pass

        except ValueError:
            tk.messagebox.showerror(
                "Information",
                "Scaling Type Error."
                + "\nPotential error source: Selected scaling type does not fit image",
            )
            gui.should_stop = False
            gui.is_running = False
            gui.do_break()

        finally:
            # save predicted area values
            compile_save_results(rootpath, dataframe)
            # write failed images in file
            if len(failed_files) >= 1:
                file = open(rootpath + "/failed_images.txt", "w")
                for fail in failed_files:
                    file.write(fail + "\n")
                file.close()
            # clean up
            gui.should_stop = False
            gui.is_running = False


def calculate_batch(
    rootpath: str,
    filetype: str,
    modelpath: str,
    loss_function: str,
    spacing: str,
    muscle: str,
    scaling: str,
    volume_wanted: str,
    distance_acsa: float,
    gui,
):
    """Calculates area predictions for batches of (EFOV) US images
    not containing a continous scaling line.
    This function takes a batch of ultrasound images, predicts the aponeurosis area,
    calculates echo intensity, and optionally calculates muscle volume based on the predicted areas.
    The results are compiled into a Pandas DataFrame and saved to an Excel file.

    Parameters
    ----------
    rootpath : str
        The root path where the ultrasound images are located.
    filetype : str
        The file extension or pattern to match image files (e.g., "*.tif").
    modelpath : str
        The path to the pre-trained aponeurosis detection model.
    loss_function : str
        The loss function used during model training (e.g., "BCE" for binary cross-entropy).
    spacing : str
        The spacing between reference markers on the image (used for scaling calibration).
    muscle : str
        The name or type of muscle being analyzed.
    scaling : str
        The type of scaling used for calibration ("Bar" for scaling bars, "Manual" for manual calibration).
    volume_wanted : str
        Whether to calculate muscle volume based on the predicted aponeurosis areas ("Yes" or "No").
    distance_acsa : float
        The distance (in centimeters) between adjacent aponeurosis areas for volume calculation.
    gui : tkinter.Tk
        The Tkinter root window to interact with the graphical user interface.

    Returns
    -------
    None

    Notes
    -----
    - The function uses a pre-trained aponeurosis detection model to predict aponeurosis areas in each ultrasound image.
    - The function calculates the echo intensity (mean grey value) of the predicted muscle area in each image.
    - The scaling type ("Bar" or "Manual") is used to determine the calibration method for scaling.
    - If volume_wanted is set to "Yes", the function calculates the muscle volume based on the predicted aponeurosis areas.
    - The function saves the results in a Pandas DataFrame and exports them to an Excel file in the rootpath.
    - The function also saves the analyzed images as a multi-page PDF file named "Analyzed_images.pdf" in the rootpath.
    - If any image processing or calculation error occurs, the function will record the failed files in "failed_images.txt".

    Example
    -------
    >>> rootpath = "/path/to/directory"
    >>> filetype = "*.tif"
    >>> modelpath = "/path/to/pretrained_model.h5"
    >>> loss_function = "BCE"
    >>> spacing = "5 cm"
    >>> muscle = "Quadriceps"
    >>> scaling = "Bar"
    >>> volume_wanted = "Yes"
    >>> distance_acsa = 2.0
    >>> gui = ...  # Your Tkinter root window
    >>> calculate_batch(rootpath, filetype, modelpath, loss_function, spacing, muscle, scaling, volume_wanted, distance_acsa, gui)
    # The function will process the ultrasound images in the specified directory, predict the aponeurosis areas,
    # calculate echo intensity, and optionally calculate muscle volume based on the predicted areas.
    # The results will be compiled into a DataFrame and saved to an Excel file.
    """
    list_of_files = glob.glob(rootpath + filetype, recursive=True)

    if len(list_of_files) == 0:
        tk.messagebox.showerror(
            "Information",
            "No image files found." + "\nPotential error source: Unmatched filetype",
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    apo_model = ApoModel(gui, model_path=modelpath, loss_function=loss_function)

    dataframe = pd.DataFrame(
        columns=[
            "File",
            "Muscle",
            "Area_cm2",
            "Echo_intensity",
            "A_C_ratio",
            "Circumference",
            "Volume_cm3",
        ]
    )
    failed_files = []

    calculated_areas = []

    with PdfPages(rootpath + "/Analyzed_images.pdf") as pdf:

        try:
            start_time = time.time()

            for imagepath in list_of_files:
                if gui.should_stop:
                    # there was an input to stop the calculations
                    break

                # load image
                imported = import_image(imagepath)
                filename, img, nonflipped_img, height, width = imported

                if scaling == "Bar":
                    calibrate_fn = calibrate_distance_static
                    # find length of the scaling line
                    calib_dist, imgscale, scale_statement = calibrate_fn(
                        nonflipped_img, spacing
                    )
                    # check for StaticScalingError
                    if calib_dist is None:
                        fail = f"Scalingbars not found in {imagepath}"
                        failed_files.append(fail)
                        warnings.warn("Image fails with StaticScalingError")
                        continue

                    # predict area on image
                    circum, pred_apo_t, fig = apo_model.predict_s(
                        gui, img, imgscale, filename, scale_statement, width, height
                    )

                else:
                    calibrate_fn = calibrate_distance_manually
                    calib_dist = calibrate_fn(nonflipped_img, spacing)

                    # predict area on image
                    circum, pred_apo_t, fig = apo_model.predict_m(
                        gui, img, width, filename, height
                    )

                # calculate echo intensity and area
                echo = calculate_echo_int(nonflipped_img, pred_apo_t)
                if echo is None:
                    warnings.warn("Image fails with EchoIntensityError")
                    continue
                area = calc_area(calib_dist, pred_apo_t)
                area_circum_ratio = (area / circum) * 100

                # add the area to the list for volume calculation
                calculated_areas.append(area)

                # append results to dataframe
                dataframe = dataframe.append(
                    {
                        "File": filename,
                        "Muscle": muscle,
                        "Area_cm2": area,
                        "Echo_intensity": echo,
                        "A_C_ratio": area_circum_ratio,
                        "Circumference": circum,
                        "Volume_cm3": "",
                    },
                    ignore_index=True,
                )

                # save figures
                pdf.savefig(fig)
                plt.close(fig)
                # time duration of analysis of single image
                duration = time.time() - start_time
                print(f"duration: {duration}")

            if volume_wanted == "Yes":

                # musclevolume calculation with the calculated areas
                muscle_volume = muscle_volume_calculation(
                    calculated_areas, distance_acsa
                )

                # append musclevolume result to dataframe
                dataframe = dataframe.append(
                    {
                        "File": "",
                        "Muscle": "",
                        "Area_cm2": "",
                        "Echo_intensity": "",
                        "A_C_ratio": "",
                        "Circumference": "",
                        "Volume_cm3": muscle_volume,
                    },
                    ignore_index=True,
                )
            else:
                pass

        except ValueError:
            tk.messagebox.showerror(
                "Information",
                "Scaling Type Error."
                + "\nPotential error source: Selected scaling type does not fit image",
            )
            gui.do_break()
            gui.should_stop = False
            gui.is_running = False

        finally:
            # save predicted area results
            compile_save_results(rootpath, dataframe)
            # write failed images in file
            if len(failed_files) >= 1:
                file = open(rootpath + "/failed_images.txt", "w")
                for fail in failed_files:
                    file.write(fail + "\n")
                file.close()
            # clean up
            gui.should_stop = False
            gui.is_running = False
