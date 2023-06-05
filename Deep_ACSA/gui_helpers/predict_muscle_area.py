"""Python module to automatically calcuate muscle area in US images"""

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

    Arguments:
        Scanning depth (cm),
        Scalingline length (pixel),
        thresholded binary model prediction.

    Returns:
        Predicted muscle area (cm²).

    Example:
        >>>calc_area(float(5), int(254), Image1.tif)
        3.813
    """
    pix_per_cm = scalingline_length / depth
    # Counts pixels with values != 0
    pred_muscle_area = cv2.countNonZero(img) / (pix_per_cm**2)
    return pred_muscle_area


def calc_area(calib_dist: float, img: np.ndarray):
    """Calculates predicted muscle aread.

    Arguments:
        Distance between scaling bars in pixel,
        thresholded binary model prediction.

    Returns:
        Predicted muscle area (cm²).

    Example:
    >>>calc_area(int(54), Image1.tif)
    3.813
    """
    pix_per_cm = calib_dist

    # Counts pixels with values != 0
    pred_muscle_area = cv2.countNonZero(img) / (pix_per_cm**2)
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

    Arguments:
        Path to root directory of images,
        type of image files,
        path to model used for predictions,
        loss_function
        ultrasound scanning depth,
        analyzed muscle.
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

    Arguments:
        Path to root directory of images,
        type of image files,
        path to txt file containing flipping information for images,
        path to model used for predictions,
        loss_function
        distance between (vertical) scaling lines (mm),
        analyzed muscle,
        scaling type.
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
