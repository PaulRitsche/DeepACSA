"""
Description
-----------
Automatic muscle area analysis for ultrasound images.

This module provides functions to import and preprocess ultrasound (US)
images, predict anatomical regions using a trained model, and compute
derived metrics such as muscle area and echo intensity. Optional muscle
volume estimation is supported using the predicted areas.

Multiple calibration methods are available, including continuous scaling
lines for EFOV images as well as scaling bars, manual calibration, or no
scaling for standard ultrasound images. Results are collected in a
Pandas DataFrame and exported to an Excel file. The module is intended
to support batch analysis of ultrasound images.
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

from DeepACSA.gui_helpers.apo_model import ApoModel
from DeepACSA.gui_helpers.calculate_muscle_volume import muscle_volume_calculation
from DeepACSA.gui_helpers.calibrate import (
    calibrate_distance_efov,
    calibrate_distance_manually,
    calibrate_distance_static,
)
from DeepACSA.gui_helpers.echo_int import calculate_echo_int

# from keras.preprocessing.image import img_to_array, load_img


plt.style.use("ggplot")
plt.switch_backend("agg")


def get_list_of_files(pathname: str):
    """Get a list of files matching a pathname pattern. 
    Parameters
    ----------
    pathname : str
        Glob-style pathname pattern to match against file paths.   
    Returns
    -------
    list of str
        File paths that match `pathname`.  
    Examples
    --------
    >>> get_list_of_files("C:/Desktop/Test/*")
    ['C:/Desktop/Test/Img1.tif', 'C:/Desktop/Test/Img2.tif',
     'C:/Desktop/Test/Flip.txt']
    """
    return glob.glob(pathname)


def import_image_efov(path_to_image: str):
    """Import and preprocess an eFOV ultrasound image for model inference.

    The function reads an image, removes fixed margins, keeps an unmodified
    copy of the cropped image, and creates a normalized model input tensor.

    Parameters
    ----------
    path_to_image : str
        Path to the eFOV ultrasound image.

    Returns
    -------
    filename : str
        Image filename without extension.
    img_copy : numpy.ndarray
        Cropped image copy (after margin removal), shape (H, W, 3).
    img : numpy.ndarray
        Normalized model input tensor, shape (1, 256, 256, 3), dtype float32.
    height : int
        Height of the cropped image (`img_copy.shape[0]`).
    width : int
        Width of the cropped image (`img_copy.shape[1]`).

    Notes
    -----
    The image is read using OpenCV and cropped as follows:

    - 75 pixels removed from the top
    - 20 pixels removed from the left
    - 10 pixels removed from the right

    The cropped image is resized to 256x256, converted to float32, and
    normalized to [0, 1].

    Examples
    --------
    >>> filename, img_copy, img, height, width = import_image_efov("C:/Desktop/Test/Img1.tif")
    """
    image_add = path_to_image
    filename = os.path.splitext(os.path.basename(image_add))[0]
    img = cv2.imread(path_to_image, 1)
    rows, cols, channels = img.shape
    img = img[75:rows, 20 : cols - 10]
    img_copy = img.copy()

    height = img.shape[0]
    width = img.shape[1]
    
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return filename, img_copy, img, height, width


def import_image(path_to_image: str, modelpath: str):
    """Import and preprocess an ultrasound image for model inference.

    The image is loaded in color, preserved in its original orientation, and
    resized/normalized to match the expected input size of the selected model.

    Parameters
    ----------
    path_to_image : str
        Path to the ultrasound image.
    modelpath : str
        Path to the model file. The path string is used to infer the expected
        input size (e.g., SwinUNet vs. other models).

    Returns
    -------
    filename : str
        Image filename without extension.
    img_resized : numpy.ndarray
        Normalized model input tensor, shape (1, H, W, 3), dtype float32.
        The spatial size depends on `modelpath`:
        - (224, 224) if `"swinunet"` is found in `modelpath` (case-insensitive)
        - (256, 256) otherwise
    nonflipped_img : numpy.ndarray
        Original loaded image (unmodified), shape (H0, W0, 3).
    original_height : int
        Height of the original image (`nonflipped_img.shape[0]`).
    original_width : int
        Width of the original image (`nonflipped_img.shape[1]`).

    Raises
    ------
    FileNotFoundError
        If OpenCV fails to load the image (i.e., `cv2.imread` returns None).

    Notes
    -----
    The resized image is converted to float32, normalized to [0, 1], and
    expanded to include a batch dimension.

    Examples
    --------
    >>> filename, x, img0, h, w = import_image("C:/Desktop/Test/Img1.tif", "C:/Desktop/Test/model.h5")
    """
    filename = os.path.splitext(os.path.basename(path_to_image))[0]
    img = cv2.imread(path_to_image, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"[ERROR] Failed to load image: {path_to_image}")

    nonflipped_img = img.copy()
    original_height, original_width = img.shape[:2]

    # Determine input size based on model type
    model_path_lower = modelpath.lower()
    if "swinunet" in model_path_lower:
        target_size = (224, 224)
    else:  # fallback to VGG/UNet
        target_size = (256, 256)

    # Resize for model input
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    return filename, img_resized, nonflipped_img, original_height, original_width

    # return filename, img, nonflipped_img, height, weight


def calc_area_efov(depth: float, scalingline_length: int, img: np.ndarray):
    """Compute predicted muscle area for an eFOV mask using a continuous scale line.

    Parameters
    ----------
    depth : float
        Depth represented by the scaling line, in centimeters.
    scalingline_length : int
        Length of the scaling line, in pixels.
    img : numpy.ndarray
        Binary (or non-zero) mask image of the predicted region of interest.
        Non-zero pixels are treated as belonging to the region.

    Returns
    -------
    float
        Predicted muscle area in square centimeters.

    Notes
    -----
    Pixels-per-centimeter is computed as::

        pix_per_cm = scalingline_length / depth

    Area is then computed from the count of non-zero mask pixels::

        area_cm2 = count_nonzero / (pix_per_cm ** 2)

    Examples
    --------
    >>> calc_area_efov(5.0, 254, pred_mask)
    3.813
    """
    pix_per_cm = scalingline_length / depth
    # Counts pixels with values != 0
    pred_muscle_area = cv2.countNonZero(img) / (pix_per_cm**2)
    return pred_muscle_area


def calc_area(calib_dist: float, img: np.ndarray):
    """Compute predicted muscle area for a mask using a calibration factor.

    Parameters
    ----------
    calib_dist : float
        Calibration factor in pixels per centimeter.
    img : numpy.ndarray
        Binary (or non-zero) mask image of the predicted region of interest.
        Non-zero pixels are treated as belonging to the region.

    Returns
    -------
    float
        Predicted muscle area in square centimeters.

    Notes
    -----
    Area is computed from the count of non-zero pixels as::

        area_cm2 = count_nonzero / (calib_dist ** 2)

    Examples
    --------
    >>> calc_area(54, pred_mask)
    3.813
    """
    pix_per_cm = calib_dist

    # Counts pixels with values != 0
    pred_muscle_area = cv2.countNonZero(img) / (pix_per_cm**2)
    return pred_muscle_area


def compile_save_results(rootpath: str, dataframe: pd.DataFrame):
    """Save analysis results to an Excel file.

    Parameters
    ----------
    rootpath : str
        Directory where the results file is written.
    dataframe : pandas.DataFrame
        Table of results to write.

    Returns
    -------
    None

    Notes
    -----
    The output file is written to ``{rootpath}/Results.xlsx`` and the
    DataFrame is saved to the ``Results`` sheet.

    Examples
    --------
    >>> compile_save_results("C:/Desktop/Test", df)
    """
    excelpath = rootpath + "/Results.xlsx"
    with pd.ExcelWriter(excelpath, mode="w") as writer:
        data = dataframe
        data.to_excel(writer, sheet_name="Results")


def calculate_batch_efov(
    rootpath: str,
    modelpath: str,
    depth: float,
    muscle: str,
    volume_wanted: str,
    distance_acsa: float,
    gui,
):
    """Run batch muscle area analysis for eFOV images with a continuous scale line.

    This function searches `rootpath` for supported image types, performs scale
    calibration via a continuous scaling line, predicts the region of interest
    using a trained model, computes muscle area and echo intensity, and writes
    results to disk. Optionally, it estimates muscle volume from the sequence of
    predicted areas.

    Parameters
    ----------
    rootpath : str
        Directory containing the eFOV ultrasound images.
    modelpath : str
        Path to the trained model used for prediction.
    depth : float
        Depth represented by the eFOV scaling line, in centimeters.
    muscle : str
        Muscle name/type used for analysis and calibration.
    volume_wanted : str
        Whether to compute volume. Expected values are `"Yes"` or `"No"`.
    distance_acsa : float
        Distance between adjacent ACSA locations, in centimeters, used for
        volume estimation.
    gui : object
        GUI controller used for progress updates and stop handling. Expected to
        expose attributes/methods used in this function (e.g., `should_stop`,
        `progress_var`, `do_break`).

    Returns
    -------
    None

    Notes
    -----
    - Supported image extensions: ``.tif``, ``.tiff``, ``.jpeg``, ``.jpg``,
      ``.png``, ``.bmp``.
    - Results are saved as:
      - ``{rootpath}/Results.xlsx``
      - ``{rootpath}/Analyzed_images.pdf``
    - Failures are recorded in ``{rootpath}/failed_images.txt``.
    - Echo intensity is computed from the original image and the predicted mask.

    Warns
    -----
    UserWarning
        If scale line detection fails (ScalinglineError) or echo intensity
        calculation fails (EchoIntensityError), the image is skipped.

    Examples
    --------
    >>> calculate_batch_efov(
    ...     rootpath="/path/to/images",
    ...     modelpath="/path/to/model.h5",
    ...     depth=4.5,
    ...     muscle="Quadriceps",
    ...     volume_wanted="Yes",
    ...     distance_acsa=2.0,
    ...     gui=gui,
    ... )
    """
    # loop through acceptable image files
    filetypes = ["*.tif", "*.jpeg", "*.tiff", "*.jpg", "*.png", "*.bmp"]
    list_of_files = []
    for filetype in filetypes:
        list_of_files.extend(
            glob.glob(os.path.join(rootpath, filetype), recursive=True)
        )

    if len(list_of_files) == 0:
        tk.messagebox.showerror(
            "Information",
            "No image files found."
            + "\nPotential error source: Unmatched filetype"
            + "\nAcceptable filetypes are *.tif, *.jpeg, *.tiff, *.jpg, *.png, *.bmp",
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    apo_model = ApoModel(gui, model_path=modelpath)

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

            # Show progress bar and reset value
            gui.progress_label.grid()
            gui.progress_bar.grid()
            gui.progress_var.set(0)
            gui.progress_bar.update_idletasks()

            for i, imagepath in enumerate(list_of_files):

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

                # Update progress
                gui.progress_label.configure(
                    text=f"Predicting image {i+1}/{len(list_of_files)}"
                )
                progress = (i + 1) / len(list_of_files)
                gui.progress_var.set(progress)
                gui.progress_bar.update_idletasks()

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
            gui.progress_bar.grid_remove()
            gui.progress_label.grid_remove()


def calculate_batch(
    rootpath: str,
    modelpath: str,
    spacing: str,
    muscle: str,
    scaling: str,
    volume_wanted: str,
    distance_acsa: float,
    gui,
):
    """Run batch muscle area analysis for images without a continuous scale line.

    This function searches `rootpath` for supported image types, performs scale
    calibration using a scaling bar, manual calibration, or no scaling, predicts
    the region of interest using a trained model, computes muscle area and echo
    intensity, and writes results to disk. Optionally, it estimates muscle volume
    from the sequence of predicted areas.

    Parameters
    ----------
    rootpath : str
        Directory containing the ultrasound images.
    modelpath : str
        Path to the trained model used for prediction.
    spacing : str
        Spacing between reference markers on the image (used for scaling).
        Passed through to the calibration routines.
    muscle : str
        Muscle name/type used for analysis output.
    scaling : str
        Scaling mode. Expected values are `"Bar"`, `"Manual"`, or `"No Scaling"`.
    volume_wanted : str
        Whether to compute volume. Expected values are `"Yes"` or `"No"`.
    distance_acsa : float
        Distance between adjacent ACSA locations, in centimeters, used for
        volume estimation.
    gui : object
        GUI controller used for progress updates and stop handling. Expected to
        expose attributes/methods used in this function (e.g., `should_stop`,
        `progress_var`, `do_break`).

    Returns
    -------
    None

    Notes
    -----
    - Supported image extensions: ``.tif``, ``.tiff``, ``.jpeg``, ``.jpg``,
      ``.png``, ``.bmp``.
    - Results are saved as:
      - ``{rootpath}/Results.xlsx``
      - ``{rootpath}/Analyzed_images.pdf``
    - Failures are recorded in ``{rootpath}/failed_images.txt``.

    Warns
    -----
    UserWarning
        If static scaling bar detection fails (StaticScalingError) or echo
        intensity calculation fails (EchoIntensityError), the image is skipped.

    Examples
    --------
    >>> calculate_batch(
    ...     rootpath="/path/to/images",
    ...     modelpath="/path/to/model.h5",
    ...     spacing="5 cm",
    ...     muscle="Quadriceps",
    ...     scaling="Bar",
    ...     volume_wanted="Yes",
    ...     distance_acsa=2.0,
    ...     gui=gui,
    ... )
    """
    # loop through acceptable image files
    filetypes = ["*.tif", "*.jpeg", "*.tiff", "*.jpg", "*.png", "*.bmp"]
    list_of_files = []
    for filetype in filetypes:
        list_of_files.extend(
            glob.glob(os.path.join(rootpath, filetype), recursive=True)
        )

    if len(list_of_files) == 0:
        tk.messagebox.showerror(
            "Information",
            "No image files found."
            + "\nPotential error source: Unmatched filetype"
            + "\nAcceptable filetypes are *.tif, *.jpeg, *.tiff, *.jpg, *.png, *.bmp",
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    apo_model = ApoModel(gui, model_path=modelpath)

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

            # Show progress bar and reset value
            gui.progress_label.grid()
            gui.progress_bar.grid()
            gui.progress_var.set(0)
            gui.progress_bar.update_idletasks()

            for i, imagepath in enumerate(list_of_files):
                if gui.should_stop:
                    # there was an input to stop the calculations
                    break

                # load image
                imported = import_image(imagepath, modelpath)
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

                elif scaling == "Manual":
                    calibrate_fn = calibrate_distance_manually
                    calib_dist = calibrate_fn(nonflipped_img, spacing)

                    # predict area on image
                    circum, pred_apo_t, fig = apo_model.predict_m(
                        gui, img, width, filename, height
                    )

                elif scaling == "No Scaling":
                    calib_dist = 1
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

                # Update progress
                gui.progress_label.configure(
                    text=f"Predicting image {i+1}/{len(list_of_files)}"
                )
                progress = (i + 1) / len(list_of_files)
                gui.progress_var.set(progress)
                gui.progress_bar.update_idletasks()

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
            gui.progress_bar.grid_remove()
            gui.progress_label.grid_remove()
