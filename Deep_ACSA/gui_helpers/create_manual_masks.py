import os
from tkinter.messagebox import WARNING, askokcancel, showerror, showinfo

import cv2
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd

from Deep_ACSA.gui_helpers.calibrate import calibrate_distance_manually
from Deep_ACSA.gui_helpers.predict_muscle_area import compile_save_results


def select_area(image):
    """
    Allow user to interactively select an area on the image using matplotlib.
    Returns a binary mask of the selected area or None if the user cancels.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image on which the area is to be selected.

    Returns
    -------
    mask : np.ndarray or None
        Binary mask of the selected area or None if selection is cancelled.
    """
    # scale the image
    try:
        showinfo(
            "Information",
            "Scale the image before creating a mask."
            + "\nClick on two scaling bars that are EXACTLY 1 CM APART."
            + "\nClick 'q' to continue.",
        )
        calib_dist = calibrate_distance_manually(image, spacing="1")
    except IndexError:
        calib_dist = 1

    # Plot image
    fig, ax = plt.subplots()
    fig.set_size_inches(20 / 2.45, 15 / 2.54)

    ax.imshow(image, cmap="gray")
    ax.set_title(
        "Click on aponeurosis to set marker."
        + "\nClick 'Enter' to confirm, 'ESC' to cancel analysis."
        + "\nUse 'Wheel' to zoom and 'Right Mouse' to delete."
    )
    ax.grid(False)

    coords = []
    (points_plot,) = ax.plot(
        [], [], "ro", markersize=4, alpha=0.5
    )  # Placeholder for points
    cancelled = False

    # Capture click positions and plot them with adjusted dot size and opacity
    def onclick(event):
        """
        Function to detect click.
        """
        # If left mouse button is clicked, add a point
        if event.button == 1:
            ix, iy = event.xdata, event.ydata
            coords.append((ix, iy))
            points_plot.set_data(zip(*coords))  # Update the plotted data
            fig.canvas.draw()

        # If right mouse button is clicked, remove the last point
        elif event.button == 3 and coords:
            try:
                coords.pop()
                if len(coords) >= 1:
                    points_plot.set_data(zip(*coords))  # Update the plotted data
                fig.canvas.draw()
            except IndexError:
                showerror("Error", "Select at least one point prior to removal!")
                return None, None

    # Handle key events
    def on_key(event):
        """
        Function to detect key stroke on keyboard
        """
        nonlocal cancelled
        if event.key == "enter" or event.key == "return":
            plt.close()
        elif event.key == "escape":
            cancelled = True
            plt.close()

    # Handle zoom with mouse wheel
    def zoom(event):
        """
        Function to zoom in on the plot.
        """
        try:
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            zoom_factor = 1.2 if event.button == "up" else 1 / 1.2
            xdata, ydata = event.xdata, event.ydata
            new_xlim = [
                xdata - (xdata - cur_xlim[0]) / zoom_factor,
                xdata + (cur_xlim[1] - xdata) / zoom_factor,
            ]
            new_ylim = [
                ydata - (ydata - cur_ylim[0]) / zoom_factor,
                ydata + (cur_ylim[1] - ydata) / zoom_factor,
            ]
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            fig.canvas.draw()

        except TypeError:
            pass

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", zoom)

    plt.show()

    if cancelled:
        return None, None

    # create Mask
    mask = np.zeros_like(image)
    if len(coords) > 2:
        pts = np.array(coords, np.int32)
        pts = pts.reshape((-1, 1, 2))
        area = cv2.fillPoly(mask, [pts], 255)

        # calculate muscle area
        muscle_area = cv2.countNonZero(area) / (calib_dist**2)
        print(muscle_area)
        return mask, muscle_area
    else:
        showinfo("Information", "Select a minimum of two points!")
        return None, None


def create_acsa_masks(
    input_dir,
    muscle_name: str = "bf_img_",
    output_imgs_dir: str = None,
    output_masks_dir: str = None,
):
    """
    Create masks for images in the specified directory by allowing the user to
    interactively select areas of interest.

    Parameters
    ----------
    input_dir : str
        Directory containing the input images.
    muscle_name : str, optional
        Prefix for the output filenames. Default is "rf_img_".
    output_imgs_dir: str, optional
        Directory to save the output images. If not provided, a 'train_images' folder will be created inside input_dir.
    output_masks_dir: str, optional
        Directory to save the output masks. If not provided, a 'train_masks' folder will be created inside input_dir.

    Returns
    -------
    str
        Message indicating the completion status.

    Examples
    --------
    >>> create_acsa_masks('/path/to/images')
    """
    # Define input images
    try:
        ext = [".tif", ".jpeg", ".tiff", ".jpg", ".png", ".bmp"]
        image_files = [f for f in os.listdir(input_dir) if f.endswith(tuple(ext))]
    except FileNotFoundError:
        showinfo(
            "Information",
            "Select input directory that contains at least one image!"
            + "\nAccepted image types: .tif, .jpeg, .tiff, .jpg, .png, .bmp",
        )
        return

    # Set default output directories if not provided
    if not output_imgs_dir:
        output_imgs_dir = os.path.join(input_dir, "train_images")
    if not output_masks_dir:
        output_masks_dir = os.path.join(input_dir, "train_masks")

    # Create output directories if they don't exist
    for directory in [output_imgs_dir, output_masks_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Determine the starting index based on existing files in the output directory
    existing_indices = [
        int(f.replace(muscle_name, "").replace(".tif", ""))
        for f in os.listdir(output_imgs_dir)
        if f.startswith(muscle_name)
    ]
    if existing_indices:
        start_idx = max(existing_indices) + 1
    else:
        start_idx = 0

    # Define DF for analysis reports
    output_df = pd.DataFrame({"image": [], "muscle_area (cm2)": []})

    for idx, image_file in enumerate(image_files, start=start_idx):
        try:
            image_path = os.path.join(input_dir, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            mask, muscle_area = select_area(image)

            if mask is None:  # User cancelled the selection
                if askokcancel(
                    "Information", "Do you really want to cancel?", icon=WARNING
                ):

                    # Save analysis of all images
                    compile_save_results(rootpath=output_imgs_dir, dataframe=output_df)
                    return

                else:
                    mask, muscle_area = select_area(image)

            # Save the images and masks using the determined index
            save_path_img = os.path.join(output_imgs_dir, f"{muscle_name}{idx}.tif")
            save_path_mask = os.path.join(output_masks_dir, f"{muscle_name}{idx}.tif")
            cv2.imwrite(save_path_img, image)
            cv2.imwrite(save_path_mask, mask)

            # Save results
            temp_df = pd.DataFrame(
                {"image": [image_file], "muscle_area (cm2)": [muscle_area]}
            )
            output_df = pd.concat([output_df, temp_df])

        except cv2.error:
            return "No Image Saved."

    # Save analysis of all images
    compile_save_results(rootpath=output_imgs_dir, dataframe=output_df)

    return "Processing complete."
