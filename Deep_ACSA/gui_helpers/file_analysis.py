"""Module to perform analysis of image files used for training model"""

import os
import tkinter as tk
from tkinter import ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button


def show_outliers_popup(df, dir1, dir2):
    """
    Display a pop-up window with a table showing outlier images between two directories.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing outlier information.
    dir1 : str
        Path to the first directory containing images.
    dir2 : str
        Path to the second directory containing images.

    Returns
    -------
    None
    """
    popup = tk.Toplevel()
    popup.title("Outliers")
    master_path = os.path.dirname(os.path.abspath(__file__))
    iconpath = master_path + "/icon.ico"
    popup.iconbitmap(iconpath)

    label = tk.Label(popup, text=f"Comparing images in {dir1} \nand {dir2}")
    label.pack(pady=10)

    tree = ttk.Treeview(
        popup,
        columns=("Outlier Image", "Directory", "Images in Dir1", "Images in Dir2"),
    )
    tree.heading("#1", text="Outlier Image")
    tree.heading("#2", text="Directory")
    tree.heading("#3", text="Images in Dir1")
    tree.heading("#4", text="Images in Dir2")

    for index, row in df.iterrows():
        tree.insert(
            "",
            "end",
            values=(
                row["Outlier Image"],
                row["Folder"],
                row["Images in Dir1"],
                row["Images in Dir2"],
            ),
        )

    tree.pack()


def find_outliers(dir1, dir2):
    """
    Find image filenames that do not occur in both directories and
    check if both directories have the same number of images.

    Parameters
    ----------
    dir1 : str
        Path to the first directory containing images.
    dir2 : str
        Path to the second directory containing images.

    Returns
    -------
    DataFrame : pd.dataframe
        A DataFrame with columns:
        - 'Outlier Image': Names of the outlier images.
        - 'Directory': The directory in which the outlier was found.
        - 'Images in Dir1': The number of images in dir1.
        - 'Images in Dir2': The number of images in dir2.

    Examples
    --------
    >>> find_outliers('/path/to/dir1', '/path/to/dir2')
                  Outlier Image         Directory  Images in Dir1  Images in Dir2
    0           image3.jpg      /path/to/dir1            NaN                 NaN
    1           image4.jpg      /path/to/dir2            NaN                 NaN
    2           NaN                 NaN                     5                   4
    """

    # List all files in the directories
    try:
        files_dir1 = set(os.listdir(dir1))
        files_dir2 = set(os.listdir(dir2))
    except FileNotFoundError:
        tk.messagebox.showerror(
            "Information",
            "Select input directory that contains at least one image!"
            + "\nAccepted image types: .tif, .jpeg, .tiff, .jpg, .png, .bmp",
        )
        return

    # Check if both directories have the same number of images
    count_dir1 = len(files_dir1)
    count_dir2 = len(files_dir2)

    # Find outliers
    outliers_dir1 = [
        (f, os.path.basename(dir1), np.nan, np.nan) for f in files_dir1 - files_dir2
    ]
    outliers_dir2 = [
        (f, os.path.basename(dir2), np.nan, np.nan) for f in files_dir2 - files_dir1
    ]

    # If there are no outliers
    if not outliers_dir1 and not outliers_dir2:
        tk.messagebox.showinfo(
            "Information", "There are no different images in both directories."
        )
        return

    # Create DataFrame
    df = pd.DataFrame(
        outliers_dir1 + outliers_dir2,
        columns=["Outlier Image", "Folder", "Images in Dir1", "Images in Dir2"],
    )

    # Add summary row
    summary_row = {
        "Outlier Image": np.nan,
        "Folder": np.nan,
        "Images in Dir1": count_dir1,
        "Images in Dir2": count_dir2,
    }
    df = df.append(summary_row, ignore_index=True)

    show_outliers_popup(df, dir1, dir2)

    return df


def overlay_directory_images(image_dir, mask_dir, alpha=0.5, start_index=0):
    """
    Overlay binary masks on ultrasound images from given directories.

    Parameters
    ----------
    image_dir : str
        Directory containing the ultrasound images.
    mask_dir : str
        Directory containing the corresponding binary masks.
    alpha : float, optional
        Opacity level of the mask when overlaid on the ultrasound. Default is 0.5.
    start_index : int, optional
        Index to start displaying the image/mask pairs. Default is 0 (first image).

    Returns
    -------
    None
        Displays an interactive plot of overlaid image pairs.

    Examples
    --------
    >>> overlay_directory_images('/path/to/ultrasound_images/', '/path/to/masks/', start_index=2)
    """
    try:
        # Get list of image and mask filenames
        image_files = sorted(
            [
                f
                for f in os.listdir(image_dir)
                if f.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
            ]
        )
        mask_files = sorted(
            [
                f
                for f in os.listdir(mask_dir)
                if f.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
            ]
        )

    except FileNotFoundError:
        tk.messagebox.showerror(
            "Information",
            "Select input directory that contains at least one image!"
            + "\nAccepted image types: .tif, .jpeg, .tiff, .jpg, .png, .bmp",
        )
        return

    # Check if both directories have the same number of files
    if len(image_files) != len(mask_files):
        tk.messagebox.showerror(
            "Information", "Both directories must have the same number of images."
        )

    # Create a function to overlay a single image and mask
    def overlay_image(image_path, mask_path, alpha=0.5):
        ultrasound = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        colored_mask[mask == 255] = [0, 255, 0]  # Green color for mask
        ultrasound_colored = cv2.cvtColor(ultrasound, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(ultrasound_colored, 1, colored_mask, alpha, 0)

    # Create an interactive plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20 / 2.45, 15 / 2.54)
    fig.set_facecolor("#7ABAA1")
    current_idx = start_index

    # Function to handle delete button click
    def delete_image(event):
        nonlocal current_idx  # This line ensures we're referencing the outer variable
        image_path = os.path.join(image_dir, image_files[current_idx])
        mask_path = os.path.join(mask_dir, mask_files[current_idx])

        os.remove(image_path)
        os.remove(mask_path)

        print(f"Deleted {image_files[current_idx]} and its corresponding mask.")

        # Remove deleted filenames from lists
        del image_files[current_idx]
        del mask_files[current_idx]

        # Check if current index is out of bounds after deletion
        if current_idx >= len(image_files):
            current_idx -= 1

        # Display the next (or previous) image
        display_current_image()

    def display_current_image():
        try:
            overlaid = overlay_image(
                os.path.join(image_dir, image_files[current_idx]),
                os.path.join(mask_dir, mask_files[current_idx]),
                alpha,
            )
            ax.imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))
            ax.set_title(
                f"Image: {image_files[current_idx]}"
                + "\nClick right/left arrow to navigate through images"
                + "\nClick Delete to delete the image-mask pair."
            )
            ax.axis("off")
            plt.draw()
        except IndexError:
            tk.messagebox.showerror(
                "Information", f"Start index must be between 0 and {len(image_files)}"
            )
            return

    def on_key(event):
        nonlocal current_idx
        if event.key == "right":
            current_idx = (current_idx + 1) % len(image_files)
            display_current_image()
        elif event.key == "left":
            current_idx = (current_idx - 1) % len(image_files)
            display_current_image()

    fig.canvas.mpl_connect("key_press_event", on_key)
    # Add a delete button to the plot
    ax_delete = plt.axes([0.8, 0.05, 0.1, 0.075])
    btn_delete = Button(ax_delete, "Delete")
    btn_delete.on_clicked(delete_image)

    display_current_image()
    plt.show()
