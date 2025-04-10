"""Module to perform analysis of image files used for training model"""

import os
import tkinter as tk
from tkinter import ttk
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

import cv2
import matplotlib
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
    iconpath = master_path + "/home_im.ico"
    # popup.iconbitmap(iconpath)

    label = tk.Label(popup, text=f"Comparing images in {dir1} \nand {dir2}")
    label.pack(pady=10)

    # Create the Treeview widget with displaycolumns attribute
    tree = ttk.Treeview(
        popup,
        columns=(
            "Outlier Image Name",
            "Directory",
            "Num. Images in Dir1",
            "Num. Images in Dir2",
        ),
        displaycolumns=(
            "Outlier Image Name",
            "Directory",
            "Num. Images in Dir1",
            "Num. Images in Dir2",
        ),
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
        """
        Overlay a mask on top of an ultrasound image.

        Parameters
        ----------
        image_path : str
            Path to the ultrasound image.
        mask_path : str
            Path to the mask image.
        alpha : float, optional
            Alpha value for the overlay (transparency), default is 0.5.

        Returns
        -------
        overlaid_image : np.ndarray
            The ultrasound image overlaid with the mask.

        Notes
        -----
        - The ultrasound and mask images should have the same dimensions.
        - The mask is displayed in green on top of the ultrasound image.

        Examples
        --------
        >>> overlaid = overlay_image("ultrasound.png", "mask.png", alpha=0.5)
        >>> cv2.imshow("Overlay", overlaid)
        >>> cv2.waitKey(0)
        >>> cv2.destroyAllWindows()
        """
        # Read the ultrasound and mask images in grayscale
        ultrasound = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Ensure both images have the same dimensions by resizing the mask
        ultrasound = cv2.resize(ultrasound, (mask.shape[1], mask.shape[0]))

        # Create a colored mask with green color for the white regions in the mask
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        colored_mask[mask > 0] = [0, 255, 0]  # Green color for mask regions

        # Convert the ultrasound image to color
        ultrasound_colored = cv2.cvtColor(ultrasound, cv2.COLOR_GRAY2BGR)

        # Overlay the colored mask on the ultrasound image
        overlaid_image = cv2.addWeighted(ultrasound_colored, 1, colored_mask, alpha, 0)

        return overlaid_image

    matplotlib.use("TkAgg")

    # Create an interactive plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20 / 2.45, 15 / 2.54)
    fig.set_facecolor("#2A484E")
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
        """
        Display the current ultrasound image overlaid with its mask.

        This function updates the plot with the current ultrasound image overlaid with its mask.
        It also sets the title and defines navigation instructions.

        Notes
        -----
        - Assumes `ax`, `image_files`, `mask_files`, `current_idx`, and `alpha` are defined externally.
        - Handles navigation between images using right and left arrow keys.
        - Displays the image title with navigation instructions.
        - Handles exceptions for invalid indices and displays an error message.

        Examples
        --------
        To display the current image, call this function within your application:

        >>> display_current_image()
        """
        try:
            # Overlay the current image with its mask
            overlaid = overlay_image(
                os.path.join(image_dir, image_files[current_idx]),
                os.path.join(mask_dir, mask_files[current_idx]),
                alpha,
            )

            # Display the overlaid image on the plot
            ax.imshow(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB))

            # Set the title with navigation instructions
            ax.set_title(
                f"Image: {image_files[current_idx]}"
                + "\nClick right/left arrow to navigate through images"
                + "\nClick Delete to delete the image-mask pair.",
                color="white",
            )

            # Turn off axis display
            ax.axis("off")

            # Redraw the plot
            plt.draw()
        except IndexError:
            tk.messagebox.showerror(
                "Information", f"Start index must be between 0 and {len(image_files)}"
            )
            return

    def on_key(event):
        """
        Handle keypress events for image navigation.

        This function is used to navigate through images using right and left arrow keys.
        It updates the `current_idx` variable and calls `display_current_image()` to show the new image.

        Parameters
        ----------
        event : mpl.backend_bases.KeyEvent
            The key event triggered by pressing a key.

        Notes
        -----
        - Assumes `current_idx` and `image_files` are defined externally.
        - Handles right and left arrow keys to navigate images in a loop.

        Examples
        --------
        To handle keypress events for image navigation, connect this function to your plot:

        >>> fig.canvas.mpl_connect("key_press_event", on_key)
        """
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


def collect_images(root_dir, target_dir, image_type):
    """
    Searches through the root directory and its subdirectories for images of the
    specified type and copies them to the target directory with modified names to
    avoid overwriting. The modified name format is original_filename_n.extension, where
    n starts from 0.

    Parameters
    ----------
    root_dir : str
        The root directory to search for images.
    target_dir : str
        The directory where found images will be stored.
    image_type : str
        The type of the images to search for (e.g., 'jpg', 'png', 'tiff').

    Notes
    -----
    The function creates the target directory if it does not already exist.
    """
    # Ensure target directory exists
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    file_counter = {}

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("." + image_type):
                # Check if the file name is already in our counter dictionary
                if file in file_counter:
                    # Increment the counter for that file name
                    file_counter[file] += 1
                else:
                    # Initialize counter for this file name
                    file_counter[file] = 0

                src_path = os.path.join(subdir, file)
                # Modify the target file name to include the counter
                file_name, file_ext = os.path.splitext(file)
                target_file_name = f"{file_name}_{file_counter[file]}{file_ext}"
                target_path = os.path.join(target_dir, target_file_name)

                shutil.copy(src_path, target_path)


def redact_images_in_directory(directory):
    """
    Redacts the upper 50 pixels of every image in the specified directory by
    drawing a black rectangle over them. The images are saved under the same
    names in the same directory.

    Parameters
    ----------
    directory : str
        The path to the directory containing the images to be redacted.
    """
    for filename in os.listdir(directory):
        # Check if the file is an image based on its extension.
        # You might want to adjust the list of extensions based on your needs.
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".tif")
        ):
            try:
                filepath = os.path.join(directory, filename)
                with Image.open(filepath) as img:
                    # Use ImageDraw to draw a black rectangle over the top 50 pixels
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([0, 0, img.width, 60], fill="black")

                    # Save the image, overwriting the original file
                    img.save(filepath)
                print(f"Processed {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
