"""
Description
-----------
Python module which contains a function, which allows to generate new training images
from the input images. The newly generated data will be saved under the same directories
as the input data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
)


def image_augmentation(input_img_folder, input_mask_folder):
    """
    Function, which generates new training data from the input images through data augmentation.
    At the moment the number of added images is set to five.

    Parameters
    ----------
    input_img_folder: str
        String, which contains the folder path of the input images.
        The newly generated images will be added to this folder.
    input_mask_folder: str
        String, which containd the folder path of the input masks.
        The newly generated masks will be added to this folder.

    Returns
    -------
    None

    Notes
    -----
    Although nothing is returned, the images in the input folders will be augmented three-fold.
    """

    # Creating image augmentation function
    gen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=5,
        width_shift_range=0.075,
        height_shift_range=0.075,
        zoom_range=0.075,
        horizontal_flip=True,
    )

    ids = os.listdir(input_mask_folder)
    seed = 131313
    batch_size = 1
    num_aug_images = 3  # Number of images added from augmented dataset.

    for i in range(int(len(ids))):

        # Choose image & mask that should be augmented
        # Directory structur: "root/some_dorectory"
        chosen_image = ids[i]
        image_path = input_img_folder + "/" + chosen_image
        mask_path = input_mask_folder + "/" + chosen_image
        image = np.expand_dims(
            plt.imread(image_path), 0
        )  # Read and expand image dimensions
        if image.ndim < 4:
            image = np.expand_dims(image, -1)
        mask = np.expand_dims(plt.imread(mask_path), 0)
        if mask.ndim < 4:
            mask = np.expand_dims(mask, -1)

        # Augment images
        aug_image = gen.flow(
            image,
            batch_size=batch_size,
            seed=seed,
            save_to_dir=input_img_folder,
            save_prefix=str(i),
            save_format="tif",
        )
        aug_mask = gen.flow(
            mask,
            batch_size=batch_size,
            seed=seed,
            save_to_dir=input_mask_folder,
            save_prefix=str(i),
            save_format="tif",
        )
        seed = seed + 1

        # Add images to folder
        for i in range(num_aug_images):
            next(aug_image)[0].astype(np.uint8)
            next(aug_mask)[0].astype(np.uint8)
