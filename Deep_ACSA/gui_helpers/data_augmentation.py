import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

img_aug_prefix = ""

def image_augmentation(input_img_folder, input_mask_folder):

    # Creating image augmentation function
    gen = ImageDataGenerator(featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=5, 
                            width_shift_range=0.075, 
                            height_shift_range=0.075,
                            zoom_range=0.075,
                            horizontal_flip=True)

    ids = os.listdir(input_mask_folder)
    seed = 131313
    batch_size = 1
    num_aug_images = 5 # Number of images added from augmented dataset. 


    for i in range(int(len(ids) + 1)):
        
        # Choose image & mask that should be augmented 
        # Directory structur: "root/some_dorectory/"
        chosen_image = ids[i] 
        image_path = input_img_folder + "/" + chosen_image 
        mask_path = input_mask_folder + "/" + chosen_image
        image = np.expand_dims(plt.imread(image_path),0)# Read and expand image dimensions
        if image.ndim < 4: 
            image = np.expand_dims(image,-1)
        mask = np.expand_dims(plt.imread(mask_path),0)
        if mask.ndim < 4: 
            mask = np.expand_dims(mask,-1)

        # Augment images 
        aug_image = gen.flow(image, batch_size=batch_size, seed=seed, save_to_dir=input_img_folder, save_prefix=img_aug_prefix+str(i), save_format="tif")
        aug_mask = gen.flow(mask, batch_size=batch_size, seed=seed, save_to_dir=input_mask_folder, save_prefix=img_aug_prefix+str(i), save_format="tif")
        seed = seed + 1 
         
        # Add images to folder
        for i in range(num_aug_images):
            next(aug_image)[0].astype(np.uint8)
            next(aug_mask)[0].astype(np.uint8)
            