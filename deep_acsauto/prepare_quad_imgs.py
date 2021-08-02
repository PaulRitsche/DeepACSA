"""Python module to prepare whole quadriceps area images for DeepACSA analysis"""

from PIL import Image, ImageOps
import glob 
import cv2
import os
import numpy as np

def prepare_quad_vl_imgs(rootpath: str, filetype: str, output: str):
    """
    Function to crop whole quadriceps images to be used in DeepACSA.
    Images can be nested to up to two subdirectories.
    Arguments: Rootpath of folder with images to be cropped, 
               type of image files (tiff, png, bmp...), 
               output directory for images, 
               name of outputted images
    Returns: Cropped and flipped images in output directory. 
    Example: 
    >>> prepare_quad_images("C:/User/Desktop/Imgs", "/**/*.png",
                            "C:/User/Desktop/Imgs/prep_imgs")
    
    """

    # Get list of images
    list_of_files = glob.glob(rootpath + filetype, recursive=True)

    # Loop trough images
    for imagepath in list_of_files:
        # Import
        img = cv2.imread(imagepath,0)
        # Get filename
        filename = os.path.splitext(os.path.basename(imagepath))[0]
        rows,cols = img.shape
        # Rotate image
        rot_M = cv2.getRotationMatrix2D(((cols - 1) /2.0, (rows - 1) / 2.0), -45, 0.9)
        img_rot =  cv2.warpAffine(img, rot_M, (cols,rows))
        # Translate image
        trans_M = np.float32([[1, 0, -35], [0, 1, 50]])
        img_trans = cv2.warpAffine(img_rot, trans_M, (cols,rows))
        img_flip = cv2.flip(img_trans, 1)
        # Crop image
        img_crop = img_flip[150:, 200:700]
        #cv2.imshow("img_crop", img_crop)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # Save image
        cv2.imwrite(output + "/" + filename + "_vl.tif", img_crop)


def prepare_quad_rf_imgs(rootpath: str, filetype: str, output: str):
    """
    Function to crop whole quadriceps images to be used in DeepACSA.
    Images can be nested to up to two subdirectories.
    Arguments: Rootpath of folder with images to be cropped, 
               type of image files (tiff, png, bmp...), 
               output directory for images, 
               name of outputted images
    Returns: Cropped and flipped images in output directory. 
    Example: 
    >>> prepare_quad_images("C:/User/Desktop/Imgs", "/**/*.bmp", 
                            "C:/User/Desktop/Imgs/prep_imgs")
    
    """

    # Get list of images
    list_of_files = glob.glob(rootpath + filetype, recursive=True)

    # Loop trough images
    for imagepath in list_of_files:
        # Import image
        img = cv2.imread(imagepath,0)
        # Get filename
        filename = os.path.splitext(os.path.basename(imagepath))[0]
        rows,cols = img.shape
        # Rotate image
        rot_M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),-10, 0.9)
        img_rot =  cv2.warpAffine(img, rot_M, (cols,rows))
        # Translate image
        trans_M = np.float32([[1, 0, -35], [0, 1, 50]])
        img_trans = cv2.warpAffine(img_rot, trans_M, (cols,rows))
        # Crop image
        img_crop = img_trans[150:500, 200:720]
        #cv2.imshow("img_crop", img_crop)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # Save image
        cv2.imwrite(output + "/" + filename + "_rf.tif", img_crop)
