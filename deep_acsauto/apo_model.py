import numpy as np
import os
from skimage.transform import resize
from skimage import morphology
from keras import backend as K
from keras.models import load_model  # Model
import tensorflow as tf

import matplotlib.pyplot as plt
plt.style.use("ggplot")


def _resize(img, width: int, height: int):
    """Resizes an image to height x width.

    Args:
        Image to be resized,
        Target width,
        Target height,
    Returns:
        The resized image.

    """
    img = resize(img, (1, height, width, 1))
    img = np.reshape(img, (height, width))
    return img


def IoU(y_true, y_pred, smooth=1):
    """Computes intersection over union (IoU), a measure of labelling accuracy.

    Arguments:
        The ground-truth bit-mask,
        The predicted bit-mask,
        A smoothing parameter,

    Returns:
        Intersection over union scores.

    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


class ApoModel:
    """Class which provides utility to predict aponeurosis on US-images.

    Attributes:
        model_path: Path to the Keras segmentation model.
        apo_threshold: Pixels above this threshold are assumed to be apo.


    Examples:
        >>> apo_model = ApoModel('path/to/model.h5')
        >>> filename, img, nonflipped_img, height, width = import_image(path)
        >>>
        >>> # get predictions only
        >>> pred_apo = apo_model.predict(img)
        >>> pred_apo_t = apo_model.predict_t(img, width, height, False)
        >>>
        >>> # get predictions and plot (the following two are identical)
        >>> pred_apo_t, fig = apo_model.predict_t(img, width, height)
        >>> pred_apo_t, fig = apo_model.predict_t(img, width, height, True)

    """

    def __init__(self, model_path: str, apo_threshold: float = 0.5):
        self.model_path = model_path
        self.model_apo = load_model(
            self.model_path,
            custom_objects={'IoU': IoU}
        )
        self.apo_threshold = apo_threshold

    def predict(self, img):
        """Runs a segmentation model on the input image.

        Arguments:
            Input image

        Returns:
            The probability for each pixel, that it belongs to the foreground.

        """
        pred_apo = self.model_apo.predict(img)
        return pred_apo

    def predict_e(self, img, img_lines, width: int,
                  height: int, return_fig: bool = True):
        """Runs a segmentation model on the input image and thresholds result.

        Arguments:
            Input image
            Image with scaling lines
            Width of the original image
            Height of the original image
            Whether or not to plot the input/output and return the figure

        Returns:
            The thresholded bit-mask and (optionally) a figure of
            input/scaling/output.

        """
        pred_apo = self.predict(img)
        pred_apo_t = (pred_apo > self.apo_threshold)

        if not return_fig:
            # don't plot the input/output, simply return mask
            return pred_apo_t

        img = _resize(img, width, height)
        pred_apo_t = _resize(pred_apo_t, width, height)
        # remove outlying pixel structures
        pred_apo_tf = morphology.remove_small_objects(pred_apo_t > 0.2, min_size=5000,
                                                      connectivity=50).astype(int)
        # remove holes in predicted area
        pred_apo_th = morphology.remove_small_holes(pred_apo_tf > 0.2, area_threshold=5000,
                                                    connectivity=50).astype(int)

        # create figure with images
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.imshow(img_lines.squeeze(), cmap="gray")
        ax1.grid(False)
        ax1.set_title('Original Image with scaling line')
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.imshow(img.squeeze(), cmap='gray')
        ax2.grid(False)
        ax2.set_title('Original image with CLAHE')
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.imshow(pred_apo_th.squeeze(), cmap="gray")
        ax3.grid(False)
        ax3.set_title('Predicted muscle area')

        return pred_apo_th, fig

    def predict_s(self, img, img_lines, dist: str, width: int,
                      height: int, return_fig: bool = True):
        """Runs a segmentation model on the input image and thresholds result.

        Arguments:
            Input image
            Image with scaling lines
            Distance between scaling bars
            Width of the original image
            Height of the original image
            Whether or not to plot the input/output and return the figure

        Returns:
            The thresholded bit-mask and (optionally) a figure of
            input/scaling/output.

        """
        pred_apo = self.predict(img)
        pred_apo_t = (pred_apo > self.apo_threshold)

        if not return_fig:
            # don't plot the input/output, simply return mask
            return pred_apo_t

        img = _resize(img, width, height)
        pred_apo_t = _resize(pred_apo_t, width, height)
        pred_apo_tf = morphology.remove_small_objects(pred_apo_t > 0.2, min_size=5000,
                                                      connectivity=50).astype(int)
        # remove holes in predicted area
        pred_apo_th = morphology.remove_small_holes(pred_apo_tf > 0.2, area_threshold=5000,
                                                    connectivity=50).astype(int)


        # create figure with images
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.imshow(img_lines.squeeze(), cmap="gray")
        ax1.grid(False)
        ax1.set_title(str(dist))
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.imshow(img.squeeze(), cmap='gray')
        ax2.grid(False)
        ax2.set_title('Original image with CLAHE')
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.imshow(pred_apo_th.squeeze(), cmap="gray")
        ax3.grid(False)
        ax3.set_title('Predicted muscle area')

        return pred_apo_th, fig

    def predict_m(self, img, width: int, height: int, return_fig: bool = True):
        """Runs a segmentation model on the input image and thresholds result.

        Arguments:
            Input image
            Image with scaling lines
            Distance between scaling bars
            Width of the original image
            Height of the original image
            Whether or not to plot the input/output and return the figure

        Returns:
            The thresholded bit-mask and (optionally) a figure of
            input/scaling/output.

        """
        pred_apo = self.predict(img)
        pred_apo_t = (pred_apo > self.apo_threshold)

        if not return_fig:
            # don't plot the input/output, simply return mask
            return pred_apo_t

        img = _resize(img, width, height)
        pred_apo_t = _resize(pred_apo_t, width, height)
        # remove outlying pixel structures
        pred_apo_tf = morphology.remove_small_objects(pred_apo_t > 0.2, min_size=5000,
                                                      connectivity=50).astype(int)
        # remove holes in predicted area
        pred_apo_th = morphology.remove_small_holes(pred_apo_tf > 0.2, area_threshold=5000,
                                                    connectivity=50).astype(int)

        # create figure with images
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.imshow(img.squeeze(), cmap='gray')
        ax1.grid(False)
        ax1.set_title('Original image with CLAHE')
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.imshow(pred_apo_th.squeeze(), cmap="gray")
        ax2.grid(False)
        ax2.set_title('Predicted muscle area')

        return pred_apo_th, fig
