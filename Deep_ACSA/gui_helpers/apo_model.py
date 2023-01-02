""" Python class to predict muscle area"""

import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from cv2 import CHAIN_APPROX_SIMPLE, RETR_LIST, arcLength, findContours
from keras import backend as K
from keras.models import load_model
from skimage import measure, morphology
from skimage.transform import resize

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
    """
    Function to compute the intersection over union score (IoU),
    a measure of prediction accuracy. This is sometimes also called Jaccard score.

    The IoU can be used as a loss metric during binary segmentation when
    convolutional neural networks are applied. The IoU is calculated for both the
    training and validation set.

    Parameters
    ----------
    y_true : tf.Tensor
        True positive image segmentation label predefined by the user.
        This is the mask that is provided prior to model training.
    y_pred : tf.Tensor
        Predicted image segmentation by the network.
    smooth : int, default = 1
        Smoothing operator applied during final calculation of
        IoU. Must be non-negative and non-zero.

    Returns
    -------
    iou : tf.Tensor
        IoU representation in the same shape as y_true, y_pred.

    Notes
    -----
    The IoU is usually calculated as IoU = intersection / union.
    The intersection is calculated as the overlap of y_true and
    y_pred, whereas the union is the sum of y_true and y_pred.

    Examples
    --------
    >>> IoU(y_true=Tensor("IteratorGetNext:1", shape=(1, 512, 512, 1), dtype=float32),
            y_pred=Tensor("VGG16_U-Net/conv2d_8/Sigmoid:0", shape=(1, 512, 512, 1), dtype=float32),
            smooth=1)
    Tensor("truediv:0", shape=(1, 512, 512), dtype=float32)
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
        >>> # get predictions only
        >>> pred_apo = apo_model.predict(img)
        >>> pred_apo_t = apo_model.predict_t(img, width, height, False)
        >>>
        >>> # get predictions and plot (the following two are identical)
        >>> pred_apo_t, fig = apo_model.predict_t(img, width, height)
        >>> pred_apo_t, fig = apo_model.predict_t(img, width, height, True)

    """

    def __init__(self, gui, model_path: str, apo_threshold: float = 0.5):

        try:
            self.model_path = model_path
            self.apo_threshold = apo_threshold
            self.model_apo = load_model(self.model_path, custom_objects={"IoU": IoU})

        # Check if model directory is correct
        except OSError:
            tk.messagebox.showerror(
                "Information",
                "Invalid model path."
                + "\nPotential error source:  Wrong (model) file selected",
            )

    def predict(self, gui, img):
        """Runs a segmentation model on the input image.

        Arguments:
            Input image

        Returns:
            The probability for each pixel, that it belongs to the foreground.

        """
        try:
            pred_apo = self.model_apo.predict(img)
            return pred_apo

        # Model path was specified incorrectly
        except AttributeError:
            gui.should_stop = False
            gui.is_running = False
            gui.do_break()
            return

    def postprocess_image(self, img):
        """Deletes unnecessary areas, fills holes and calculates the length
           of the detected largest contour.

        Arguments:
            Input image

        Returns:
            Image containing only largest area of pixels with holes removed.
            Float containing circumference.
        """
        # find pixel regions and label them
        label_img = measure.label(img)
        regions = measure.regionprops(label_img)

        # sort regions for area
        regions.sort(key=lambda x: x.area, reverse=True)

        # find label with larges area
        if len(regions) > 1:
            for rg in regions[1:]:
                label_img[rg.coords[:, 0], rg.coords[:, 1]] = 0

        label_img[label_img != 0] = 1
        pred_apo_tf = label_img

        # remove holes in predicted area
        pred_apo_th = morphology.remove_small_holes(
            pred_apo_tf > 0.5, area_threshold=5000, connectivity=100
        ).astype(int)
        # calculate circumference
        pred_apo_conts = pred_apo_th.astype(np.uint8)
        conts, hirarchy = findContours(pred_apo_conts, RETR_LIST, CHAIN_APPROX_SIMPLE)
        for cont in conts:
            circum = arcLength(cont, True)

        return circum, pred_apo_th

    def predict_e(
        self,
        gui,
        img,
        img_lines,
        filename: str,
        width: int,
        height: int,
        return_fig: bool = True,
    ):
        """Runs a segmentation model on the input image and thresholds result.

        Arguments:
            Input image
            Image with scaling lines
            Name of image
            Width of the original image
            Height of the original image
            Whether or not to plot the input/output and return the figure

        Returns:
            The thresholded bit-mask and (optionally) a figure of
            input/scaling/output.

        """
        pred_apo = self.predict(gui, img)
        pred_apo_t = pred_apo > self.apo_threshold

        if not return_fig:
            # don't plot the input/output, simply return mask
            return pred_apo_t

        img = _resize(img, width, height)
        pred_apo_t = _resize(pred_apo_t, width, height)

        # postprocess image
        circum, pred_apo_th = self.postprocess_image(pred_apo_t)

        # create figure with images
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.imshow(img_lines.squeeze(), cmap="gray")
        ax1.grid(False)
        ax1.set_title(f"Image ID: {filename}" + "\nOriginal Image with scaling line")
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.imshow(img.squeeze(), cmap="gray")
        ax2.grid(False)
        ax2.set_title("Resized and normalized  original image")
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.imshow(pred_apo_th.squeeze(), cmap="gray")
        ax3.grid(False)
        ax3.set_title("Predicted muscle area")

        return circum, pred_apo_th, fig

    def predict_s(
        self,
        gui,
        img,
        img_lines,
        filename: str,
        dist: str,
        width: int,
        height: int,
        return_fig: bool = True,
    ):
        """Runs a segmentation model on the input image and thresholds result.

        Arguments:
            Input image
            Image with scaling lines
            Name of file
            Distance between scaling bars
            Width of the original image
            Height of the original image
            Whether or not to plot the input/output and return the figure

        Returns:
            The thresholded bit-mask and (optionally) a figure of
            input/scaling/output.

        """
        pred_apo = self.predict(gui, img)
        pred_apo_t = pred_apo > self.apo_threshold

        if not return_fig:
            # don't plot the input/output, simply return mask
            return pred_apo_t

        img = _resize(img, width, height)
        pred_apo_t = _resize(pred_apo_t, width, height)

        # postprocess image
        circum, pred_apo_th = self.postprocess_image(pred_apo_t)

        # create figure with images
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.imshow(img_lines.squeeze(), cmap="gray")
        ax1.grid(False)
        ax1.set_title(
            f"Image ID: {filename}" + f"\nDistance between scaling bars {dist}"
        )
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.imshow(img.squeeze(), cmap="gray")
        ax2.grid(False)
        ax2.set_title("Resized and normalized  original image")
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.imshow(pred_apo_th.squeeze(), cmap="gray")
        ax3.grid(False)
        ax3.set_title("Predicted muscle area")

        return circum, pred_apo_th, fig

    def predict_m(
        self, gui, img, width: int, filename: str, height: int, return_fig: bool = True
    ):
        """Runs a segmentation model on the input image and thresholds result.

        Arguments:
            Input image
            Image with scaling lines
            Name of the image
            Distance between scaling bars
            Width of the original image
            Height of the original image
            Whether or not to plot the input/output and return the figure

        Returns:
            The thresholded bit-mask and (optionally) a figure of
            input/scaling/output.

        """
        pred_apo = self.predict(gui, img)
        pred_apo_t = pred_apo > self.apo_threshold

        if not return_fig:
            # don't plot the input/output, simply return mask
            return pred_apo_t

        img = _resize(img, width, height)
        pred_apo_t = _resize(pred_apo_t, width, height)

        # postprocess image
        circum, pred_apo_th = self.postprocess_image(pred_apo_t)

        # create figure with images
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.imshow(img.squeeze(), cmap="gray")
        ax1.grid(False)
        ax1.set_title(f"Image ID: {filename}" + "\nOriginal image with CLAHE")
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.imshow(pred_apo_th.squeeze(), cmap="gray")
        ax2.grid(False)
        ax2.set_title("Predicted muscle area")

        return circum, pred_apo_th, fig
