""" Python class to predict muscle area"""

import tkinter as tk

import cv2
import matplotlib.pyplot as plt
import numpy as np

# from cv2 import CHAIN_APPROX_SIMPLE, RETR_LIST, arcLength, findContours
from keras.models import load_model
from skimage import measure, morphology
from skimage.transform import resize

from Deep_ACSA.gui_helpers.model_training import IoU, dice_score, focal_loss

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


class ApoModel:
    """Class which provides utility to predict aponeurosis on US-images.

    Attributes
    ----------
    model_path : str
        Path to the Keras segmentation model.
    apo_threshold : float, optional
        Pixels above this threshold are assumed to be aponeurosis.
    loss_function : str
            The used loss function for training the model.
    model_apo : keras.Model
        The loaded segmentation model.

    Examples
    --------
    >>> apo_model = ApoModel(gui, 'path/to/model.h5', 'IoU')
    >>> # get predictions only
    >>> pred_apo = apo_model.predict(img)
    >>> pred_apo_t = apo_model.predict_t(img, width, height, False)
    >>>
    >>> # get predictions and plot (the following two are identical)
    >>> pred_apo_t, fig = apo_model.predict_t(img, width, height)
    >>> pred_apo_t, fig = apo_model.predict_t(img, width, height, True)
    """

    def __init__(
        self, gui, model_path: str, loss_function: str, apo_threshold: float = 0.5
    ):
        """
        Initialize the ApoModel class.

        Parameters
        ----------
        gui : GUI
            The GUI object.
        model_path : str
            Path to the Keras segmentation model.
        loss_function : str
            The used loss function for training the model.
        apo_threshold : float, optional
            Pixels above this threshold are assumed to be aponeurosis.

        Raises
        ------
        OSError
            If the model directory is incorrect.

        """
        try:
            self.model_path = model_path
            self.apo_threshold = apo_threshold
            print(loss_function)
            # Check for used loss function
            if loss_function == "IoU":
                self.model_apo = load_model(
                    self.model_path, custom_objects={"IoU": IoU}
                )

            if loss_function == "Dice Loss":
                self.model_apo = load_model(
                    self.model_path,
                    custom_objects={"dice_score": dice_score, "IoU": IoU},
                )

            if loss_function == "Focal Loss":
                self.model_apo = load_model(
                    self.model_path,
                    custom_objects={"focal_loss": focal_loss, "IoU": IoU},
                )

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
        # Find pixel regions and label them
        label_img = measure.label(img)
        regions = measure.regionprops(label_img)

        # Sort regions by area
        regions.sort(key=lambda x: x.area, reverse=True)

        # Find label with the largest area
        if len(regions) > 1:
            for rg in regions[1:]:
                label_img[rg.coords[:, 0], rg.coords[:, 1]] = 0

        label_img[label_img != 0] = 1
        pred_apo_tf = label_img

        # Remove holes in predicted area
        pred_apo_th = morphology.remove_small_holes(
            pred_apo_tf > 0.5, area_threshold=5000, connectivity=100
        ).astype(int)

        # Smooth the edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        pred_apo_th = cv2.dilate(pred_apo_th.astype(np.uint8), kernel, iterations=2)
        pred_apo_th = cv2.erode(pred_apo_th, kernel, iterations=2)

        # Calculate circumference
        pred_apo_conts = pred_apo_th.astype(np.uint8)
        conts, hierarchy = cv2.findContours(
            pred_apo_conts, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        circum = 0.0
        for cont in conts:
            circum += cv2.arcLength(cont, True)
            cv2.drawContours(pred_apo_th, [cont], 0, (255, 255, 255), 1)

        return circum, pred_apo_th

    def predict_e(
        self,
        gui,
        img: np.ndarray,
        img_lines: np.ndarray,
        filename: str,
        width: int,
        height: int,
        return_fig: bool = True,
    ):
        """Runs a segmentation model on the input image and
        thresholds the result.

        The input image here containes the scaling lines.

        Parameters
        ----------
        gui :
            The GUI object.
        img : np.ndarray
            The input image.
        img_lines : np.ndarray
            The image with scaling lines.
        filename : str
            The name of the image.
        width : int
            The width of the original image.
        height : int
            The height of the original image.
        return_fig : bool, optional
            Whether or not to plot the input/output and return the figure.

        Returns
        -------
        Union[np.ndarray, Tuple[float, np.ndarray, plt.Figure]]
            If `return_fig` is False, returns the thresholded bit-mask.
            If `return_fig` is True, returns the circumference,
            thresholded bit-mask,
            and a figure of input/scaling/output.

        """
        pred_apo = self.predict(gui, img)
        pred_apo_t = pred_apo > self.apo_threshold
        print(pred_apo_t)

        if not return_fig:
            # Don't plot the input/output, simply return the mask.
            return pred_apo_t

        img = _resize(img, width, height)
        pred_apo_t = _resize(pred_apo_t, width, height)

        # Postprocess the image.
        circum, pred_apo_th = self.postprocess_image(pred_apo_t)

        cv2.imshow("window", pred_apo_th)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Create figure with images.
        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.imshow(img_lines.squeeze(), cmap="gray")
        ax1.grid(False)
        ax1.set_title(f"Image ID: {filename}" + "\nOriginal Image with scaling line")
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.imshow(img.squeeze(), cmap="gray")
        ax2.contour(
            pred_apo_th.squeeze(), levels=[0.5], colors="cyan", linewidths=4, alpha=0.4
        )
        ax2.grid(False)
        ax2.set_title(
            "Normalized and resized image with predicted muscle area (overlay)"
        )

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
        """Runs a segmentation model on the input image and
        thresholds the result.

        The input image here was scaled using the scaling bars.

        Parameters
        ----------
        gui :
            The GUI object.
        img : np.ndarray
            The input image.
        img_lines : np.ndarray
            The image with scaling lines.
        filename : str
            The name of the image.
        width : int
            The width of the original image.
        height : int
            The height of the original image.
        return_fig : bool, optional
            Whether or not to plot the input/output and return the figure.

        Returns
        -------
        Union[np.ndarray, Tuple[float, np.ndarray, plt.Figure]]
            If `return_fig` is False, returns the thresholded bit-mask.
            If `return_fig` is True, returns the circumference,
            thresholded bit-mask,
            and a figure of input/scaling/output.
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
        ax1.imshow(img_lines.squeeze(), cmap="gray")
        ax1.grid(False)
        ax1.set_title(
            f"Image ID: {filename}" + f"\nDistance between scaling bars {dist}"
        )
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.imshow(img.squeeze(), cmap="gray")
        ax2.contour(
            pred_apo_th.squeeze(), levels=[0.5], colors="cyan", linewidths=4, alpha=0.4
        )
        ax2.grid(False)
        ax2.set_title(
            "Normalized and resized image with predicted muscle area (overlay)"
        )

        return circum, pred_apo_th, fig

    def predict_m(
        self, gui, img, width: int, filename: str, height: int, return_fig: bool = True
    ):
        """Runs a segmentation model on the input image and
        thresholds the result.

        The input image here was scaled manualy.

        Parameters
        ----------
        gui :
            The GUI object.
        img : np.ndarray
            The input image.
        img_lines : np.ndarray
            The image with scaling lines.
        filename : str
            The name of the image.
        width : int
            The width of the original image.
        height : int
            The height of the original image.
        return_fig : bool, optional
            Whether or not to plot the input/output and return the figure.

        Returns
        -------
        Union[np.ndarray, Tuple[float, np.ndarray, plt.Figure]]
            If `return_fig` is False, returns the thresholded bit-mask.
            If `return_fig` is True, returns the circumference,
            thresholded bit-mask,
            and a figure of input/scaling/output.
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
        ax2 = fig.add_subplot(2, 1, 1)
        ax2.imshow(img.squeeze(), cmap="gray")
        ax2.contour(
            pred_apo_th.squeeze(), levels=[0.5], colors="cyan", linewidths=4, alpha=0.4
        )
        ax2.grid(False)
        ax2.set_title(
            "Normalized and resized image with predicted muscle area (overlay)"
        )

        return circum, pred_apo_th, fig
