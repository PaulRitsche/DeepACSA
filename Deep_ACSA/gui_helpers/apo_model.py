""" Python class to predict muscle area

Description
-----------
This module provides a Python class called "ApoModel" for predicting muscle areas
in ultrasound (US) images. It uses a pre-trained segmentation model to predict the
probability of each pixel belonging to the foreground (aponeurosis).
The class supports various image types, such as those with scaling lines,
scaling bars, or manually scaled images. It also offers post-processing functions
to remove unnecessary areas, fill holes, and calculate the circumference of the largest contour.
The module allows users to return the thresholded bit-mask and optionally plot the input
image with the predicted muscle area overlay. Its purpose is to automate muscle area analysis
in US images for research and medical purposes.

Function scopes
---------------
_resize
    Resizes an input image to the specified height and width.
For scope of the functions used in the class ApoModel see class documentation.
"""

import tkinter as tk

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from skimage import measure, morphology
from skimage.transform import resize

from Deep_ACSA.gui_helpers.model_training import IoU, dice_score, focal_loss

matplotlib.use("Agg")


def _resize(img, width: int, height: int):
    """
    Resizes an image to height x width.

    Parameters
    ----------
    img : np.ndarray
        The input image to be resized.
    width : int
        The desired width of the output image.
    height : int
        The desired height of the output image.

    Returns
    -------
    np.ndarray
        The resized image as a 2-dimensional NumPy array with shape (height, width).

    Notes
    -----
    This function uses resize function from skimage.transform to initially resize the image,
    and then reshapes the result to the specified width and height using the 'numpy.reshape' function.

    Example
    -------
    >>> img = np.array([[1, 2], [3, 4]])
    >>> _resize(img, 3, 4)
    array([[1, 2, 1],
           [3, 4, 3],
           [1, 2, 1],
           [3, 4, 3]])

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

    Methods
    -------
    __init__(self, gui, model_path: str, loss_function: str, apo_threshold: float = 0.5):
        Initialize the ApoModel class.
    predict(self, gui, img):
        Runs a segmentation model on the input image.
    postprocess_image(self, img):
        Deletes unnecessary areas, fills holes, and calculates the length of the detected largest contour.
    predict_e(self, gui, img: np.ndarray, img_lines: np.ndarray, filename: str, width: int, height: int, return_fig: bool = True):
        Runs a segmentation model on the input image scaled with scaling lines and thresholds the result.
    predict_s(self, gui, img, img_lines, filename: str, dist: str, width: int, height: int, return_fig: bool = True):
        Runs a segmentation model on the input image scaled using provided scaling bars and thresholds the result.
    predict_m(self, gui, img, width: int, filename: str, height: int, return_fig: bool = True):
        Runs a segmentation model on the input image scaled manually and thresholds the result.

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

        Notes
        -----
        This constructor initializes the instance with the specified GUI object,
        the path to the pre-trained model, the chosen loss function, and the anomaly
        detection threshold. The model is loaded based on the selected loss function.

        Supported loss functions are:
        - "IoU" (Intersection over Union)
        - "Dice Loss"
        - "Focal Loss"

        Example
        -------
        >>> gui = GUI
        >>> model_path = "path/to/your/model.h5"
        >>> loss_function = "IoU"
        >>> apo_threshold = 0.7
        >>> instance = ApoModel(gui, model_path, loss_function, apo_threshold)

        """

        try:
            self.model_path = model_path
            self.apo_threshold = apo_threshold

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

        Parameters
        ----------
        gui : GUI
            The GUI object associated with this method.

        img : np.ndarray
            The input image on which the segmentation model will be applied.

        Returns
        -------
        np.ndarray
            The probability for each pixel, indicating its likelihood to belong to the foreground.

        Notes
        -----
        This method takes an input image and applies a segmentation model to predict the
        probability of each pixel belonging to the foreground. The 'model_apo' attribute of
        the class should be previously loaded with a segmentation model.

        Example
        -------
        >>> gui = GUI
        >>> img = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> detector = ApoModel
        >>> prediction = detector.predict(gui, img)
        >>> print(prediction)
        array([[0.8, 0.9],
            [0.7, 0.6]])

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

        Parameters
        ----------
        img : np.ndarray
            The input image to be postprocessed.

        Returns
        -------
        float, np.ndarray
            A float value representing the circumference of the detected largest contour.
            An image containing only the largest area of pixels with holes removed.

        Notes
        -----
        This method takes an input image and performs the following steps:
        1. Finds pixel regions and labels them using `measure.label`.
        2. Sorts the regions by area in descending order.
        3. Removes all regions except the one with the largest area, effectively
           keeping only the largest area in the image.
        4. Fills holes in the largest area using `morphology.remove_small_holes`.
        5. Smooths the edges of the predicted area.
        6. Calculates the circumference of the largest contour in the image.

        Example
        -------
        >>> img = np.array([[0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0]])
        >>> model = ApoModel()
        >>> circumference, processed_img = model.postprocess_image(img)
        >>> print(circumference)
        16.0
        >>> print(processed_img)
        array([[0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0]])

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

        Notes
        -----
        This method runs a segmentation model on the input image and thresholds the
        resulting probabilities using the specified `apo_threshold`. If `return_fig`
        is True, it also returns a matplotlib figure displaying the original image
        with scaling lines, the normalized and resized image with the predicted
        muscle area overlay.

        Example
        -------
        >>> gui = GUI
        >>> img = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> img_lines = np.array([[0.5, 0.5], [0.5, 0.5]])
        >>> filename = "example.png"
        >>> width = 640
        >>> height = 480
        >>> model = ApoModel()
        >>> circum, pred_mask, fig = model.predict_e(gui, img, img_lines, filename, width, height)
        >>> print(circum)
        16.0
        >>> print(pred_mask)
        array([[False, False],
              [False, False]])
        >>> print(fig)
        <matplotlib.figure.Figure object at 0x...>
        """
        pred_apo = self.predict(gui, img)
        pred_apo_t = pred_apo > self.apo_threshold

        if not return_fig:
            # Don't plot the input/output, simply return the mask.
            return pred_apo_t

        img = _resize(img, width, height)
        pred_apo_t = _resize(pred_apo_t, width, height)

        # Postprocess the image.
        circum, pred_apo_th = self.postprocess_image(pred_apo_t)

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

        Notes
        -----
        This method runs a segmentation model on the input image, which was scaled
        using the provided scaling bars. It then thresholds the resulting probabilities
        using the specified `apo_threshold`. If `return_fig` is True, it also returns
        a matplotlib figure displaying the original image with scaling lines, the
        normalized and resized image with the predicted muscle area overlay.

        Example
        -------
        >>> gui = GUI
        >>> img = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> img_lines = np.array([[0.5, 0.5], [0.5, 0.5]])
        >>> filename = "example.png"
        >>> dist = "10 mm"
        >>> width = 640
        >>> height = 480
        >>> model = ApoModel()
        >>> circum, pred_mask, fig = model.predict_s(gui, img, img_lines, filename, dist, width, height)
        >>> print(circum)
        16.0
        >>> print(pred_mask)
        array([[False, False],
               [False, False]])
        >>> print(fig)
        <matplotlib.figure.Figure object at 0x...>
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

        Notes
        -----
        This method runs a segmentation model on the input image, which was scaled manually.
        It then thresholds the resulting probabilities using the specified `apo_threshold`.
        If `return_fig` is True, it also returns a matplotlib figure displaying the
        normalized and resized image with the predicted muscle area overlay.

        Example
        -------
        >>> gui = GUI
        >>> img = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> filename = "example.png"
        >>> width = 640
        >>> height = 480
        >>> model = ApoModel()
        >>> circum, pred_mask, fig = model.predict_m(gui, img, width, filename, height)
        >>> print(circum)
        16.0
        >>> print(pred_mask)
        array([[False, False],
               [False, False]])
        >>> print(fig)
        <matplotlib.figure.Figure object at 0x...>
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
