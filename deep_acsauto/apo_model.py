import numpy as np
from skimage.transform import resize
from keras import backend as K
from keras.models import load_model  # Model

import matplotlib.pyplot as plt
plt.style.use("ggplot")


<<<<<<< HEAD
def _resize(img, width: int, height: int):
=======
def _resize(img, height: int, width: int):
>>>>>>> 1c084e7f70bc8eec3873facdba02b8bd6b7b3732
    """Resizes an image to height x width.

    Args:
        Image to be resized,
<<<<<<< HEAD
        Target width,
        Target height,
=======
        Target height,
        Target width,
>>>>>>> 1c084e7f70bc8eec3873facdba02b8bd6b7b3732

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

    def predict_t(self, img, width: int, height: int, return_fig: bool = True):
        """Runs a segmentation model on the input image and thresholds result.

        Arguments:
            Input image
            Width of the original image
            Height of the original image
            Whether or not to plot the input/output and return the figure

        Returns:
            The thresholded bit-mask and (optionally) a figure of input/output.

        """
        pred_apo = self.predict(img)
        pred_apo_t = (pred_apo > self.apo_threshold).astype(np.uint8)

        if not return_fig:
            # don't plot the input/output, simply return mask
            return pred_apo_t

        img = _resize(img, width, height)
        pred_apo_t = _resize(pred_apo_t, width, height)

        fig = plt.figure(figsize=(20, 20))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(img.squeeze(), cmap='gray')
        ax1.grid(False)
        ax1.set_title('Original image')
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(pred_apo_t.squeeze(), cmap="gray")
        ax2.grid(False)
        ax2.set_title('Aponeuroses')

        return pred_apo_t, fig
