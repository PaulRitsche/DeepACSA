"""
Description
-----------
This module contains functions to train a VGG16 encoder U-net decoder CNN.
The module was specifically designed to be executed from a GUI.
When used from the GUI, the module saves the trained model and weights to
a given directory. The user needs to provide paths to the image and label/
mask directories. Instructions for correct image labelling can be found
in the Labelling directory.


Functions scope
---------------
conv_block
    Function to build a convolutional block for the U-net decoder path of the network.
    The block is built using several keras.layers functionalities.
decoder_block
    Function to build a decoder block for the U-net decoder path of the network.
    The block is built using several keras.layers functionalities.
build_vgg16_model
    Function that builds a convolutional network consisting of an VGG16 encoder path
    and a U-net decoder path.
IoU
    Function to compute the intersection over union score (IoU),
    a measure of prediction accuracy. This is sometimes also called Jaccard score.
dice_score
    Function to compute the Dice score, a measure of prediction accuracy.
focal_loss
     Function to compute the focal loss, a measure of prediction accuracy.
load_images
    Function to load images and manually labeled masks from a specified
    directory.
train_model
    Function to train a convolutional neural network with VGG16 encoder and
    U-net decoder. All the steps necessary to properly train a neural
    network are included in this function.

Notes
-----
Additional information and usage examples can be found at the respective
functions documentations.
"""
import os
import tkinter as tk

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
)
from keras.models import Model
from keras.optimizers import Adam
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16

# from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import img_to_array, load_img
from tqdm import tqdm


def conv_block(inputs, num_filters: int):
    """
    Function to build a convolutional block for the U-net decoder path of the network to be build.
    The block is built using several keras.layers functionalities.

    Here, we decided to use 'padding = same' and and a convolutional kernel of 3.
    This is adaptable in the code but will influence the model outcome.
    The convolutional block consists of two convolutional layers. Each creates a convolution kernel
    that is convolved with the layer input to produce a tensor of outputs.

    Parameters
    ----------
    inputs : KerasTensor
        Concattenated Tensorflow.Keras Tensor outputted from previous layer. The Tensor can be
        altered by adapting, i.e. the filter numbers but this will change the model training output.
        The input is then convolved using the built kernel.
    num_filters : int
        Integer variable determining the number of filters used during model training.
        Here, we started with 'num_filers = 512'. The filter number is halfed each
        layer. The number of filters can be adapted in the code.
        Must be non-negative and non-zero.

    Returns
    -------
    x : KerasTensor
        Tensorflow.Keras Tensor used during model Training.
        The Tensor can be altered by adapting the input paramenters to the function or
        the upsampling but this will change the model training. The number of filters
        is halfed.

    Notes
    -----
    - The function applies two Conv2D layers with 3x3 kernel size and "same" padding.
    - After each convolutional layer, BatchNormalization is applied to normalize the activations.
    - ReLU activation is applied after each BatchNormalization layer.
    - The purpose of this block is to improve the feature representation and stability of training in deeper networks.

    Example
    -------
    >>> conv_block(inputs=KerasTensor(type_spec=TensorSpec(shape=(None, 256, 256, 128),
                   dtype=tf.float32, name=None),
                   num_filters=128)
    KerasTensor(type_spec=TensorSpec(shape=(None, 256, 256, 64), dtype=tf.float32, name=None)
    """
    # Define Conv2D layer witch Batchnor and Activation relu
    x = Conv2D(filters=num_filters, kernel_size=3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # Define second Conv2D layer witch Batchnor and Activation relu
    x = Conv2D(filters=num_filters, kernel_size=3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def decoder_block(inputs, skip_features, num_filters):
    """
    Function to build a decoder block for the U-net decoder path of the network to be build.
    The block is build using several keras.layers functionalities.

    The block is built by applying a deconvolution (Keras.Conv2DTranspose) to upsample to input
    by a factor of 2. A concatenation with the skipped features from the mirrored
    vgg16 convolutional layer follows. Subsequently a convolutional block (see conv_block
    function) is applied to convolve the input with the built kernel.

    Parameters
    ----------
    inputs : KerasTensor
        Concattenated Tensorflow.Keras Tensor outputted from previous layer. The Tensor can be
        altered by adapting, i.e. the filter numbers but this will change the model training output.
    skip_features : Keras Tensor
        Skip connections to the encoder path of the vgg16 encoder.
    num_filters : int
        Integer variable determining the number of filters used during model training.
        Here, we started with 'num_filers = 512'. The filter number is halfed each
        layer. The number of filters can be adapted in the code. Must be non-neagtive and non-zero.

    Returns
    -------
    x : KerasTensor
        Tensorflow.Keras Tensor used during model Training. The tensor is upsampled using
        Keras.Conv2DTranspose with a kernel of (2,2), 'stride=2' and 'padding=same'.
        The upsampling increases image size by a factor of 2. The number of filters is halfed.
        The Tensor can be altered by adapting the input paramenters to the function or
        the upsampling but this will change the model training.

    Notes
    -----
    - The function applies a Conv2DTranspose layer with 2x2 kernel size and stride 2 to upsample the inputs.
    - The upsampled tensor is then concatenated with the skip connection tensor from the corresponding encoder block.
    - The concatenated tensor is passed through a pre-specified convolutional block defined by the 'conv_block' function.

    Example
    -------
    >>> decoder_block(inputs=KerasTensor(type_spec=TensorSpec(shape=(None, 64, 64, 512),
                      skip_features=KerasTensor(type_spec=TensorSpec(shape=(None, 64, 64, 512),
                      dtype=tf.float32, name=None)),
                      num_filters=256)
    KerasTensor(type_spec=TensorSpec(shape=(None, 128, 128, 256), dtype=tf.float32, name=None)
    """
    # Define a whole decoder block
    x = Conv2DTranspose(
        filters=num_filters, kernel_size=(2, 2), strides=2, padding="same"
    )(inputs)
    x = Concatenate()([x, skip_features])
    # Use pre-specified convolutional block
    x = conv_block(inputs=x, num_filters=num_filters)

    return x


def build_vgg16_unet(input_shape: tuple):
    """
    Function that builds a convolutional network consisting of an VGG16 encoder path
    and a U-net decoder path.

    The model is built using several Tensorflow.Keras functions. First, the whole VGG16
    model is imported and built using pretrained imagenet weights and the input shape.
    Then, the encoder layers are pulled from the model as well as the bridge part.
    Subsequently the decoder path from the U-net is built based on the VGG16 inputs.
    Lastly, a 1x1 convolution is applied with sigmoid activation to perform binary
    segmentation on the input.

    Parameters
    ----------
    input_shape : tuple
        Tuple describing the input shape. Must be of shape (...,...,...).
        Here we used (512,512,3) as input shape. The image size (512,512,)
        can be easily adapted. The channel numer (,,3) is given by the
        model and the pretrained weights. We advide the user not to change
        the image size segmentation results were best with the predefined
        size.

    Returns
    -------
    model
        The built VGG16 encoder U-net decoder convolutional network
        for binary segmentation on the input.
        The model can subsequently be used for training.

    Notes
    -----
    See our paper () and references for more detailed model description

    Example
    -------
    >>> input_shape = (256, 256, 3)
    >>> model = build_vgg16_unet(input_shape)
    # The function will create a VGG16 U-Net model for image segmentation with the specified input shape.
    # The model will be ready for compilation and training.

    References
    ----------
    VGG16: Simonyan, Karen, and Andrew Zisserman. “Very deep convolutional networks for large-scale image recognition.” arXiv preprint arXiv:1409.1556 (2014)
    U-net: Ronneberger, O., Fischer, P. and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." arXiv preprint arXiv:1505.04597 (2015)
    """
    # Get input shape
    _inputs = Input(input_shape)

    # Load vgg16 model
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=_inputs)

    # Get encoder part
    # skip connections
    s1 = vgg16.get_layer("block1_conv2").output  # 256
    s2 = vgg16.get_layer("block2_conv2").output  # 128
    s3 = vgg16.get_layer("block3_conv3").output  # 64
    s4 = vgg16.get_layer("block4_conv3").output  # 32

    # Get bottleneck/bridge part
    b1 = vgg16.get_layer("block5_conv3").output  # 16

    # Get decoder part
    d1 = decoder_block(inputs=b1, skip_features=s4, num_filters=512)
    d2 = decoder_block(inputs=d1, skip_features=s3, num_filters=256)
    d3 = decoder_block(inputs=d2, skip_features=s2, num_filters=128)
    d4 = decoder_block(inputs=d3, skip_features=s1, num_filters=64)

    # Model outputs
    _outputs = Conv2D(
        filters=1, kernel_size=(1, 1), padding="same", activation="sigmoid"
    )(d4)
    model = Model(_inputs, _outputs, name="VGG16_U-Net")

    return model


def IoU(y_true, y_pred, smooth: int = 1) -> float:
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

    Example
    -------
    >>> IoU(y_true=Tensor("IteratorGetNext:1", shape=(1, 512, 512, 1), dtype=float32),
            y_pred=Tensor("VGG16_U-Net/conv2d_8/Sigmoid:0", shape=(1, 512, 512, 1), dtype=float32),
            smooth=1)
    Tensor("truediv:0", shape=(1, 512, 512), dtype=float32)
    """
    # Caclulate Intersection
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    # Calculate Union
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)

    return iou


def dice_score(y_true, y_pred, smooth=1e-6) -> float:
    """
    Function to compute the Dice score, a measure of prediction accuracy.

    The Dice score can be used as a loss metric during binary segmentation when
    convolutional neural networks are applied. The Dice score is calculated for both the
    training and validation set.

    Parameters
    ----------
    y_true : tf.Tensor
        True positive image segmentation label predefined by the user.
        This is the mask that is provided prior to model training.
    y_pred : tf.Tensor
        Predicted image segmentation by the network.
    smooth : float
        Smoothing factor used for score calculation.

    Returns
    -------
    score : tf.Tensor
        Dice score representation in the same shape as y_true, y_pred.

    Notes
    -----
    The IoU is usually calculated as Dice = 2 * intersection / union.
    The intersection is calculated as the overlap of y_true and
    y_pred, whereas the union is the sum of y_true and y_pred.

    Example
    -------
    >>> IoU(y_true=Tensor("IteratorGetNext:1", shape=(1, 512, 512, 1), dtype=float32),
            y_pred=Tensor("VGG16_U-Net/conv2d_8/Sigmoid:0", shape=(1, 512, 512, 1), dtype=float32),
            smooth=1)
    Tensor("dice_score/truediv:0", shape=(1, 512, 512), dtype=float32)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice = (2 * intersection + smooth) / (
        K.sum(y_pred, -1) + K.sum(y_true, -1) + smooth
    )
    return 1 - dice


def focal_loss(y_true, y_pred, alpha: float = 0.8, gamma: float = 2) -> float:
    """
    Function to compute the focal loss, a measure of prediction accuracy.

    The focal loss can be used as a loss metric during binary segmentation when
    convolutional neural networks are applied. The focal loss score is calculated for both,
    the training and validation set. The focal loss is specifically applicable when
    class imbalances, i.e. between foregroung (muscle aponeurosis) and background (not
    muscle aponeurosis), are existent.

    Parameters
    ----------
    y_true : tf.Tensor
        True positive image segmentation label predefined by the user.
        This is the mask that is provided prior to model training.
    y_pred : tf.Tensor
        Predicted image segmentation by the network.
    alpha : float, default = 0.8
        Coefficient used on positive exaples, must be non-negative and non-zero.
    gamma : float, default = 2
        Focussing parameter, must be non-negative and non-zero.

    Returns
    -------
    f_loss : tf.Tensor
        Tensor containing the calculated focal loss score.

    Notes
    -----
    - Focal Loss is defined as -alpha * (1 - p)^gamma * log(p), where p is the predicted probability
      for the positive class (y_pred) and (1 - p) is the predicted probability for the negative class.
    - The loss function focuses more on hard-to-classify examples (low-confidence predictions) due to
      the presence of the gamma term, which increases the loss for well-classified examples.
    - The alpha parameter controls the weight assigned to the positive class, with alpha = 0.5 giving
      equal weight to both classes and higher alpha values favoring the positive class.

    Example
    -------
    >>> IoU(y_true=Tensor("IteratorGetNext:1", shape=(1, 512, 512, 1), dtype=float32),
            y_pred=Tensor("VGG16_U-Net/conv2d_8/Sigmoid:0", shape=(1, 512, 512, 1), dtype=float32),
            smooth=1)
    Tensor("focal_loss/Mean:0", shape=(), dtype=float32)
    """
    # Cacluate binary crossentropy loss
    BCE = K.binary_crossentropy(y_true, y_pred)
    # calculate exponentiated BCE
    BCE_EXP = K.exp(-BCE)
    # calculate focal loss
    f_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return f_loss


def loadImages(img_path: str, mask_path: str):
    """
    Function to load images and manually labeled masks from a specified
    directory.

    The images and masks are loaded, resized and normalized in order
    to be suitable and usable for model training. The specified directories
    must lead to the images and masks. The number of images and masks must be
    equal. The images and masks can be in any common image format.
    The names of the images and masks must match. The image and corresponding
    mask must have the same name.

    Parameters
    ----------
    img_path : str
        Path that leads to the directory containing the training images.
        Image must be in RGB format.
    mask_path : str
        Path that leads to the directory containing the mask images.
        Masks must be binary.

    Returns
    -------
    train_imgs : np.ndarray
        Resized, normalized training images stored in a numpy array.
    mask_imgs : np.ndarray
        Resized, normalized training masks stored in a numpy array.

    Notes
    -----
    See labelling instruction for correct masks creation and use,
    if needed, the supplied ImageJ script to label your images.

    Example
    -------
    >>> loadImages(img_path = "C:/Users/admin/Dokuments/images",
                   mask_path = "C:/Users/admin/Dokuments/masks")
    train_imgs([[[[0.22414216 0.19730392 0.22414216] ... [0.22414216 0.19730392 0.22414216]]])
    mask_imgs([[[[0.] ... [0.]]])
    """
    # Images will be re-scaled accordingly
    im_width = 256
    im_height = 256

    # list of all images in the path
    ids = os.listdir(img_path)
    print("Total no. of images = ", len(ids))

    # Create empty numpy arrays
    train_imgs = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    train_masks = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

    # ´Loop through list of ids found in img_path and mask_path
    for n, id_ in enumerate(tqdm(ids)):
        # Load and resize image
        img = load_img(img_path + id_, color_mode="rgb")
        img = img_to_array(img)
        img = resize(
            img, (im_width, im_height, 3), mode="constant", preserve_range=True
        )

        # Load and resize mask
        mask = img_to_array(load_img(mask_path + id_, color_mode="grayscale"))
        mask = resize(
            mask, (im_width, im_height, 1), mode="constant", preserve_range=True
        )

        # Normalize image & mask and insert in array
        train_imgs[n] = img / 255.0
        train_masks[n] = mask / 255.0

    return train_imgs, train_masks


def trainModel(
    img_path: str,
    mask_path: str,
    out_path: str,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    loss: str,
    gui,
) -> None:
    """
    Function to train a convolutional neural network with VGG16 encoder and
    U-net decoder. All the steps necessary to properly train an neural
    network are included in this function.

    This functions build upon all the other functions included in this module.
    Given that all input parameters are correctly specified, the images and
    masks are loaded, splittet into test and training sets, the model is
    compiled according to user specification and the model is trained.

    Parameters
    ----------
    img_path : str
        Path that leads to the directory containing the training images.
        Image must be in RGB format.
    mask_path : str
        Path that leads to the directory containing the mask images.
        Masks must be binary.
    out_path:
        Path that leads to the directory where the trained model should be saved.
    batch_size : int
        Integer value that determines the batch size per iteration through the
        network during model training. Although a larger batch size has
        advantages during model trainig, the images used here are large. Thus,
        the larger the batch size, the more compute power is needed or the
        longer the training duration. Must be non-negative and non-zero.
    learning_rate : float
        Float value determining the learning rate used during model training.
        Must be non-negative and non-zero.
    epochs : int
        Integer value that determines the amount of epochs that the model
        is trained befor training is aborted. The total amount of epochs
        will only be used if early stopping does not happen.
        Must be non-negative and non-zero.
    loss : str
        String variable that determines the loss function that is used during training.
        Three different types are supported here:
        - Binary cross-entropy. loss == "BCE"
        - Dice score. loss == "Dice"
        - Focal loss. loss == "FL"
        Each loss will yield a different result during model training.
    gui : tk.TK
        A tkinter.TK class instance that represents a GUI. By passing this
        argument, interaction with the GUI is possible i.e., stopping
        the model training model process.

    Returns
    -------
    None

    Notes
    -----
    For specific explanations for the included functions see the respective
    function docstrings in this module.
    This function can either be run from the command prompt or is called
    by the GUI. Note that the functioned was specifically designed to be
    called from the GUI. Thus, tk.messagebox will pop up when errors are
    raised even if the GUI is not started.

    Example
    -------
    >>> trainModel(img_path= "C:/Users/admin/Dokuments/images",
                   mask_path="C:/Users/admin/Dokuments/masks",
                   out_path="C:/Users/admin/Dokuments/results",
                   batch_size=1, learning_rate=0.005,
                   epochs=3, loss="BCE", gui)

    """
    # Check input paramters
    if batch_size <= 0 or learning_rate <= 0 or epochs <= 0:
        # Make sure some kind of filetype is specified.
        tk.messagebox.showerror(
            "Information", "Training parameters must be non-zero" + " and non-negative."
        )
        gui.should_stop = False
        gui.is_running = False
        gui.do_break()
        return

    # Images will be re-scaled accordingly
    im_width = 256
    im_height = 256

    # Adapt folder paths
    # This is necessary to concattenate id to path
    img_path = img_path + "/"
    mask_path = mask_path + "/"
    out_path = out_path + "/"

    try:
        # Load images
        train_imgs, train_masks = loadImages(img_path=img_path, mask_path=mask_path)

        # Inform user in GUI
        cont = tk.messagebox.askokcancel(
            "Information",
            "Images & Masks were successfully loaded!" + "\nDou you wish to proceed?",
        )
        if cont is True:

            ## Prepare data for model training
            # Split data into training and validation
            img_train, img_valid, mask_train, mask_valid = train_test_split(
                train_imgs, train_masks, test_size=0.1, random_state=42
            )

            ## Compose the VGG16 Unet model for aponeurosis detection
            # Compile the aponeurosis model VGG16
            VGG16_UNet = build_vgg16_unet((im_width, im_height, 3))
            model_apo = VGG16_UNet

            # Decide which loss metric is used
            if loss == "BCE":
                model_apo.compile(
                    optimizer=Adam(),
                    loss="binary_crossentropy",
                    metrics=["accuracy", IoU],
                )
            elif loss == "Dice":
                model_apo.compile(
                    optimizer=Adam(), loss=dice_score, metrics=["accuracy", IoU]
                )
            elif loss == "FL":
                model_apo.compile(
                    optimizer=Adam(), loss=focal_loss, metrics=["accuracy", IoU]
                )
            else:
                raise TypeError("Specify correct loss metric.")

            # Show a summary of the model structure
            model_apo.summary()

            # VGG16
            # Set some training parameters
            callbacks = [
                EarlyStopping(patience=8, verbose=1),
                ReduceLROnPlateau(
                    factor=0.1, patience=10, min_lr=learning_rate, verbose=1
                ),
                ModelCheckpoint(
                    out_path + "Test_Apo.h5",
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=False,
                ),  # Give the model a name (the .h5 part)
                CSVLogger(out_path + "Test_apo.csv", separator=",", append=False),
            ]

            # Inform user in GUI
            cont2 = tk.messagebox.askokcancel(
                "Information",
                "Model was successfully compiled!" + "\nDo you wish to proceed?",
            )
            # User chose to continue
            if cont2 is True:
                # VGG16
                results = model_apo.fit(
                    img_train,
                    mask_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(img_valid, mask_valid),
                )

                # Inform user in GUI
                tk.messagebox.showinfo(
                    "Information",
                    "Model was successfully trained"
                    + "\nResults are saved to specified output path.",
                )

                # Variables stored in results.history: val_loss, val_acc, val_IoU, loss, acc, IoU, lr
                fig, ax = plt.subplots(1, 2, figsize=(7, 7))
                ax[0].plot(results.history["loss"], label="Training loss")
                ax[0].plot(results.history["val_loss"], label="Validation loss")
                ax[0].set_title("Learning curve")
                ax[0].plot(
                    np.argmin(results.history["val_loss"]),
                    np.min(results.history["val_loss"]),
                    marker="x",
                    color="r",
                    label="best model",
                )
                ax[0].set_xlabel("Epochs")
                ax[0].set_ylabel("log_loss")
                ax[0].legend()

                ax[1].plot(results.history["val_IoU"], label="Training IoU")
                ax[1].plot(results.history["IoU"], label="Validation IoU")
                ax[1].set_title("IoU curve")
                ax[1].set_xlabel("Epochs")
                ax[1].set_ylabel("IoU score")
                ax[1].legend()
                plt.savefig(out_path + "Training_Results.tif")

            else:
                # User cancelled process after model compilation
                # clean up
                gui.do_break()
                gui.should_stop = False
                gui.is_running = False

        else:
            # User cancelled process after image loading
            # clean up
            gui.do_break()
            gui.should_stop = False
            gui.is_running = False

    # Error handling
    except ValueError:
        tk.messagebox.showerror(
            "Information",
            "Check input parameters."
            + "\nPotential error sources:"
            + "\n - Training parameters invalid",
        )
        # clean up
        gui.do_break()
        gui.should_stop = False
        gui.is_running = False

    except FileNotFoundError:
        tk.messagebox.showerror(
            "Information",
            "Check input directories."
            + "\nPotential error sources:"
            + "\n - Invalid specified input directories"
            + "\n - Unequal number of images or masks"
            + "\n - Names for images and masks don't match",
        )
        # clean up
        gui.do_break()
        gui.should_stop = False
        gui.is_running = False

    except PermissionError:
        tk.messagebox.showerror(
            "Information",
            "Check input directories."
            + "\nPotential error sources:"
            + "\n - Invalid specified input directories",
        )
        # clean up
        gui.do_break()
        gui.should_stop = False
        gui.is_running = False

    finally:
        # clean up
        gui.should_stop = False
        gui.is_running = False
