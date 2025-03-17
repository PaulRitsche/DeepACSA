"""
Description
-----------
This module contains a class with methods to automatically and manually annotate
transversal ultrasonography images. When the class is initiated,
a graphical user interface is opened. This is the main GUI of the DeepACSA package.
From here, the user is able to navigate all functionalities of the package.
These extend the methods in this class. The main functionalities of the GUI contained
in this module are automatic and manual evalution of muscle ultrasonography images.
Inputted images are analyzed and the parameter muscle anatomical cross-sectional area is
returned for each image.
The parameters are analyzed using convolutional neural networks (U-net, VGG16).
This module and all submodules contained in the /gui_helpers modality are extensions
and improvements of the work presented in Ritsche et al. (2022). There, the core functionalities
of this code are already outlined and the comparability of the model segmentations to
manual analysis (current gold standard) is described. Here, we have improved the code.

Functions scope
---------------
For scope of the functions see class documentation.

Notes
-----
Additional information and usage exaples can be found in the video
tutorials provided for this package.

References
----------
VGG16: Simonyan, K., and Zisserman, A. “Very deep convolutional networks for large-scale image recognition.” arXiv preprint arXiv:1409.1556 (2014)
U-net: Ronneberger, O., Fischer, P. and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." arXiv preprint arXiv:1505.04597 (2015)
DeepACSA: Ritsche, P., Wirth, P., Cronin, N., Sarto, F., Narici, M., Faude, O., Franchi, M. "DeepACSA: Automatic Segmentation of Cross-Sectional Area in Ultrasound Images of Lower Limb Muscles Using Deep Learning" (2022)
"""

import os
import customtkinter as ctk
import tkinter as tk
from threading import Lock, Thread
from tkinter import E, N, S, StringVar, Tk, W, filedialog, ttk

import matplotlib

from DeepACSA import gui_helpers

matplotlib.use("TkAgg")


class DeepACSA(ctk.CTk):
    """
    Python class to automatically or manually annotate tranversal muscle
    ultrasonography images of human lower limb muscles.
    An analysis tkinter GUI is opened upon initialization of the class.
    By clicking the buttons, the user can switch between image analysis and model training.
    The GUI consistsof the following elements.
    - Input Directory:
    By pressing the "Input" button, the user is asked to select
    an input directory containing all images to be
    analyzed. This can also be entered directly in the entry field.
    - Apo Model Path:
    By pressing the "Apo Model" button, the user is asked to select
    the aponeurosis model used for aponeurosis segmentation. The absolute
    model path can be entered directly in the enty field as well.
    - Break:
    By pressing the "break" button, the user is able to stop the analysis process
    after each finished image or image frame analysis.
    - Run:
    By pressing the "run" button, the user can start the analysis process.
    - Model training:
    By pressing the "train model" button, a new window opens and the
    user can train an own neural network based on existing/own
    training data. Furthermore, the new window allows to augment input images through
    data augmentation to generate new training images.

    Attributes
    ----------
    self._lock : _thread.lock
        Thread object to lock the self._lock variable for access by another thread.
    self._is_running : bool, default = False
        Boolen variable determining the active state
        of the GUI. If False, the is not running. If True
        the GUI is running.
    self._should_stop : bool, default = False
        Boolen variable determining the active state
        of the GUI. If False, the is allowed to continue running. If True
        the GUI is stopped and background processes deactivated.
    self.main : tk.TK
        tk.TK instance which is the base frame for the GUI.
    self.input : tk.Stringvar
        tk.Stringvariable containing the path to the input directory.
    self.model : tk.Stringvar
        tk.Stringvariable containing the path to the aponeurosis
        model.
    self.scaling : tk.Stringvar
        tk.Stringvariable containing the selected scaling type.
        This can be "bar", "manual" or "no scaling".
    self.muscle_volume_calculation_wanted : tk.Stringvar
        tk.Stringvariale defining whether the muscle volume
        should be calculated.
        This can be "yes" or "no".
    self.filetype : tk.Stringvar
        tk.Stringvariabel containing the selected filetype for
        the images to be analyzed. The user can select from the
        dopdown list or enter an own filetype. The formatting
        should be kept constant.
    self.depth : tk.Stringvar
        tk.Stringvariable containing the selected image depth.
        This must be according to the image depth used during
        image acquisition.
    self.spacing : tk.Stringvar
        tk.Stringvariable containing the selected spacing distance
        used for computation of pixel / cm ratio. This must only be
        specified when the analysis type "bar" or "manual" is selected.
    self.distance : tk.Stringvar
        tk.Stringvariable containing the space between acquired images
        along one muscle. The space must be given in mm.
    self.train_image_dir : tk.Stringvar
        tk.Straingvar containing the path to the directory of the training images.
        Image must be in RGB format.
    self.mask_dir : tk.Stringvar
        tk.Stringvariable containing the path to the directory of the mask images.
        Masks must be binary.
    self.out_dir : tk.Stringvar
        tk.Stringvariable containing the path to the directory where the trained model
        should be saved.
    self.batch_size : tk.Stringvar
        tk.Stringvariable containing the batch size per iteration through the
        network during model training. Must be non-negative and non-zero.
    self.learning_rate : tk.Stringvariable
        tk.Stringvariable the learning rate used during model training.
        Must be non-negative and non-zero.
    self.epochs : tk.Stringvar
        tk.Straingvariable containing the amount of epochs that the model
        is trained befor training is aborted. Must be non-negative and non-zero.
    self.loss : tk.Stringvar
        tk.Stringvariable containing the loss function that is used during training.

    Methods
    -------
    get_root_dir
        Instance method to ask the user to select the input directory.
    get_model_path
        Instance method to ask the user to select the apo model path.
    run_code
        Instance method to execute the analysis process when the
        "run" button is pressed.
    do_break
        Instance method to break the analysis process when the
        button "break" is pressed.
    train_model_window
        Instance method to open new window for model training.
    get_train_dir
        Instance method to ask the user to select the training image
        directory path.
    get_mask_dir
        Instance method to ask the user to select the training mask
        directory path.
    get_output_dir
        Instance method to ask the user to select the output
        directory path.
    train_model
        Instance method to execute the model training when the
        "start training" button is pressed.
    augment_images
        Instance method to augment input images and masks,
        when the "Augment Images" button is pressed.

    Notes
    -----
    This class contains only instance attributes.
    The instance methods contained in this class are solely purposed for
    support of the main GUI instance method. They cannot be used
    independantly or seperately.

    For more detailed documentation of the functions employed
    in this GUI upon running the analysis or starting model training
    see the respective modules in the /gui_helpers subfolder.

    See Also
    --------
    model_training.py, predict_muscle_area.py
    """

    def __init__(self, *args, **kwargs):
        """Initialize the DeepACSA GUI application.

        Parameters
        ----------
            root (Tk): The Tkinter root window.

        Returns
        -------
            None
        """
        super().__init__(*args, **kwargs)

        # set up threading
        self._lock = Lock()
        self._is_running = False
        self._should_stop = False

        # set up gui
        self.title("DeepACSA")
        master_path = os.path.dirname(os.path.abspath(__file__))
        ctk.set_default_color_theme(
            master_path + "/gui_helpers/gui_files/ui_color_theme.json"
        )
        iconpath = master_path + "/gui_helpers/icon.ico"
        self.iconbitmap(iconpath)

        self.main = ctk.CTkFrame(self)
        self.main.grid(column=0, row=0, sticky=(N, S, W, E))
        # Configure resizing of user interface
        self.main.columnconfigure(0, weight=1)
        self.main.columnconfigure(1, weight=1)
        self.main.columnconfigure(2, weight=1)
        self.main.columnconfigure(3, weight=1)
        self.main.columnconfigure(4, weight=1)
        self.main.columnconfigure(5, weight=1)
        # root.columnconfigure(0, weight=1)
        # root.rowconfigure(0, weight=1)

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#7ABAA1")
        style.configure(
            "TLabel",
            font=("Lucida Sans", 12),
            foreground="black",
            background="#7ABAA1",
        )
        style.configure(
            "TRadiobutton",
            background="#7ABAA1",
            foreground="black",
            font=("Lucida Sans", 12),
        )
        style.configure(
            "TButton",
            background="#EADCC3",
            foreground="black",
            font=("Lucida Sans", 11),
        )
        style.configure(
            "TEntry", font=("Lucida Sans", 12), background="linen", foregrund="black"
        )
        style.configure("TSeparator", background="linen", foreground="linen")
        style.configure("TCombobox", background="#7ABAA1", foreground="black")

        # Paths
        # Input directory
        self.input = ctk.StringVar()
        input_entry = ctk.CTkEntry(self.main, width=30, textvariable=self.input)
        input_entry.grid(column=2, row=2, columnspan=3, sticky=(W, E))
        # self.input.set("Desktop/DeepACSA/")
        # Model path
        self.model = ctk.StringVar()
        model_entry = ctk.CTkEntry(self.main, width=30, textvariable=self.model)
        model_entry.grid(column=2, row=3, columnspan=3, sticky=(W, E))
        # self.model.set("C:/Users/admin/Documents/DeepACSA/notebooks/VGG16pre-VL-256.h5")

        self.scaling = ctk.StringVar(value="Bar")
        self.scaling_menu = ctk.CTkComboBox(
            self.main,
            variable=self.scaling,
            values=["Line", "Bar", "Manual"],
            command=self.on_scaling_change,
        )
        self.scaling_menu.grid(column=2, row=7, sticky=W)

        self.volume_calc_wanted = ctk.StringVar(value="No")
        self.volume_menu = ctk.CTkComboBox(
            self.main,
            variable=self.volume_calc_wanted,
            values=["Yes", "No"],
            command=self.on_volume_change,
            state="readonly",
        )
        self.volume_menu.grid(column=3, row=14, sticky=(W, E))

        # Comboboxes
        # Loss Function
        self.loss_function = StringVar()
        loss_entry = ctk.CTkComboBox(
            self.main,
            width=15,
            values=["IoU", "Dice Loss", "Focal Loss"],
            variable=self.loss_function,
            state="readonly",
        )
        loss_entry.grid(column=4, row=4, sticky=(W, E))
        self.loss_function.set("Loss Function")

        # Muscles
        self.muscle = StringVar()
        muscle_entry = ctk.CTkComboBox(
            self.main,
            width=10,
            values=["VL", "RF", "GM", "GL", "BF"],
            state="readonly",
            variable=self.muscle,
        )
        muscle_entry.grid(column=2, row=8, sticky=(W, E))
        self.muscle.set("RF")

        # Buttons
        # Input directory
        input_button = ctk.CTkButton(self.main, text="Input", command=self.get_root_dir)
        input_button.grid(column=5, row=2, sticky=E)

        # Model path button
        model_button = ctk.CTkButton(
            self.main, text="Model", command=self.get_model_path
        )
        model_button.grid(column=5, row=3, sticky=E)

        # Break Button
        break_button = ctk.CTkButton(self.main, text="Break", command=self.do_break)
        break_button.grid(column=1, row=17, sticky=W)

        # Run Button
        run_button = ctk.CTkButton(self.main, text="Run", command=self.run_code)
        run_button.grid(column=2, row=17, sticky=(W, E))

        # Advanced button with style
        style.configure(
            "B.TButton",
            background="black",
            foreground="white",
            font=("Lucida Sans", 11),
        )
        advanced_button = ctk.CTkButton(
            self.main,
            text="Advanced Methods",
            command=self.advanced_methods,
            # style="B.TButton",
        )
        advanced_button.grid(column=5, row=17, sticky=E)

        # Labels
        ctk.CTkLabel(self.main, text="Directories", font=("Verdana", 14)).grid(
            column=1, row=1, sticky=W
        )
        ctk.CTkLabel(self.main, text="Root Directory").grid(column=1, row=2)
        ctk.CTkLabel(self.main, text="Model Path").grid(column=1, row=3)
        ctk.CTkLabel(self.main, text="Loss Function").grid(column=1, row=4)
        ctk.CTkLabel(self.main, text="Image Properties", font=("Verdana", 14)).grid(
            column=1, row=6, sticky=W
        )
        ctk.CTkLabel(self.main, text="Scaling Type").grid(column=1, row=7)
        ctk.CTkLabel(self.main, text="Muscle").grid(column=1, row=8)
        ctk.CTkLabel(self.main, text="Muscle Volume", font=("Verdana", 14)).grid(
            column=1, row=13, sticky=W
        )
        ctk.CTkLabel(self.main, text="Volume Calculation").grid(column=1, row=14)

        # Separators
        ttk.Separator(self.main, orient="horizontal", style="TSeparator").grid(
            column=0, row=5, columnspan=9, sticky=(W, E)
        )
        ttk.Separator(self.main, orient="horizontal", style="TSeparator").grid(
            column=0, row=12, columnspan=9, sticky=(W, E)
        )
        ttk.Separator(self.main, orient="horizontal", style="TSeparator").grid(
            column=0, row=16, columnspan=9, sticky=(W, E)
        )

        for child in self.main.winfo_children():
            child.grid_configure(padx=5, pady=5)

        self.bind("<Return>", self.run_code)  # execute by pressing return

    # --------------------------------------------------------------------------------------------------
    # Functionalities used in GUI

    def get_root_dir(self):
        """
        Instance method to ask the user to select the input directory.
        All image files (of the same specified filetype) in
        the input directory are analysed.
        """
        root_dir = filedialog.askdirectory()
        self.input.set(root_dir)
        return root_dir

    def get_model_path(self):
        """
        Instance method to ask the user to select the apo model path.
        This must be an absolute path and the model must be a .h5 file.
        """
        model_dir = filedialog.askopenfilename()
        self.model.set(model_dir)
        return model_dir

    def on_volume_change(self, *args):
        """
        Instance method to adpat GUI layout based on volume
        calculation selection.
        """
        if self.muscle_volume_calculation_wanted.get() == "Yes":
            # Distance between ACSA for Volume Calculation
            self.volume_label = ttk.Label(self.main, text="Slice Distance (cm)")
            self.volume_label.grid(column=1, row=15)

            self.distance = StringVar()
            self.distance_entry = ttk.Entry(
                self.main, width=10, textvariable=self.distance
            )
            self.distance_entry.grid(column=2, row=15, sticky=(W, E))
            self.distance.set(7)

        if self.muscle_volume_calculation_wanted.get() == "No":
            # Destroy widget on selection
            if hasattr(self, "distance"):
                self.distance_entry.grid_remove()
                self.volume_label.grid_remove()

    def on_scaling_change(self, *args):
        """
        Instance method to adpat GUI layout based on scaling
        calculation selection.
        """
        if self.scaling.get() == "Bar":
            # Spacing
            self.spacing_label = ttk.Label(self.main, text="Spacing (mm)")
            self.spacing_label.grid(column=1, row=10)
            self.spacing = StringVar()
            spacing = (5, 10, 15, 20)
            self.spacing_entry = ttk.Combobox(
                self.main, width=10, textvariable=self.spacing
            )
            self.spacing_entry["values"] = spacing
            self.spacing_entry["state"] = "readonly"
            self.spacing_entry.grid(column=2, row=10, sticky=(W, E))
            self.spacing.set(10)

            if hasattr(self, "depth"):
                self.depth_label.grid_remove()
                self.depth_entry.grid_remove()

        elif self.scaling.get() == "Line":
            # Image Depth
            self.depth_label = ttk.Label(self.main, text="Depth (cm)")
            self.depth_label.grid(column=1, row=9)
            self.depth = StringVar()
            depth = (2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8)
            self.depth_entry = ttk.Combobox(
                self.main, width=10, textvariable=self.depth
            )
            self.depth_entry["values"] = depth
            # depth_entry["state"] = "readonly"
            self.depth_entry.grid(column=2, row=9, sticky=(W, E))
            self.depth.set(5.5)

            if hasattr(self, "spacing"):
                self.spacing_label.grid_remove()
                self.spacing_entry.grid_remove()

        elif self.scaling.get() == "Manual":

            if hasattr(self, "spacing"):
                self.spacing_label.grid_remove()
                self.spacing_entry.grid_remove()

            if hasattr(self, "depth"):
                self.depth_label.grid_remove()
                self.depth_entry.grid_remove()

    # ---------------------------------------------------------------------------------------------------
    # Open new toplevel instance for advanced methods
    def advanced_methods(self):
        """
        Function to open a toplevel where masks
        can either be created for training purposes or
        can be inspected subsequent to labelling.
        """
        # Open Window
        self.advanced_window = tk.Toplevel(bg="#7ABAA1")
        self.advanced_window.title("Advanced Methods Window")
        master_path = os.path.dirname(os.path.abspath(__file__))
        iconpath = master_path + "/gui_helpers/icon.ico"
        self.advanced_window.iconbitmap(iconpath)
        self.advanced_window.grab_set()

        ttk.Label(self.advanced_window, text="Select Method").grid(column=1, row=0)

        # Mask Option
        self.advanced_option = StringVar()
        advanced_entry = ttk.Combobox(
            self.advanced_window, width=20, textvariable=self.advanced_option
        )
        advanced_entry["values"] = ("Train Model", "Create Masks", "Inspect Masks")
        advanced_entry["state"] = "readonly"
        advanced_entry.grid(column=1, row=1, sticky=(W, E))
        self.advanced_option.set(" ")
        self.advanced_option.trace("w", self.on_mask_change)

        # Add padding
        for child in self.advanced_window.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def on_mask_change(self, *args):
        """
        Depending on which mask opration is selected,
        this function adapts the GUI.

        Instance method to open new window for model training.
        The window is opened upon pressing of the "analysis parameters"
        button.

        Several parameters are displayed.
        - Image Directory:
        The user must select or input the image directory. This
        path must to the directory containing the training images.
        Images must be in RGB format.
        - Mask Directory:
        The user must select or input the mask directory. This
        path must to the directory containing the training images.
        Masks must be binary.
        - Output Directory:
        The user must select or input the mask directory. This
        path must lead to the directory where the trained model
        and the model weights should be saved.
        - Batch Size:
        The user must input the batch size used during model training by
        selecting from the dropdown list or entering a value.
        Although a larger batch size has advantages during model trainig,
        the images used here are large. Thus, the larger the batch size,
        the more compute power is needed or the longer the training duration.
        Integer, must be non-negative and non-zero.
        - Learning Rate:
        The user must enter the learning rate used for model training by
        selecting from the dropdown list or entering a value.
        Float, must be non-negative and non-zero.
        - Epochs:
        The user must enter the number of Epochs used during model training by
        selecting from the dropdown list or entering a value.
        The total amount of epochs will only be used if early stopping does not happen.
        Integer, must be non-negative and non-zero.
        - Loss Function:
        The user must enter the loss function used for model training by
        selecting from the dropdown list. These can be "BCE" (binary
        cross-entropy), "Dice" (Dice coefficient) or "FL"(Focal loss).

        Model training is started by pressing the "start training" button. Although
        all parameters relevant for model training can be adapted, we advise users with
        limited experience to keep the pre-defined settings. These settings are best
        practice and devised from the original papers that proposed the models used
        here. Singularly the batch size should be adapted to 1 if comupte power is limited
        (no GPU or GPU with RAM lower than 8 gigabyte).

        There is an "Augment Images" button, which allows to generate new training images.
        The images and masks for the data augmentation are taken from the chosen image directory
        and mask directory. The new images are saved under the same directories.
        """
        # make new frame
        self.advanced_window_frame = ttk.Frame(
            self.advanced_window, padding="10 10 12 12"
        )
        self.advanced_window_frame.grid(column=1, row=2, sticky=(N, S, W, E))

        try:

            if self.advanced_option.get() == "Inspect Masks":

                if hasattr(self, "mask_button"):
                    self.mask_button.destroy()

                # Train image directory
                self.raw_image_dir = StringVar()
                image_entry = ttk.Entry(
                    self.advanced_window_frame,
                    width=30,
                    textvariable=self.raw_image_dir,
                )
                image_entry.grid(column=1, row=2, columnspan=2, sticky=(W, E))
                self.raw_image_dir.set("Select Raw Image Directory")

                dir1_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Image Dir",
                    command=lambda: (self.raw_image_dir.set(filedialog.askdirectory())),
                )
                dir1_button.grid(column=3, row=2, sticky=(W, E))

                # Mask directory
                self.mask_image_dir = StringVar()
                self.mask_image_entry = ttk.Entry(
                    self.advanced_window_frame,
                    width=30,
                    textvariable=self.mask_image_dir,
                )
                self.mask_image_entry.grid(column=1, row=3, columnspan=2, sticky=(W, E))
                self.mask_image_dir.set("Select Mask Image Directory")

                self.dir2_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Mask Dir",
                    command=lambda: (
                        self.mask_image_dir.set(filedialog.askdirectory())
                    ),
                )
                self.dir2_button.grid(column=3, row=3, sticky=(W, E))

                # Start index
                self.start_label = ttk.Label(
                    self.advanced_window_frame, text="Start at Image:"
                )
                self.start_label.grid(column=1, row=4, sticky=W)
                start_idx = StringVar()
                self.idx = ttk.Entry(
                    self.advanced_window_frame, width=10, textvariable=start_idx
                )
                self.idx.grid(column=2, row=4, sticky=W)
                start_idx.set("0")

                # Inspect button
                self.inspect_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Inspect Masks",
                    command=lambda: (
                        self.advanced_window.destroy(),
                        gui_helpers.find_outliers(
                            dir1=self.raw_image_dir.get(),
                            dir2=self.mask_image_dir.get(),
                        ),
                        gui_helpers.overlay_directory_images(
                            image_dir=self.raw_image_dir.get(),
                            mask_dir=self.mask_image_dir.get(),
                            start_index=int(start_idx.get()),
                        ),
                    ),
                )
                self.inspect_button.grid(column=2, row=5, sticky=(W, E))

            elif self.advanced_option.get() == "Create Masks":

                # Forget widgets
                if hasattr(self, "train_image_dir"):
                    for widget in self.advanced_window_frame.winfo_children():
                        widget.destroy()

                if hasattr(self, "dir2_button"):
                    self.dir2_button.destroy()
                    self.mask_image_entry.destroy()
                    self.inspect_button.destroy()
                    self.idx.destroy()
                    self.start_label.destroy()

                # Train image directory
                self.raw_image_dir = StringVar()
                image_entry = ttk.Entry(
                    self.advanced_window_frame,
                    width=30,
                    textvariable=self.raw_image_dir,
                )
                image_entry.grid(column=1, row=2, columnspan=2, sticky=(W, E))
                self.raw_image_dir.set("Select Raw Image Directory")

                dir1_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Image Dir",
                    command=lambda: (self.raw_image_dir.set(filedialog.askdirectory())),
                )
                dir1_button.grid(column=3, row=2, sticky=(W, E))

                # Train image directory
                self.raw_image_dir = StringVar()
                mask_entry = ttk.Entry(
                    self.advanced_window_frame,
                    width=30,
                    textvariable=self.raw_image_dir,
                )
                mask_entry.grid(column=1, row=2, columnspan=2, sticky=(W, E))
                self.raw_image_dir.set("Select Raw Image Directory")

                # Mask button
                self.mask_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Create Masks",
                    command=lambda: (
                        self.advanced_window.destroy(),
                        gui_helpers.create_acsa_masks(
                            input_dir=self.raw_image_dir.get(), muscle_name="image"
                        ),
                    ),
                )
                self.mask_button.grid(column=2, row=3, sticky=(W, E))

            elif self.advanced_option.get() == "Train Model":
                # Labels
                ttk.Label(
                    self.advanced_window_frame,
                    text="Training Directories",
                    font=("Verdana", 14),
                ).grid(column=1, row=0, padx=10)
                ttk.Label(self.advanced_window_frame, text="Image Directory").grid(
                    column=1, row=2
                )
                ttk.Label(self.advanced_window_frame, text="Mask Directory").grid(
                    column=1, row=3
                )
                ttk.Label(self.advanced_window_frame, text="Output Directory").grid(
                    column=1, row=4
                )

                ttk.Label(
                    self.advanced_window_frame,
                    text="Hyperparameters",
                    font=("Verdana", 14),
                ).grid(column=1, row=6, padx=10)
                ttk.Label(self.advanced_window_frame, text="Batch Size").grid(
                    column=1, row=7
                )
                ttk.Label(self.advanced_window_frame, text="Learning Rate").grid(
                    column=1, row=8
                )
                ttk.Label(self.advanced_window_frame, text="Epochs").grid(
                    column=1, row=9
                )
                ttk.Label(self.advanced_window_frame, text="Loss Function").grid(
                    column=1, row=10
                )

                # Entryboxes
                # Train image directory
                self.train_image_dir = StringVar()
                train_image_entry = ttk.Entry(
                    self.advanced_window_frame,
                    width=30,
                    textvariable=self.train_image_dir,
                )
                train_image_entry.grid(column=2, row=2, columnspan=3, sticky=(W, E))
                self.train_image_dir.set("C:/Users/admin/Documents/DeepACSA")

                # Mask directory
                self.mask_dir = StringVar()
                mask_entry = ttk.Entry(
                    self.advanced_window_frame, width=30, textvariable=self.mask_dir
                )
                mask_entry.grid(column=2, row=3, columnspan=3, sticky=(W, E))
                self.mask_dir.set("C:/Users/admin/Documents/DeepACSA")

                # Output path
                self.out_dir = StringVar()
                out_entry = ttk.Entry(
                    self.advanced_window_frame, width=30, textvariable=self.out_dir
                )
                out_entry.grid(column=2, row=4, columnspan=3, sticky=(W, E))
                self.out_dir.set("C:/Users/admin/Documents/DeepACSA")

                # Buttons
                # Train image button
                train_img_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Images",
                    command=self.get_train_dir,
                )
                train_img_button.grid(column=5, row=2, sticky=E)

                # Mask button
                mask_button = ttk.Button(
                    self.advanced_window_frame, text="Masks", command=self.get_mask_dir
                )
                mask_button.grid(column=5, row=3, sticky=E)

                # Data augmentation button
                data_augmentation_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Augment Images",
                    command=self.augment_images,
                )
                data_augmentation_button.grid(column=4, row=12, sticky=E)

                # Input directory
                out_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Output",
                    command=self.get_output_dir,
                )
                out_button.grid(column=5, row=4, sticky=E)

                # Model train button
                model_button = ttk.Button(
                    self.advanced_window_frame,
                    text="Start Training",
                    command=self.train_model,
                )
                model_button.grid(column=5, row=12, sticky=E)

                # Comboboxes
                # Batch size
                self.batch_size = StringVar()
                size = ("1", "2", "3", "4", "5", "6")
                size_entry = ttk.Combobox(
                    self.advanced_window_frame, width=10, textvariable=self.batch_size
                )
                size_entry["values"] = size
                size_entry.grid(column=2, row=7, sticky=(W, E))
                self.batch_size.set("1")

                # Learning rate
                self.learn_rate = StringVar()
                learn = ("0.005", "0.001", "0.0005", "0.0001", "0.00005", "0.00001")
                learn_entry = ttk.Combobox(
                    self.advanced_window_frame, width=10, textvariable=self.learn_rate
                )
                learn_entry["values"] = learn
                learn_entry.grid(column=2, row=8, sticky=(W, E))
                self.learn_rate.set("0.00001")

                # Number of training epochs
                self.epochs = StringVar()
                epoch = ("30", "40", "50", "60", "70", "80")
                epoch_entry = ttk.Combobox(
                    self.advanced_window_frame, width=10, textvariable=self.epochs
                )
                epoch_entry["values"] = epoch
                epoch_entry.grid(column=2, row=9, sticky=(W, E))
                self.epochs.set("3")

                # Loss function
                self.loss_function = StringVar()
                loss = ("BCE", "Dice", "FL")
                loss_entry = ttk.Combobox(
                    self.advanced_window_frame,
                    width=10,
                    textvariable=self.loss_function,
                )
                loss_entry["values"] = loss
                loss_entry["state"] = "readonly"
                loss_entry.grid(column=2, row=10, sticky=(W, E))
                self.loss_function.set("BCE")

                # Seperators
                ttk.Separator(
                    self.advanced_window_frame, orient="horizontal", style="TSeparator"
                ).grid(column=0, row=5, columnspan=9, sticky=(W, E))
                ttk.Separator(
                    self.advanced_window_frame, orient="horizontal", style="TSeparator"
                ).grid(column=0, row=11, columnspan=9, sticky=(W, E))

        except FileNotFoundError:
            tk.messagebox.showerror("Information", "Enter the coorect folder path!")

        # Add padding
        for child in self.advanced_window_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

    # ---------------------------------------------------------------------------------------------------
    ## Methods used for model training

    def get_train_dir(self):
        """
        Instance method to ask the user to select the training image
        directory path. All image files (of the same specified filetype) in
        the directory are analysed. This must be an absolute path.
        """
        train_image_dir = filedialog.askdirectory()
        self.train_image_dir.set(train_image_dir)

    def get_mask_dir(self):
        """
        Instance method to ask the user to select the training mask
        directory path. All mask files (of the same specified filetype) in
        the directory are analysed.The mask files and the corresponding
        image must have the exact same name. This must be an absolute path.
        """
        mask_dir = filedialog.askdirectory()
        self.mask_dir.set(mask_dir)

    def get_output_dir(self):
        """
        Instance method to ask the user to select the output
        directory path. Here, all file created during model
        training (model file, weight file, graphs) are saved.
        This must be an absolute path.
        """
        out_dir = filedialog.askdirectory()
        self.out_dir.set(out_dir)

    def train_model(self):
        """
        Instance method to execute the model training when the
        "start training" button is pressed.

        By pressing the button, a seperate thread is started
        in which the model training is run. This allows the user to break any
        training process at certain stages. When the analysis can be
        interrupted, a tk.messagebox opens asking the user to either
        continue or terminate the analysis. Moreover, the threading allows interaction
        with the GUI during ongoing analysis process.
        """
        try:
            # See if GUI is already running
            if self.is_running:
                # don't run again if it is already running
                return
            self.is_running = True

            # Get input paremeter
            selected_images = self.train_image_dir.get()
            selected_masks = self.mask_dir.get()
            selected_outpath = self.out_dir.get()

            # Make sure some kind of filetype is specified.
            if (
                len(selected_images) < 3
                or len(selected_masks) < 3
                or len(selected_outpath) < 3
            ):
                tk.messagebox.showerror("Information", "Specified directories invalid.")
                self.should_stop = False
                self.is_running = False
                self.do_break()
                return

            selected_batch_size = int(self.batch_size.get())
            selected_learning_rate = float(self.learn_rate.get())
            selected_epochs = int(self.epochs.get())
            selected_loss_function = self.loss_function.get()

            # Start thread
            thread = Thread(
                target=gui_helpers.trainModel,
                args=(
                    selected_images,
                    selected_masks,
                    selected_outpath,
                    selected_batch_size,
                    selected_learning_rate,
                    selected_epochs,
                    selected_loss_function,
                    self,
                ),
            )

            thread.start()

        # Error handling
        except ValueError:
            tk.messagebox.showerror(
                "Information", "Analysis parameter entry fields" + " must not be empty."
            )
            self.do_break()
            self.should_stop = False
            self.is_running = False

    ## Method used for data augmentation

    def augment_images(self):
        """
        Instance method to augment input images, when the "Augment Images" button is pressed.
        Input parameters for the gui_helpers.image_augmentation function are taken from the chosen
        image and mask directories. The newly generated data will be saved under the same
        directories.
        """
        try:
            # See if GUI is already running
            if self.is_running:
                # don't run again if it is already running
                return
            self.is_running = True

            # Get input paremeters
            selected_images = self.train_image_dir.get()
            selected_masks = self.mask_dir.get()

            # Make sure some kind of filetype is specified.
            if len(selected_images) < 3 or len(selected_masks) < 3:
                tk.messagebox.showerror("Information", "Specified directories invalid.")
                self.should_stop = False
                self.is_running = False
                self.do_break()
                return

            gui_helpers.image_augmentation(selected_images, selected_masks, self)

        # Error handling
        except ValueError:
            tk.messagebox.showerror(
                "Information",
                "Check input parameters"
                + "\nPotential error source: Invalid directories",
            )
            self.do_break()
            self.should_stop = False
            self.is_running = False

    # ---------------------------------------------------------------------------------------------------
    # Methods to run the code

    def run_code(self):
        """
        Instance method to execute the analysis process when the
        "run" button is pressed.

        Which analysis process is executed depends on the user
        selection. By pressing the button, a seperate thread is started
        in which the analysis is run. This allows the user to break any
        analysis process. Moreover, the threading allows interaction
        with the main GUI during ongoing analysis process. This function
        handles most of the errors occuring during specification of
        file and analysis parameters. All other exeptions are raised in
        other function of this package.

        Raises
        ------
        AttributeError
            The execption is raised when the user didn't specify the
            file or training parameters correctly. A tk.messagebox
            is openend containing hints how to solve the issue.
        FileNotFoundError
            The execption is raised when the user didn't specify the
            file or training parameters correctly. A tk.messagebox
            is openend containing hints how to solve the issue.
        PermissionError
            The execption is raised when the user didn't specify the
            file or training parameters correctly. A tk.messagebox
            is openend containing hints how to solve the issue.
        """
        try:
            if self.is_running:
                # don't run again if it is already running
                return
            self.is_running = True

            selected_input_dir = self.input.get()
            selected_model_path = self.model.get()
            selected_loss = self.loss_function.get()
            selected_muscle = self.muscle.get()
            selected_scaling = self.scaling.get()
            selected_volume_calculation = self.muscle_volume_calculation_wanted.get()

            # Use distance for volumne only if selected
            if selected_volume_calculation == "Yes":
                distance_acsa = float(self.distance.get())
            else:
                distance_acsa = 1

            if len(selected_input_dir) == 0:
                tk.messagebox.showerror(
                    "Information",
                    "Check input parameters."
                    + "\nPotential error source:  Invalid specified input directory",
                )
                self.should_stop = False
                self.is_running = False
                self.do_break()
                return

            elif len(selected_model_path) == 0:
                tk.messagebox.showerror(
                    "Information",
                    "Check input parameters."
                    + "\nPotential error source:  Invalid specified model path",
                )
                self.should_stop = False
                self.is_running = False
                self.do_break()
                return

            elif selected_loss == "Loss Function":
                tk.messagebox.showerror(
                    "Information",
                    "Check model loss function."
                    + "\nPotential error source:  Invalid specified loss",
                )
                self.should_stop = False
                self.is_running = False
                self.do_break()
                return

            if selected_scaling == "Line":
                selected_depth = float(self.depth.get())

                # Catch depth error.
                if selected_depth <= 0:
                    tk.messagebox.showerror(
                        "Information",
                        "Check input parameters."
                        + "\nPotential error source:  Invalid specified depth.",
                    )
                    self.should_stop = False
                    self.is_running = False
                    self.do_break()
                    return

                thread = Thread(
                    target=gui_helpers.calculate_batch_efov,
                    args=(
                        selected_input_dir,
                        selected_model_path,
                        selected_loss,
                        selected_depth,
                        selected_muscle,
                        selected_volume_calculation,
                        distance_acsa,
                        self,
                    ),
                )
            else:
                if selected_scaling == "Bar":
                    selected_spacing = float(self.spacing.get())
                else:
                    selected_spacing = 0
                thread = Thread(
                    target=gui_helpers.calculate_batch,
                    args=(
                        selected_input_dir,
                        selected_model_path,
                        selected_loss,
                        selected_spacing,
                        selected_muscle,
                        selected_scaling,
                        selected_volume_calculation,
                        distance_acsa,
                        self,
                    ),
                )

            thread.start()

        except ValueError:
            tk.messagebox.showerror(
                "Information",
                "Check input parameters."
                + "\nPotential error source:  Invalid specified depth, distance or volume",
            )
            self.should_stop = False
            self.is_running = False
            self.do_break()

    @property
    def should_stop(self):
        """
        Instance method to define the should_stop
        property getter method. By defining this as a property,
        should_stop is treated like a public attribute even
        though it is private.

        This is used to stop the analysis process running
        in a seperate thread.

        Returns
        -------
        should_stop : bool
            Boolean variable to decide whether the analysis process
            started from the GUI should be stopped. The process is
            stopped when should_stop = True.
        """
        self._lock.acquire()
        should_stop = self._should_stop
        self._lock.release()
        return should_stop

    @property
    def is_running(self):
        """
        Instance method to define the is_running
        property getter method. By defining this as a property,
        is_running is treated like a public attribute even
        though it is private.

        This is used to stop the analysis process running
        in a seperate thread.

        Returns
        -------
        is_running : bool
            Boolean variable to check whether the analysis process
            started from the GUI is running. The process is only
            stopped when is_running = True.
        """
        self._lock.acquire()
        is_running = self._is_running
        self._lock.release()
        return is_running

    @should_stop.setter
    def should_stop(self, flag: bool):
        """
        Instance method to define the should_stop
        property setter method. The setter method is used
        to set the self._should_stop attribute as if it was
        a public attribute. The argument "flag" is thereby
        validated to ensure proper input (boolean)
        """
        self._lock.acquire()
        self._should_stop = flag
        self._lock.release()

    @is_running.setter
    def is_running(self, flag: bool):
        """
        Instance method to define the is_running
        property setter method. The setter method is used
        to set the self._is_running attribute as if it was
        a public attribute. The argument "flag" is thereby
        validated to ensure proper input (boolean)
        """
        self._lock.acquire()
        self._is_running = flag
        self._lock.release()

    def do_break(self):
        """
        Instance method to break the analysis process when the
        button "break" is pressed.

        This changes the instance attribute self.should_stop
        to True, given that the analysis is already running.
        The attribute is checked befor every iteration
        of the analysis process.
        """
        if self.is_running:
            self.should_stop = True


# ---------------------------------------------------------------------------------------------------
# Function required to run the GUI frm the prompt


def runMain() -> None:
    """
    Function that enables usage of the gui from command promt
    as pip package.

    Notes
    -----
    The GUI can be executed by typing 'python -m deep_acsa_gui.py' in the command
    subsequtently to installing the pip package and activating the
    respective library.

    It is not necessary to download any files from the repository when the pip
    package is installed.

    For documentation of DL_Track see top of this module.
    """
    app = DeepACSA()
    app.mainloop()


# This statement is required to execute the GUI by typing 'python deep_acsa_gui.py' in the prompt
# when navigated to the folder containing the file and all dependencies.
if __name__ == "__main__":
    app = DeepACSA()
    app.mainloop()
