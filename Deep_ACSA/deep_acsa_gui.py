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

import tkinter as tk
from threading import Lock, Thread
from tkinter import E, N, S, StringVar, Tk, W, filedialog, ttk

from Deep_ACSA import gui_helpers


class DeepACSA:
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

    def __init__(self, root):

        # set up threading
        self._lock = Lock()
        self._is_running = False
        self._should_stop = False

        # set up gui
        root.title("DeepACSA")

        self.main = ttk.Frame(root, padding="10 10 12 12")
        self.main.grid(column=0, row=0, sticky=(N, S, W, E))
        # Configure resizing of user interface
        self.main.columnconfigure(0, weight=1)
        self.main.columnconfigure(1, weight=1)
        self.main.columnconfigure(2, weight=1)
        self.main.columnconfigure(3, weight=1)
        self.main.columnconfigure(4, weight=1)
        self.main.columnconfigure(5, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="SkyBlue4")
        style.configure(
            "TLabel",
            font=("Lucida Sans", 12),
            foreground="black",
            background="SkyBlue4",
        )
        style.configure(
            "TRadiobutton",
            background="SkyBlue4",
            foreground="black",
            font=("Lucida Sans", 12),
        )
        style.configure(
            "TButton", background="linen", foreground="black", font=("Lucida Sans", 11)
        )
        style.configure(
            "TEntry", font=("Lucida Sans", 12), background="linen", foregrund="black"
        )
        style.configure("TCombobox", background="SkyBlue4", foreground="black")

        # Paths
        # Input directory
        self.input = StringVar()
        input_entry = ttk.Entry(self.main, width=30, textvariable=self.input)
        input_entry.grid(column=2, row=2, columnspan=3, sticky=(W, E))
        # self.input.set("Desktop/DeepACSA/")
        # Model path
        self.model = StringVar()
        model_entry = ttk.Entry(self.main, width=30, textvariable=self.model)
        model_entry.grid(column=2, row=3, columnspan=3, sticky=(W, E))
        # self.model.set("C:/Users/admin/Documents/DeepACSA/notebooks/VGG16pre-VL-256.h5")

        # Radiobuttons
        # Image Type
        self.scaling = StringVar()
        efov = ttk.Radiobutton(
            self.main, text="Line", variable=self.scaling, value="Line"
        )
        efov.grid(column=2, row=7, sticky=W)
        static = ttk.Radiobutton(
            self.main, text="Bar", variable=self.scaling, value="Bar"
        )
        static.grid(column=3, row=7, sticky=(W, E))
        manual = ttk.Radiobutton(
            self.main, text="Manual", variable=self.scaling, value="Manual"
        )
        manual.grid(column=4, row=7, sticky=E)
        self.scaling.set("Bar")

        # Volume Calculation
        self.muscle_volume_calculation_wanted = StringVar()
        yes_volume = ttk.Radiobutton(
            self.main,
            text="Yes",
            variable=self.muscle_volume_calculation_wanted,
            value="Yes",
        )
        yes_volume.grid(column=2, row=14, sticky=W)
        no_volume = ttk.Radiobutton(
            self.main,
            text="No",
            variable=self.muscle_volume_calculation_wanted,
            value="No",
        )
        no_volume.grid(column=3, row=14, sticky=(W, E))
        self.muscle_volume_calculation_wanted.set("No")

        # Comboboxes
        # Loss Function
        self.loss_function = StringVar()
        loss = ("IoU", "Dice Loss", "Focal Loss")
        loss_entry = ttk.Combobox(self.main, width=15, textvariable=self.loss_function)
        loss_entry["values"] = loss
        loss_entry["state"] = "readonly"
        loss_entry.grid(column=4, row=4, sticky=E)
        self.loss_function.set("Loss Function")

        # Filetype
        self.filetype = StringVar()
        filetype = (
            "/**/*.tif",
            "/**/*.tiff",
            "/**/*.png",
            "/**/*.bmp",
            "/**/*.jpeg",
            "/**/*.jpg",
        )
        filetype_entry = ttk.Combobox(self.main, width=10, textvariable=self.filetype)
        filetype_entry["values"] = filetype
        # filetype_entry["state"] = "readonly"
        filetype_entry.grid(column=2, row=6, sticky=E)
        self.filetype.set("/**/*.jpg")

        # Muscles
        self.muscle = StringVar()
        muscle = ("VL", "RF", "GM", "GL", "BF")
        muscle_entry = ttk.Combobox(self.main, width=10, textvariable=self.muscle)
        muscle_entry["values"] = muscle
        muscle_entry["state"] = "readonly"
        muscle_entry.grid(column=2, row=8, sticky=(W, E))
        self.muscle.set("RF")

        # Image Depth
        self.depth = StringVar()
        depth = (2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8)
        depth_entry = ttk.Combobox(self.main, width=10, textvariable=self.depth)
        depth_entry["values"] = depth
        # depth_entry["state"] = "readonly"
        depth_entry.grid(column=2, row=9, sticky=(W, E))
        self.depth.set(5.5)

        # Spacing
        self.spacing = StringVar()
        spacing = (5, 10, 15, 20)
        spacing_entry = ttk.Combobox(self.main, width=10, textvariable=self.spacing)
        spacing_entry["values"] = spacing
        spacing_entry["state"] = "readonly"
        spacing_entry.grid(column=2, row=10, sticky=(W, E))
        self.spacing.set(10)

        # Distance between ACSA for Volume Calculation
        self.distance = StringVar()
        distance_entry = ttk.Entry(self.main, width=10, textvariable=self.distance)
        distance_entry.grid(column=2, row=15, sticky=(W, E))
        self.distance.set(7)

        # Buttons
        # Input directory
        input_button = ttk.Button(self.main, text="Input", command=self.get_root_dir)
        input_button.grid(column=5, row=2, sticky=E)

        # Model path
        model_button = ttk.Button(self.main, text="Model", command=self.get_model_path)
        model_button.grid(column=5, row=3, sticky=E)

        # Break Button
        break_button = ttk.Button(self.main, text="Break", command=self.do_break)
        break_button.grid(column=1, row=16, sticky=W)
        # Run Button
        run_button = ttk.Button(self.main, text="Run", command=self.run_code)
        run_button.grid(column=2, row=16, sticky=(W, E))
        # Train Button
        train_button = ttk.Button(
            self.main, text="Train Model", command=self.train_model_window
        )
        train_button.grid(column=5, row=16, sticky=(W, E))

        # Labels
        ttk.Label(self.main, text="Directories", font=("Verdana", 14)).grid(
            column=1, row=1, sticky=W
        )
        ttk.Label(self.main, text="Root Directory").grid(column=1, row=2)
        ttk.Label(self.main, text="Model Path").grid(column=1, row=3)
        ttk.Label(self.main, text="Image Properties", font=("Verdana", 14)).grid(
            column=1, row=5, sticky=W
        )
        ttk.Label(self.main, text="Image Type").grid(column=1, row=6)
        ttk.Label(self.main, text="Scaling Type").grid(column=1, row=7)
        ttk.Label(self.main, text="Muscle").grid(column=1, row=8)
        ttk.Label(self.main, text="Depth (cm)").grid(column=1, row=9)
        ttk.Label(self.main, text="Spacing (mm)").grid(column=1, row=10)
        ttk.Label(self.main, text="Muscle Volume", font=("Verdana", 14)).grid(
            column=1, row=13, sticky=W
        )
        ttk.Label(self.main, text="Volume Calculation").grid(column=1, row=14)
        ttk.Label(self.main, text="Distance (cm)").grid(column=1, row=15)

        for child in self.main.winfo_children():
            child.grid_configure(padx=5, pady=5)

        root.bind("<Return>", self.run_code)  # execute by pressing return

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
            selected_filetype = self.filetype.get()
            selected_muscle = self.muscle.get()
            selected_depth = float(self.depth.get())
            selected_spacing = self.spacing.get()
            selected_scaling = self.scaling.get()
            selected_volume_calculation = self.muscle_volume_calculation_wanted.get()
            distance_acsa = float(self.distance.get())

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

            elif len(selected_filetype) < 3:
                tk.messagebox.showerror(
                    "Information",
                    "Check input parameters."
                    + "\nPotential error source:  Invalid specified filetype",
                )
                self.should_stop = False
                self.is_running = False
                self.do_break()
                return

            if selected_scaling == "Line":
                thread = Thread(
                    target=gui_helpers.calculate_batch_efov,
                    args=(
                        selected_input_dir,
                        selected_filetype,
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
                thread = Thread(
                    target=gui_helpers.calculate_batch,
                    args=(
                        selected_input_dir,
                        selected_filetype,
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
                + "\nPotential error source:  Invalid specified depth or distance",
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
    # Open new toplevel instance for model training

    def train_model_window(self):
        """
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
        # Open Window
        window = tk.Toplevel(bg="SkyBlue4")
        window.title("Model Training Window")
        window.grab_set()

        # Labels
        ttk.Label(window, text="Training Parameters", font=("Verdana", 14)).grid(
            column=1, row=0, padx=10
        )
        ttk.Label(window, text="Image Directory").grid(column=1, row=2)
        ttk.Label(window, text="Mask Directory").grid(column=1, row=3)
        ttk.Label(window, text="Output Directory").grid(column=1, row=4)
        ttk.Label(window, text="Batch Size").grid(column=1, row=5)
        ttk.Label(window, text="Learning Rate").grid(column=1, row=6)
        ttk.Label(window, text="Epochs").grid(column=1, row=7)
        ttk.Label(window, text="Loss Function").grid(column=1, row=8)

        # Entryboxes
        # Train image directory
        self.train_image_dir = StringVar()
        train_image_entry = ttk.Entry(
            window, width=30, textvariable=self.train_image_dir
        )
        train_image_entry.grid(column=2, row=2, columnspan=3, sticky=(W, E))
        self.train_image_dir.set("C:/Users/admin/Documents/DeepACSA")

        # Mask directory
        self.mask_dir = StringVar()
        mask_entry = ttk.Entry(window, width=30, textvariable=self.mask_dir)
        mask_entry.grid(column=2, row=3, columnspan=3, sticky=(W, E))
        self.mask_dir.set("C:/Users/admin/Documents/DeepACSA")

        # Output path
        self.out_dir = StringVar()
        out_entry = ttk.Entry(window, width=30, textvariable=self.out_dir)
        out_entry.grid(column=2, row=4, columnspan=3, sticky=(W, E))
        self.out_dir.set("C:/Users/admin/Documents/DeepACSA")

        # Buttons
        # Train image button
        train_img_button = ttk.Button(window, text="Images", command=self.get_train_dir)
        train_img_button.grid(column=5, row=2, sticky=E)

        # Mask button
        mask_button = ttk.Button(window, text="Masks", command=self.get_mask_dir)
        mask_button.grid(column=5, row=3, sticky=E)

        # Data augmentation button
        data_augmentation_button = ttk.Button(
            window, text="Augment Images", command=self.augment_images
        )
        data_augmentation_button.grid(column=4, row=10, sticky=E)

        # Input directory
        out_button = ttk.Button(window, text="Output", command=self.get_output_dir)
        out_button.grid(column=5, row=4, sticky=E)

        # Model train button
        model_button = ttk.Button(
            window, text="Start Training", command=self.train_model
        )
        model_button.grid(column=5, row=10, sticky=E)

        # Comboboxes
        # Batch size
        self.batch_size = StringVar()
        size = ("1", "2", "3", "4", "5", "6")
        size_entry = ttk.Combobox(window, width=10, textvariable=self.batch_size)
        size_entry["values"] = size
        size_entry.grid(column=2, row=5, sticky=(W, E))
        self.batch_size.set("1")

        # Learning rate
        self.learn_rate = StringVar()
        learn = ("0.005", "0.001", "0.0005", "0.0001", "0.00005", "0.00001")
        learn_entry = ttk.Combobox(window, width=10, textvariable=self.learn_rate)
        learn_entry["values"] = learn
        learn_entry.grid(column=2, row=6, sticky=(W, E))
        self.learn_rate.set("0.00001")

        # Number of training epochs
        self.epochs = StringVar()
        epoch = ("30", "40", "50", "60", "70", "80")
        epoch_entry = ttk.Combobox(window, width=10, textvariable=self.epochs)
        epoch_entry["values"] = epoch
        epoch_entry.grid(column=2, row=7, sticky=(W, E))
        self.epochs.set("3")

        # Loss function
        self.loss_function = StringVar()
        loss = ("BCE", "Dice", "FL")
        loss_entry = ttk.Combobox(window, width=10, textvariable=self.loss_function)
        loss_entry["values"] = loss
        loss_entry["state"] = "readonly"
        loss_entry.grid(column=2, row=8, sticky=(W, E))
        self.loss_function.set("BCE")

        # Add padding
        for child in window.winfo_children():
            child.grid_configure(padx=5, pady=5)

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
        gui_helpers.image_augmentation(self.train_image_dir.get(), self.mask_dir.get())


# ---------------------------------------------------------------------------------------------------
# Function required to run the GUI frm the prompt


def runMain() -> None:
    """
    Function that enables usage of the gui from command promt
    as pip package.

    Notes
    -----
    The GUI can be executed by typing 'python -m deep_acsa_gui.py' in the command
    subsequtently to installing the pip package´and activating the
    respective library.

    It is not necessary to download any files from the repository when the pip
    package is installed.

    For documentation of DL_Track see top of this module.
    """
    root = Tk()
    DeepACSA(root)
    root.mainloop()


# This statement is required to execute the GUI by typing 'python deep_acsa_gui.py' in the prompt
# when navigated to the folder containing the file and all dependencies.
if __name__ == "__main__":
    root = Tk()
    DeepACSA(root)
    root.mainloop()
