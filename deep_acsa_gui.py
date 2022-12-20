"""Python module to create GUI for DeepACSA"""

import os
import tkinter as tk
from tkinter import StringVar, Tk, N, S, W, E
from tkinter import ttk, filedialog
from tkinter.tix import *
from threading import Thread, Lock
from PIL import Image

import gui_helpers


class DeepACSA:
    """Class which provides the utility of a graphical user interface.

    Attributes:
        input_dir: Path to root directory containings all files.
        model_path: Path to the keras segmentation model.
        depth: Scanning depth (cm) of ultrasound image.
        muscle: Muscle that is visible on ultrasound image.
        spacing: Distance (mm) between two scaling lines on ultrasound images.
        scaling: Scanning modaltity of ultrasound image.

    """

    def __init__(self, root):

        # set up threading
        self._lock = Lock()
        self._is_running = False
        self._should_stop = False

        # set up gui
        root.title("DeepACSA")
        root.iconbitmap("icon.ico")

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
        style.theme_use('clam')
        style.configure('TFrame', background = 'SkyBlue4')
        style.configure('TLabel', font=('Lucida Sans', 12),
                        foreground = 'black', background = 'SkyBlue4')
        style.configure('TRadiobutton', background = 'SkyBlue4',
                        foreground = 'black', font = ('Lucida Sans', 12))
        style.configure('TButton', background = 'linen',
                        foreground = 'black', font = ('Lucida Sans', 11))
        style.configure('TEntry', font = ('Lucida Sans', 12), background = 'linen',
                        foregrund = 'black')
        style.configure('TCombobox', background = 'SkyBlue4', foreground = 'black')

        # Tooltips
        tip = Balloon(root)
        tip.config(bg="HotPink3", bd=3)
        tip.label.config(bg="linen", fg="black")
        tip.message.config(bg="linen", fg="black", font=("Lucida Sans", 10))
        for sub in tip.subwidgets_all():
            sub.configure(bg='linen')

        # Paths
        # Input directory
        self.input = StringVar()
        input_entry = ttk.Entry(self.main, width=30, textvariable=self.input)
        input_entry.grid(column=2, row=2, columnspan=3, sticky=(W, E))
        #self.input.set("Desktop/DeepACSA/")
        # Model path
        self.model = StringVar()
        model_entry = ttk.Entry(self.main, width=30, textvariable=self.model)
        model_entry.grid(column=2, row=3, columnspan=3, sticky=(W, E))
        #self.model.set("C:/Users/admin/Documents/DeepACSA/notebooks/VGG16pre-VL-256.h5")

        # Radiobuttons
        # Image Type
        self.scaling = StringVar()
        efov = ttk.Radiobutton(self.main, text="Line", variable=self.scaling,
                               value="Line")
        efov.grid(column=2, row=7, sticky=W)
        static = ttk.Radiobutton(self.main, text="Bar", variable=self.scaling,
                                 value="Bar")
        static.grid(column=3, row=7, sticky=(W, E))
        manual = ttk.Radiobutton(self.main, text="Manual", variable=self.scaling,
                                 value="Manual")
        manual.grid(column=4, row=7, sticky=E)
        tip.bind_widget(efov,
                        balloonmsg="Choose image type." +
                        " \nIf image taken in panoramic mode, choose EFOV." +
                        " \nIf image taken in static B-mode, choose Static." +
                        " \nIf image taken in other modality, choose Manual" +
                        " \nin order to scale the image manually.")
        self.scaling.set("Bar")

        # Volume Calculation
        self.muscle_volume_calculation_wanted = StringVar()
        yes_volume = ttk.Radiobutton(self.main, text="Yes", variable=self.muscle_volume_calculation_wanted,
                                     value="Yes")
        yes_volume.grid(column=2, row=14, sticky=W)
        no_volume = ttk.Radiobutton(self.main, text="No", variable=self.muscle_volume_calculation_wanted,
                                      value="No")
        no_volume.grid(column=3, row=14, sticky=(W,E))
        tip.bind_widget(yes_volume,
                        balloonmsg="Choose whether or not to calculate volume." +
                        " \nImages need to be in one directory and consecutively labeled." +
                        " \nImages should must be of the same muscle.")
        self.muscle_volume_calculation_wanted.set("No")

        # Comboboxes
        # Filetype
        self.filetype = StringVar()
        filetype = ("/**/*.tif", "/**/*.tiff", "/**/*.png", "/**/*.bmp",
                    "/**/*.jpeg", "/**/*.jpg")
        filetype_entry = ttk.Combobox(self.main, width=10, textvariable=self.filetype)
        filetype_entry["values"] = filetype
        # filetype_entry["state"] = "readonly"
        filetype_entry.grid(column=2, row=6, sticky=E)
        tip.bind_widget(filetype_entry,
                        balloonmsg="Specifiy filetype of images in root" +
                        " \nthat are taken as whole quadriceps images." +
                        " \nThese images are being prepared for model prediction.")
        self.filetype.set("/**/*.jpg")
        # Muscles
        self.muscle = StringVar()
        muscle = ("VL", "RF", "GM", "GL")
        muscle_entry = ttk.Combobox(self.main, width=10, textvariable=self.muscle)
        muscle_entry["values"] = muscle
        muscle_entry["state"] = "readonly"
        muscle_entry.grid(column=2, row=8, sticky=(W, E))
        tip.bind_widget(muscle_entry,
                        balloonmsg="Choose muscle from dropdown list, " +
                        "\ndepending on which muscle is analyzed.")
        self.muscle.set("RF")
        # Image Depth
        self.depth = StringVar()
        depth = (2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8)
        depth_entry = ttk.Combobox(self.main, width=10, textvariable=self.depth)
        depth_entry["values"] = depth
        # depth_entry["state"] = "readonly"
        depth_entry.grid(column=2, row=9, sticky=(W, E))
        tip.bind_widget(depth_entry,
                        balloonmsg="Choose image depth from dropdown list " +
                        "\nor enter costum depth. Analyzed images must have " +
                        "\nthe same depth.")
        self.depth.set(5.5)
        # Spacing
        self.spacing = StringVar()
        spacing = (5, 10, 15, 20)
        spacing_entry = ttk.Combobox(self.main, width=10, textvariable=self.spacing)
        spacing_entry["values"] = spacing
        spacing_entry["state"] = "readonly"
        spacing_entry.grid(column=2, row=10, sticky=(W, E))
        tip.bind_widget(spacing_entry,
                        balloonmsg="Choose disance between scaling bars" +
                                   "\nin image form dropdown list. " +
                                   "\nDistance needs to be similar " +
                                   "\nin all analyzed images.")
        self.spacing.set(10)
        # Distance between ACSA for Volume Calculation
        self.distance = StringVar()
        distance_entry = ttk.Entry(self.main, width=10, textvariable=self.distance)
        distance_entry.grid(column=2, row=15, sticky=(W, E))
        tip.bind_widget(distance_entry,
                        balloonmsg="Choose distance between image slices" +
                                   "\nincluded in calulation in centimeter." +
                                   "\nBeware, distance must be equal between slices.")
        self.distance.set(7)

        # Buttons
        # Input directory
        input_button = ttk.Button(self.main, text="Input",
                                  command=self.get_root_dir)
        input_button.grid(column=5, row=2, sticky=E)
        tip.bind_widget(input_button,
                        balloonmsg="Choose root directory." +
                        " This is the folder containing all images.")
        # Model path
        model_button = ttk.Button(self.main, text="Model",
                                  command=self.get_model_path)
        model_button.grid(column=5, row=3, sticky=E)
        tip.bind_widget(model_button,
                        balloonmsg="Choose model path." +
                        " This is the path to the respective model.")
        # Break Button
        break_button = ttk.Button(self.main, text="Break", command=self.do_break)
        break_button.grid(column=1, row=16, sticky=W)
        # Run Button
        run_button = ttk.Button(self.main, text="Run", command=self.run_code)
        run_button.grid(column=2, row=16, sticky=(W, E))
        # Train Button
        train_button=ttk.Button(self.main, text="Train Model", command=self.train_model_window)
        train_button.grid(column=5, row=16, sticky=(W, E))

        # Labels
        ttk.Label(self.main, text="Directories",font=('Verdana', 14)).grid(column=1, row=1, sticky=W)
        ttk.Label(self.main, text="Root Directory").grid(column=1, row=2)
        ttk.Label(self.main, text="Model Path").grid(column=1, row=3)
        ttk.Label(self.main, text="Image Properties", font=('Verdana', 14)).grid(column=1, row=5,
                  sticky=W)
        ttk.Label(self.main, text="Image Type").grid(column=1, row=6)
        ttk.Label(self.main, text="Scaling Type").grid(column=1, row=7)
        ttk.Label(self.main, text="Muscle").grid(column=1, row=8)
        ttk.Label(self.main, text="Depth (cm)").grid(column=1, row=9)
        ttk.Label(self.main, text="Spacing (mm)").grid(column=1, row=10)
        ttk.Label(self.main, text="Muscle Volume", font=('Verdana', 14)).grid(column=1, row=13, sticky=W)
        ttk.Label(self.main, text="Volume Calculation").grid(column=1, row=14)
        ttk.Label(self.main, text="Distance (cm)").grid(column=1, row=15)

        for child in self.main.winfo_children():
            child.grid_configure(padx=5, pady=5)

        root.bind("<Return>", self.run_code)  # execute by pressing return

#--------------------------------------------------------------------------------------------------
# Functionalities used in GUI

    def get_root_dir(self):
        """ Asks the user to select the root directory.
            Can have up to two sub-levels.
            All images files (of the same type) in root are analysed.
        """
        root_dir = filedialog.askdirectory()
        self.input.set(root_dir)
        return root_dir

    def get_model_path(self):
        """ Asks the user to select the model path.
        """
        model_dir = filedialog.askopenfilename()
        self.model.set(model_dir)
        return model_dir

    def run_code(self):
        """ The code is run upon clicking.
        """
        try:
            if self.is_running:
                # don't run again if it is already running
                return
            self.is_running = True

            selected_input_dir = self.input.get()
            selected_model_path = self.model.get()
            selected_filetype = self.filetype.get()
            selected_muscle = self.muscle.get()
            selected_depth = float(self.depth.get())
            selected_spacing = self.spacing.get()
            selected_scaling = self.scaling.get()
            selected_volume_calculation = self.muscle_volume_calculation_wanted.get()
            distance_acsa = float(self.distance.get())

            if len(selected_input_dir) == 0:
                tk.messagebox.showerror("Information", "Check input parameters." +
                "\nPotential error source:  Invalid specified input directory")
                self.do_break()
                self.should_stop = False
                self.is_running = False

            elif len(selected_model_path) == 0:
                tk.messagebox.showerror("Information", "Check input parameters." +
                "\nPotential error source:  Invalid specified model path")
                self.do_break()
                self.should_stop = False
                self.is_running = False

            elif len(selected_filetype) == 0:
                tk.messagebox.showerror("Information", "Check input parameters." +
                "\nPotential error source:  Invalid specified filetype")
                self.do_break()
                self.should_stop = False
                self.is_running = False

            if selected_scaling == "Line":
                thread = Thread(
                    target=gui_helpers.calculate_batch_efov,
                    args=(
                        selected_input_dir,
                        selected_filetype,
                        selected_model_path,
                        selected_depth,
                        selected_muscle,
                        selected_volume_calculation,
                        distance_acsa,
                        self,
                    )
                )
            else:
                thread = Thread(
                    target=gui_helpers.calculate_batch,
                    args=(
                        selected_input_dir,
                        selected_filetype,
                        selected_model_path,
                        selected_spacing,
                        selected_muscle,
                        selected_scaling,
                        selected_volume_calculation,
                        distance_acsa,
                        self,
                    )
                )

            thread.start()

        except ValueError:
            tk.messagebox.showerror("Information", "Check input parameters." +
            "\nPotential error source:  Invalid specified depth or distance")
            self.do_break()
            self.should_stop = False
            self.is_running = False

    @property
    def should_stop(self):
        self._lock.acquire()
        should_stop = self._should_stop
        self._lock.release()
        return should_stop

    @property
    def is_running(self):
        self._lock.acquire()
        is_running = self._is_running
        self._lock.release()
        return is_running

    @should_stop.setter
    def should_stop(self, flag: bool):
        self._lock.acquire()
        self._should_stop = flag
        self._lock.release()

    @is_running.setter
    def is_running(self, flag: bool):
        self._lock.acquire()
        self._is_running = flag
        self._lock.release()

    def do_break(self):
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
        """
        # Open Window
        window = tk.Toplevel(bg="SkyBlue4")
        window.title("Model Training Window")
        window.iconbitmap("icon.ico")
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
        self.train_image_dir.set(
            "C:/Users/admin/Documents/DL_Track/Train_Data_DL_Track/apo_test"
        )

        # Mask directory
        self.mask_dir = StringVar()
        mask_entry = ttk.Entry(window, width=30, textvariable=self.mask_dir)
        mask_entry.grid(column=2, row=3, columnspan=3, sticky=(W, E))
        self.mask_dir.set(
            "C:/Users/admin/Documents/DL_Track/Train_Data_DL_Track/apo_mask_test"
        )

        # Output path
        self.out_dir = StringVar()
        out_entry = ttk.Entry(window, width=30, textvariable=self.out_dir)
        out_entry.grid(column=2, row=4, columnspan=3, sticky=(W, E))
        self.out_dir.set("C:/Users/admin/Documents")

        # Buttons
        # Train image button
        train_img_button = ttk.Button(window, text="Images", command=self.get_train_dir)
        train_img_button.grid(column=5, row=2, sticky=E)

        # Mask button
        mask_button = ttk.Button(window, text="Masks", command=self.get_mask_dir)
        mask_button.grid(column=5, row=3, sticky=E)

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
        loss = ("BCE") # "Dice", "FL")
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

    # ---------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    root = Tk()
    DeepACSA(root)
    root.mainloop()
