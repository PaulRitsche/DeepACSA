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


if __name__ == "__main__":
    root = Tk()
    DeepACSA(root)
    root.mainloop()
