from tkinter import StringVar, Tk, N, S, W, E
from tkinter import ttk, filedialog
from tkinter.tix import *
import os
from PIL import Image, ImageTk
from predict_muscle_area import calculate_batch, calculate_batch_efov
from prepare_quad_imgs import prepare_quad_vl_imgs, prepare_quad_rf_imgs

from threading import Thread, Lock


class DeepACSA:
    """Class which provides the utility of a graphical user interface.

    Attributes:
        input_dir: Path to root directory containings all files. 
        model_path: Path to the keras segmentation model.
        flag_path: Path to .txt file containing flip-flags for images. 
        depth: Scanning depth (cm) of ultrasound image. 
        muscle: Muscle that is visible on ultrasound image.
        spacing: Distance (mm) between two scaling lines on ultrasound images.
        scaling: Scanning modaltity of ultrasound image. 


    Examples:
        >>> 
    """
    def __init__(self, root):

        # set up threading
        self._lock = Lock()
        self._is_running = False
        self._should_stop = False

        # set up gui
        root.title("DeepACSA")
        root.iconbitmap("icon.ico")

        main = ttk.Frame(root, padding="10 10 12 12")
        main.grid(column=0, row=0, sticky=(N, S, W, E))
        # Configure resizing of user interface
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=1)
        main.columnconfigure(3, weight=1)
        main.columnconfigure(4, weight=1)
        main.columnconfigure(5, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # Tooltips
        tip = Balloon(root)

        # Paths
        # Input directory
        self.input = StringVar()
        input_entry = ttk.Entry(main, width=30, textvariable=self.input)
        input_entry.grid(column=2, row=2, columnspan=3, sticky=(W, E))
        # Model path
        self.model = StringVar()
        model_entry = ttk.Entry(main, width=30, textvariable=self.model)
        model_entry.grid(column=2, row=3, columnspan=3, sticky=(W, E))
        # Flag path
        self.flags = StringVar()
        flags_entry = ttk.Entry(main, width=14, textvariable=self.flags)
        flags_entry.grid(column=2, row=4, columnspan=3, sticky=(W, E))

        # Radiobuttons
        # Image Preparing
        self.image_preparation = StringVar()
        yes = ttk.Radiobutton(main, text="Yes", variable=self.image_preparation, 
                              value="Yes")
        yes.grid(column=2, row=12, sticky=W)
        no = ttk.Radiobutton(main, text="No", variable=self.image_preparation, 
                             value="No")
        no.grid(column=3, row=12, sticky=(W,E))

        # Image Type
        self.scaling = StringVar()
        efov = ttk.Radiobutton(main, text="Line", variable=self.scaling,
                               value="Line")
        efov.grid(column=2, row=7, sticky=W)
        static = ttk.Radiobutton(main, text="Bar", variable=self.scaling,
                                 value="Bar")
        static.grid(column=3, row=7, sticky=(W, E))
        manual = ttk.Radiobutton(main, text="Manual", variable=self.scaling,
                                 value="Manual")
        manual.grid(column=4, row=7, sticky=E)
        tip.bind_widget(efov,
                        balloonmsg="Choose image type from dropdown list." +
                        " If image taken in panoramic mode, choose EFOV." +
                        " If image taken in static B-mode, choose Static." +
                        " If image taken in other modality, choose Manual" +
                        " in order to scale the image manually.")
        # Comboboxes
        # Filetype
        self.filetype = StringVar()
        filetype = ("/*.tif", "/*.tiff", "/*.png", "/*.bmp", "/*.jpeg")
        filetype_entry = ttk.Combobox(main, width=10, textvariable=self.filetype)
        filetype_entry["values"] = filetype
        # filetype_entry["state"] = "readonly"
        filetype_entry.grid(column=2, row=6, sticky=E)
        tip.bind_widget(efov,
                        balloonmsg="Specifiy filetype of images in root" +
                        " that are taken as whole quadriceps images." +
                        " These images are being prepared for model prediction.")

        # Muscles
        self.muscle = StringVar()
        muscle = ("VL", "RF", "GM", "GL")
        muscle_entry = ttk.Combobox(main, width=10, textvariable=self.muscle)
        muscle_entry["values"] = muscle
        muscle_entry["state"] = "readonly"
        muscle_entry.grid(column=2, row=8, sticky=(W, E))
        tip.bind_widget(muscle_entry,
                        balloonmsg="Choose muscle from dropdown list, " +
                        "depending on which muscle is analyzed.")
        # Image Depth
        self.depth = StringVar()
        depth = (2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8)
        depth_entry = ttk.Combobox(main, width=10, textvariable=self.depth)
        depth_entry["values"] = depth
        # depth_entry["state"] = "readonly"
        depth_entry.grid(column=2, row=9, sticky=(W, E))
        tip.bind_widget(depth_entry,
                        balloonmsg="Choose image depth from dropdown list " +
                        "or enter costum depth. Analyzed images must have " +
                        "the same depth.")
        
        # Spacing
        self.spacing = StringVar()
        spacing = (5, 10, 15, 20)
        spacing_entry = ttk.Combobox(main, width=10, textvariable=self.spacing)
        spacing_entry["values"] = spacing
        spacing_entry["state"] = "readonly"
        spacing_entry.grid(column=2, row=10, sticky=(W, E))
        tip.bind_widget(spacing_entry,
                        balloonmsg="Choose disance between scaling bars" +
                                   " in image form dropdown list. " +
                                   "Distance needs to be similar " +
                                   "in all analyzed images.")

        # Buttons
        # Input directory
        input_button = ttk.Button(main, text="Input",
                                  command=self.get_root_dir)
        input_button.grid(column=5, row=2, sticky=E)
        tip.bind_widget(input_button,
                        balloonmsg="Choose root directory." +
                        " This is the folder containing all images.")
        # Model path
        model_button = ttk.Button(main, text="Model",
                                  command=self.get_model_path)
        model_button.grid(column=5, row=3, sticky=E)
        tip.bind_widget(model_button,
                        balloonmsg="Choose model path." +
                        " This is the path to the respective model.")
        # Flip Flag path
        flags_button = ttk.Button(main, text="Flip Flag",
                                  command=self.get_flag_dir)
        flags_button.grid(column=5, row=4, sticky=E)
        tip.bind_widget(flags_button,
                        balloonmsg="Choose Flag File Path." +
                        " This is the path to the .txt file containing" +
                        " flipping info.")
        # Prepare Imgs Button
        prepare_button = ttk.Button(main, text="Prepare Images", 
                                    command=self.prepare_imgs)
        prepare_button.grid(column=5, row=12, sticky=E)
        tip.bind_widget(prepare_button,
                        balloonmsg="Choose whether to prepare your images." +
                        " This should be done befor inputting them to the model." +
                        " The program will save the images in root.")

        # Break Button
        break_button = ttk.Button(main, text="Break", command=self.do_break)
        break_button.grid(column=1, row=13, sticky=W)
        # Run Button
        run_button = ttk.Button(main, text="Run", command=self.run_code)
        run_button.grid(column=2, row=13, sticky=(W, E))

        # Labels
        ttk.Label(main, text="Directories",font=("bold")).grid(column=1, row=1, sticky=W)
        ttk.Label(main, text="Root Directory").grid(column=1, row=2)
        ttk.Label(main, text="Model Path").grid(column=1, row=3)
        ttk.Label(main, text="Flip Flag Path").grid(column=1, row=4)
        ttk.Label(main, text="Image Properties", font=("bold")).grid(column=1, row=5, sticky=W)
        ttk.Label(main, text="Image Type").grid(column=1, row=6)
        ttk.Label(main, text="Scaling Type").grid(column=1, row=7)
        ttk.Label(main, text="Muscle").grid(column=1, row=8)
        ttk.Label(main, text="Depth (cm)").grid(column=1, row=9)
        ttk.Label(main, text="Spacing (mm)").grid(column=1, row=10)
        ttk.Label(main, text="Image Preparation", font=("bold")).grid(column=1, row=11, sticky=W)
        ttk.Label(main, text="Prepare Images").grid(column=1, row=12)
        for child in main.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # depth_entry.focus()
        root.bind("<Return>", self.run_code)  # execute by pressing return

    def get_root_dir(self):

        root_dir = filedialog.askdirectory()
        self.input.set(root_dir)
        return root_dir

    def get_model_path(self):

        model_dir = filedialog.askopenfilename()
        self.model.set(model_dir)
        return model_dir

    def get_flag_dir(self):

        flag_dir = filedialog.askopenfilename()
        self.flags.set(flag_dir)
        return flag_dir

    def prepare_imgs(self):

        selected_image_preparation = self.image_preparation.get()
        selected_filetype = self.filetype.get()
        selected_input_dir = self.input.get()

        if selected_image_preparation == "Yes":
            # Create directory for prepared imgs
            dirname = selected_input_dir + "/prepared_imgs"
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            # Prepare imgs
            prepare_quad_rf_imgs(selected_input_dir, 
                                 selected_filetype,
                                 dirname)
            prepare_quad_vl_imgs(selected_input_dir, 
                                 selected_filetype,
                                 dirname)

        else: 
            pass

    def run_code(self):
   
        if self.is_running:
            # don't run again if it is already running
            return
        self.is_running = True

        selected_muscle = self.muscle.get()
        selected_depth = float(self.depth.get())
        selected_spacing = self.spacing.get()
        selected_scaling = self.scaling.get()
        selected_input_dir = self.input.get()
        selected_model_path = self.model.get()
        selected_flag_path = self.flags.get()
        selected_filetype = self.filetype.get()

        if selected_scaling == "Line":
            thread = Thread(
                target=calculate_batch_efov,
                args=(
                    selected_input_dir,
                    selected_filetype,
                    selected_model_path,
                    selected_depth,
                    selected_muscle,
                    self,
                )
            )
        else:
            thread = Thread(
                target=calculate_batch,
                args=(
                    selected_input_dir,
                    selected_filetype,
                    selected_flag_path,
                    selected_model_path,
                    selected_depth,
                    selected_spacing,
                    selected_muscle,
                    selected_scaling,
                    self,
                )
            )

        thread.start()

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
        self.should_stop = True
        


if __name__ == "__main__":
    root = Tk()
    DeepACSA(root)
    root.mainloop()
