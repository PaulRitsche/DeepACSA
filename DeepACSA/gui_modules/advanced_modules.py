""" """

import os
import customtkinter as ctk
from CTkToolTip import *
import math
import numpy as np
from tkinter import ttk, W, E, N, S, StringVar, BooleanVar, filedialog, Canvas
from DeepACSA import gui_helpers
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from threading import Lock, Thread


class AdvancedAnalysis:
    """
    Function to open a toplevel where masks
    can either be created for training purposes or
    can be inspected subsequent to labelling.
    """

    def __init__(self, parent):

        self.parent = parent
        # self.parent.load_settings()

        # set up threading
        self._lock = Lock()
        self._is_running = False
        self._should_stop = False

        # Open Window
        self.advanced_window = ctk.CTkToplevel(fg_color="#2A484E")
        self.advanced_window.title("Advanced Methods Window")

        head_path = os.path.dirname(os.path.abspath(__file__))
        iconpath = head_path + "/gui_helpers/home_im.ico"
        # self.advanced_window.iconbitmap(iconpath)

        # if platform.startswith("win"):
        #     self.a_window.after(200, lambda: self.a_window.iconbitmap(iconpath))

        # Configure resizing of user interface
        for row in range(21):
            self.advanced_window.rowconfigure(row, weight=1)
        for column in range(4):
            self.advanced_window.rowconfigure(column, weight=1)

        self.advanced_window.columnconfigure(0, weight=1)
        self.advanced_window.columnconfigure(1, weight=5)
        self.advanced_window.rowconfigure(0, weight=1)
        self.advanced_window.minsize(width=600, height=400)

        self.advanced_window.grab_set()

        ctk.CTkLabel(self.advanced_window, text="Select Method").grid(column=1, row=0)

        # Mask Option
        self.advanced_option = StringVar()
        advanced_entry = [
            "Train Model",
            "Inspect Masks",
            "Remove Image Parts",
            "Create Masks",
        ]
        advanced_entry = ctk.CTkComboBox(
            self.advanced_window,
            width=20,
            variable=self.advanced_option,
            state="readonly",
            values=advanced_entry,
        )
        advanced_entry.grid(column=1, row=1, sticky=(W, E))
        self.advanced_option.set("...")
        self.advanced_option.trace_add("write", self.on_mask_change)

        # Add padding
        for child in self.advanced_window.winfo_children():
            child.grid_configure(padx=5, pady=5)

        # Initialize video-related attributes
        self.video_path = None
        self.processed_frames = []
        self.current_frame_index = 0

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
        try:

            # Clear the existing frame if it exists
            if hasattr(self, "advanced_window_frame"):
                for widget in self.advanced_window_frame.winfo_children():
                    widget.destroy()

            self.advanced_window_frame = ctk.CTkFrame(self.advanced_window)
            self.advanced_window_frame.grid(column=1, row=2, sticky=(N, S, W, E))

            # Set the desired window size
            self.desired_width = 800
            self.desired_height = 600

            if self.advanced_option.get() == "Inspect Masks":

                if hasattr(self, "mask_button"):
                    self.mask_button.destroy()

                # Train image directory
                self.raw_image_dir = StringVar()
                dir1_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Image Dir",
                    command=lambda: (self.raw_image_dir.set(filedialog.askdirectory())),
                )
                dir1_button.grid(column=0, row=2, sticky=(W, E))

                # Mask directory
                self.mask_image_dir = StringVar()
                self.dir2_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Mask Dir",
                    command=lambda: (
                        self.mask_image_dir.set(filedialog.askdirectory())
                    ),
                )
                self.dir2_button.grid(column=1, row=2, sticky=(W, E))

                # Start index
                self.start_label = ctk.CTkLabel(
                    self.advanced_window_frame, text="Start at Image:"
                )
                self.start_label.grid(column=0, row=4, sticky=W)
                start_idx = StringVar()
                self.idx = ctk.CTkEntry(
                    self.advanced_window_frame, width=10, textvariable=start_idx
                )
                self.idx.grid(column=1, row=4, sticky=W)
                start_idx.set("0")

                ttk.Separator(self.advanced_window_frame, orient="horizontal").grid(
                    column=1, row=5, columnspan=4
                )

                # Inspect button
                self.inspect_button = ctk.CTkButton(
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
                self.inspect_button.grid(column=1, row=6, sticky=(W, E))

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
                image_entry = ctk.CTkEntry(
                    self.advanced_window_frame,
                    width=100,
                    textvariable=self.raw_image_dir,
                )
                image_entry.grid(column=1, row=1, columnspan=2, sticky=(W, E))
                self.raw_image_dir.set("Select Input Image Directory")

                dir1_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Image Dir",
                    command=lambda: (self.raw_image_dir.set(filedialog.askdirectory())),
                )
                dir1_button.grid(column=3, row=1, sticky=(W, E))

                self.mask_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Create Masks",
                    command=self._start_mask_interaction,
                )
                self.mask_button.grid(column=2, row=2, sticky=(W, E))
                tooltip_remove = CTkToolTip(
                    self.mask_button,
                    message="Click to start manual analysis/labelling. \nThis creates binary masks of the selected area. \nResulting masks and images can be used for model training.",
                    delay=0.5,
                    bg_color="#A8D8CD",
                    text_color="#000000",
                    alpha=0.7,
                )

            elif self.advanced_option.get() == "Train Model":
                # Labels
                ctk.CTkLabel(
                    self.advanced_window_frame,
                    text="Training Directories",
                    font=("Verdana", 20),
                ).grid(column=1, row=0, padx=10)

                # Train image button
                train_img_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Images",
                    command=self.get_train_dir,
                )
                train_img_button.grid(column=0, row=2, sticky=E)

                # Mask button
                mask_button = ctk.CTkButton(
                    self.advanced_window_frame, text="Masks", command=self.get_mask_dir
                )
                mask_button.grid(column=1, row=3, sticky=(W, E))

                # Input directory
                out_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Output",
                    command=self.get_output_dir,
                )
                out_button.grid(column=2, row=2, sticky=W)

                ttk.Separator(
                    self.advanced_window_frame, orient="horizontal", style="TSeparator"
                ).grid(column=0, row=5, columnspan=9, sticky=(W, E))

                # Hyperparameters
                ctk.CTkLabel(
                    self.advanced_window_frame,
                    text="Hyperparameters",
                    font=("Verdana", 20),
                ).grid(column=0, row=6, padx=10)

                # Batch Size Label
                ctk.CTkLabel(self.advanced_window_frame, text="Batch Size").grid(
                    column=0, row=7
                )
                # Batch size Combobox
                self.batch_size = StringVar()
                size = ["1", "2", "3", "4", "5", "6"]
                size_entry = ctk.CTkComboBox(
                    self.advanced_window_frame,
                    width=10,
                    variable=self.batch_size,
                    values=size,
                )
                size_entry.grid(column=1, row=7, sticky=(W, E))
                self.batch_size.set("1")

                # Learning Rate Label
                ctk.CTkLabel(self.advanced_window_frame, text="Learning Rate").grid(
                    column=0, row=8
                )
                # Learning Rate Combobox
                self.learn_rate = StringVar()
                learn = ["0.005", "0.001", "0.0005", "0.0001", "0.00005", "0.00001"]
                learn_entry = ctk.CTkComboBox(
                    self.advanced_window_frame,
                    width=10,
                    variable=self.learn_rate,
                    values=learn,
                )
                learn_entry.grid(column=1, row=8, sticky=(W, E))
                self.learn_rate.set("0.00001")

                # Epochs Label
                ctk.CTkLabel(self.advanced_window_frame, text="Epochs").grid(
                    column=0, row=9
                )
                # Number of training epochs
                self.epochs = StringVar()
                epoch = ["30", "40", "50", "60", "70", "80"]
                epoch_entry = ctk.CTkComboBox(
                    self.advanced_window_frame,
                    width=10,
                    variable=self.epochs,
                    values=epoch,
                )
                epoch_entry.grid(column=1, row=9, sticky=(W, E))
                self.epochs.set("3")

                # Loss function Label
                ctk.CTkLabel(self.advanced_window_frame, text="Loss Function").grid(
                    column=0, row=10
                )
                # Loss function Combobox
                self.loss_function = StringVar()
                loss_entry = ctk.CTkComboBox(
                    self.advanced_window_frame,
                    width=10,
                    variable=self.loss_function,
                    values=["BCE"],
                )
                loss_entry["state"] = "readonly"
                loss_entry.grid(column=1, row=10, sticky=(W, E))
                self.loss_function.set("BCE")

                ttk.Separator(
                    self.advanced_window_frame, orient="horizontal", style="TSeparator"
                ).grid(column=0, row=11, columnspan=9, sticky=(W, E))

                # Data augmentation button
                data_augmentation_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Augment Images",
                    command=self.augment_images,
                )
                data_augmentation_button.grid(column=1, row=12, sticky=(W, E))

                # Augmentation Warning
                ctk.CTkLabel(
                    self.advanced_window_frame,
                    text="*Use augmentation only with \nbacked-up original images*",
                    font=("Segue UI", 8, "bold"),
                    text_color="#000000",
                ).grid(column=0, row=12, sticky=E)

                # Model train button
                self.model_button = ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Start Training",
                    command=self.train_model,
                )
                self.model_button.grid(column=2, row=12, sticky=E)

            elif self.advanced_option.get() == "Remove Image Parts":

                ctk.CTkButton(
                    self.advanced_window_frame,
                    text="Load Image",
                    command=self.load_image,
                ).grid(column=1, row=0, columnspan=3, sticky=W)

            # Add padding
            for child in self.advanced_window_frame.winfo_children():
                child.grid_configure(padx=5, pady=5)

        except FileNotFoundError:
            tk.messagebox.showerror("Information", "Enter the correct folder path!")

        # Add padding
        for child in self.advanced_window_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def load_image(self):
        """
        Loads all valid image files from a folder and displays the first one.
        """
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if not folder_path:
            return

        self.image_input_dir = folder_path
        valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        image_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]
        )

        if not image_files:
            print("No valid image files found in folder.")
            return

        self.processed_frames = []
        for filename in image_files:
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                self.processed_frames.append(img)

        if self.processed_frames:
            print(f"Loaded {len(self.processed_frames)} images.")
            self.display_frame(0)
            # self.update_slider_range()
        else:
            print("No images could be read.")

    def display_frame(self, frame_index):
        """
        Displays a specific frame on the canvas.
        """
        frame = self.processed_frames[frame_index]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to the desired dimensions
        frame_resized = cv2.resize(frame_rgb, (self.desired_width, self.desired_height))

        frame_image = Image.fromarray(frame_resized)
        frame_tk = ImageTk.PhotoImage(image=frame_image)

        if not hasattr(self, "video_canvas"):
            self.video_canvas = ctk.CTkLabel(self.advanced_window_frame, text="")
            self.video_canvas.grid(column=0, row=0, columnspan=2, sticky=(W, E))

        self.video_canvas.imgtk = frame_tk
        self.video_canvas.configure(image=frame_tk)

        # Allow user to make a selection on the first frame
        if frame_index == 0:
            self.selection = None
            self.canvas = Canvas(
                self.advanced_window_frame,
                width=self.desired_width,
                height=self.desired_height,
            )
            self.canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
            self.canvas.grid(column=0, row=0, columnspan=2, sticky=(W, E))
            self.canvas.bind("<ButtonPress-1>", self.on_button_press)
            self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        else:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
            self.canvas.imgtk = frame_tk  # Keep a reference to avoid garbage collection

        remove_button = ctk.CTkButton(
            self.advanced_window_frame,
            text="Remove Parts",
            command=self.remove_image_parts,
        )
        remove_button.grid(column=0, row=6, sticky=W, padx=5, pady=5)

        ctk.CTlabel(
            self.advanced_window_frame,
            text="*Select region to be cropped with left mouse drag. \nSelect new region to remove old region. \nPress button to save the image. \nParts are removed from all images in the selected folder",
            font=("Verdana", 10, "bold"),
            text_color="#000000",
        ).grid(column=1, row=6, columnspan=2, sticky=W)

    def on_button_press(self, event):
        self.selection = (event.x, event.y, event.x, event.y)

    def on_mouse_drag(self, event):
        x0, y0, _, _ = self.selection
        self.selection = (x0, y0, event.x, event.y)
        self.canvas.delete("selection")
        self.canvas.create_rectangle(
            x0, y0, event.x, event.y, outline="red", tag="selection"
        )

    def on_button_release(self, event):
        x0, y0, x1, y1 = self.selection
        self.selection = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def remove_image_parts(self):
        """
        Removes the selected rectangular region from all images in a folder
        and saves them with the same filenames in the selected output path.
        """
        input_path = self.image_input_dir
        output_path = input_path + "/processed"
        os.makedirs(output_path, exist_ok=True)

        if not input_path or not output_path:
            print("No input or output path specified.")
            return

        if not hasattr(self, "selection") or self.selection is None:
            print("No selection made.")
            return

        try:
            x0, y0, x1, y1 = self.selection

            # Supported image extensions
            valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            image_files = [
                f for f in os.listdir(input_path) if f.lower().endswith(valid_ext)
            ]

            if not image_files:
                print("No images found in the folder.")
                return

            for filename in image_files:
                image_path = os.path.join(input_path, filename)
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Warning: Failed to load {filename}. Skipping.")
                    continue

                # Resize to expected size (optional, if needed)
                if hasattr(self, "desired_width") and hasattr(self, "desired_height"):
                    img = cv2.resize(img, (self.desired_width, self.desired_height))

                # Black out the selected region
                img[y0:y1, x0:x1] = 0

                # Save result
                save_path = os.path.join(output_path, filename)
                cv2.imwrite(save_path, img)

                ctk.CTkLabel(
                    self.advanced_window_frame,
                    text=f"Masked image saved to\n{save_path}",
                    font=("Verdana", 10, "italic"),
                ).grid(column=0, row=7, columnspan=2, sticky=(W, E))

            print(f"Masked images saved to: {output_path}")

        except Exception as e:
            print(f"Error during image processing: {str(e)}")

    def select_area_tk(self, image, callback):
        """
        Interactive area and calibration selection directly in the Tkinter canvas.
        First 2 clicks define the calibration (1 cm), remaining clicks define a polygon.
        The area mask is passed to the callback once complete.
        """

        self.calibration_points = []
        self.mask_polygon_points = []
        self.calib_dist = None
        self.current_image = image.copy()
        self.drawing_polygon = True
        self.callback = callback

        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.desired_width, self.desired_height))
        self.frame_image = ImageTk.PhotoImage(Image.fromarray(frame_resized))

        self.instruction_label = ctk.CTkLabel(
            self.advanced_window_frame,
            text="Click 2 points that are exactly 1 cm apart (for calibration).\nRight-click = undo last point.",
            bg_color="#5e7d7d",
            font=("Segoe UI", 12, "italic"),
        )
        self.instruction_label.grid(
            column=0, row=2, columnspan=4, sticky=(W, E), pady=(5, 2)
        )

        self.canvas = Canvas(
            self.advanced_window_frame,
            width=self.desired_width,
            height=self.desired_height,
            cursor="crosshair",
        )
        self.canvas.grid(column=0, row=1, columnspan=4, sticky=(W, E))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.frame_image)

        self.canvas.bind("<Button-1>", self._on_area_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.advanced_window_frame.master.bind("<Return>", self._on_confirm_polygon)

    def _on_area_click(self, event):
        """
        Handle left mouse click to select points for calibration or polygon.
        """
        x, y = event.x, event.y
        if len(self.calibration_points) < 2:
            self.calibration_points.append((x, y))
            self.canvas.create_oval(
                x - 2, y - 2, x + 2, y + 2, outline="red", fill="red"
            )
            if len(self.calibration_points) == 2:
                dx = self.calibration_points[1][0] - self.calibration_points[0][0]
                dy = self.calibration_points[1][1] - self.calibration_points[0][1]
                self.calib_dist = math.sqrt(dx**2 + dy**2)
                self.instruction_label.configure(
                    text=f"Calibrated: 1 cm = {self.calib_dist:.1f} px. Now click to outline the area with left-click. \nRight-click = undo last point, Enter = confirm selection, ESC = cancel."
                )
                print(f"Calibrated: 1 cm = {self.calib_dist:.1f} px")
        else:
            self.mask_polygon_points.append((x, y))
            self.canvas.create_oval(
                x - 2, y - 2, x + 2, y + 2, outline="green", fill="green"
            )

    def _on_right_click(self, event):
        """
        Undo the last point clicked (either calibration or polygon point).
        """
        if len(self.calibration_points) < 2:
            self.instruction_label.configure(
                text="Click 2 points that are exactly 1 cm apart (for calibration).\nRight-click = undo last point, ENTER = confirm, ESC = cancel."
            )
        else:
            self.instruction_label.configure(
                text="Click to outline area.\nRight-click = undo last point, ENTER = confirm, ESC = cancel."
            )

        if self.mask_polygon_points:
            self.mask_polygon_points.pop()
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.frame_image)
            # Redraw calibration points
            for x, y in self.calibration_points:
                self.canvas.create_oval(
                    x - 2, y - 2, x + 2, y + 2, outline="red", fill="red"
                )
            # Redraw polygon points
            for x, y in self.mask_polygon_points:
                self.canvas.create_oval(
                    x - 2, y - 2, x + 2, y + 2, outline="green", fill="green"
                )

    def _on_confirm_polygon(self, event):
        """
        Confirm the polygon selection and calculate the area.
        """
        if self.calib_dist is None or len(self.mask_polygon_points) < 3:
            self.instruction_label.configure(
                text="Need 2 calibration points and 3+ polygon points to save mask.\nSelect more points.\nRight-click = undo last point, ENTER = confirm, ESC = cancel."
            )
            return

        h, w = self.current_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        scale_x = self.current_image.shape[1] / self.desired_width
        scale_y = self.current_image.shape[0] / self.desired_height

        points_scaled = [
            (int(x * scale_x), int(y * scale_y)) for x, y in self.mask_polygon_points
        ]
        cv2.fillPoly(mask, [np.array(points_scaled, dtype=np.int32)], 255)

        muscle_area = cv2.countNonZero(mask) / (self.calib_dist**2)

        if self.calib_dist is None or len(self.mask_polygon_points) < 3:
            self.instruction_label.configure(
                text="Need 2 calibration and 3+ area points. Click more or press ESC to cancel.\nRight-click = undo last point, ENTER = confirm, ESC = cancel."
            )
            return

        self.instruction_label.configure(text="✔ Mask confirmed.")

        print(f"Area: {muscle_area:.2f} cm²")

        self.canvas.destroy()
        self.callback(mask, muscle_area)

    def _start_mask_interaction(self):
        """
        Start the mask interaction process.
        """
        input_dir = self.raw_image_dir.get()
        self.image_files = sorted(
            [
                f
                for f in os.listdir(input_dir)
                if f.lower().endswith((".jpg", ".png", ".tif", ".jpeg"))
            ]
        )

        if not self.image_files:
            print("No images found.")
            return

        # Create subfolders for output
        self.image_output_dir = os.path.join(input_dir, "_images")
        self.mask_output_dir = os.path.join(input_dir, "_masks")
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(self.mask_output_dir, exist_ok=True)

        self.current_image_index = 0
        self._load_next_image()

    def _load_next_image(self):
        """
        Load the next image from the list and display it.
        """
        if self.current_image_index >= len(self.image_files):
            print("All images processed.")
            self.instruction_label.configure(text="All images processed.")
            return

        filename = self.image_files[self.current_image_index]
        image_path = os.path.join(self.raw_image_dir.get(), filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping unreadable file: {filename}")
            self.current_image_index += 1
            self._load_next_image()
            return

        self.current_filename = filename
        self.select_area_tk(image, callback=self._handle_mask_result)

    def _handle_mask_result(self, mask, area):
        """
        Handle the mask result and save the image and mask.
        """
        mask_path = os.path.join(self.mask_output_dir, self.current_filename)
        image_path = os.path.join(self.image_output_dir, self.current_filename)

        # Save image and mask with the same name
        full_image = self.current_image  # Original image saved as well
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(image_path, full_image)

        print(f"Saved: {self.current_filename} | Area: {area:.2f} cm²")
        self.current_image_index += 1
        self._load_next_image()

    def get_train_dir(self):
        """
        Instance method to ask the user to select the training image
        directory path. All image files (of the same specified filetype) in
        the directory are analysed. This must be an absolute path.
        """
        self.train_image_dir = filedialog.askdirectory()
        ctk.CTkLabel(
            self.advanced_window_frame,
            text=f"Folder: {os.path.basename(self.train_image_dir)}",
            font=("Segue UI", 8, "bold"),
        ).grid(column=0, row=3)

    def get_mask_dir(self):
        """
        Instance method to ask the user to select the training mask
        directory path. All mask files (of the same specified filetype) in
        the directory are analysed.The mask files and the corresponding
        image must have the exact same name. This must be an absolute path.
        """
        self.mask_dir = filedialog.askdirectory()
        ctk.CTkLabel(
            self.advanced_window_frame,
            text=f"Folder: {os.path.basename(self.mask_dir)}",
            font=("Segue UI", 8, "bold"),
        ).grid(column=1, row=3)

    def get_output_dir(self):
        """
        Instance method to ask the user to select the output
        directory path. Here, all file created during model
        training (model file, weight file, graphs) are saved.
        This must be an absolute path.
        """
        self.out_dir = filedialog.askdirectory()
        ctk.CTkLabel(
            self.advanced_window_frame,
            text=f"Folder: {os.path.basename(self.out_dir)}",
            font=("Segue UI", 8, "bold"),
        ).grid(column=2, row=3)

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

            # configure stop button
            self.model_button.configure(text="Stop Training", command=self.do_break)

            # Get input paremeter
            selected_images = self.train_image_dir
            selected_masks = self.mask_dir
            selected_outpath = self.out_dir

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

    # Method used for data augmentation
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
            selected_images = self.train_image_dir
            selected_masks = self.mask_dir

            # Make sure some kind of filetype is specified.
            if len(selected_images) < 3 or len(selected_masks) < 3:
                tk.messagebox.showerror("Information", "Specified directories invalid.")
                self.should_stop = False
                self.is_running = False
                self.do_break()
                return

            gui_helpers.image_augmentation(selected_images, selected_masks, self)

            # Inform user in GUI
            tk.messagebox.showinfo(
                "Information",
                "Data augmentation successful."
                + "\nResults are saved to specified input paths.",
            )

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

    # -----------------------------------------------------------
    # --- Threading properties ---

    @property
    def should_stop(self) -> bool:
        """Instance method to define the should_stop
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
        # Get private variable _should_stop
        should_stop = self._should_stop
        self._lock.release()
        return should_stop

    @property
    def is_running(self) -> bool:
        """Instance method to define the is_running
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
        """Instance method to define the should_stop
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
        """Instance method to define the is_running
        property setter method. The setter method is used
        to set the self._is_running attribute as if it was
        a public attribute. The argument "flag" is thereby
        validated to ensure proper input (boolean)
        """
        self._lock.acquire()
        self._is_running = flag
        self._lock.release()

    def do_break(self):
        """Instance method to break the analysis process when the
        button "break" is pressed.

        This changes the instance attribute self.should_stop
        to True, given that the analysis is already running.
        The attribute is checked befor every iteration
        of the analysis process.
        """

        self.should_stop = True
        self.is_running = False

        self.model_button.configure(text="Start Training", command=self.train_model)
