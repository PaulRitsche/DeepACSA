import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import pandas as pd
from PIL import Image, ImageTk
import numpy as np

from DeepACSA.gui_helpers.image_processing import (
    excel_expo,
    find_starting_points,
    measure_area,
    preprocess_image,
    process_images,
)


class ACSAutoApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("ACSAuto - Anatomical Cross-Sectional Area Analysis")
        self.geometry("1000x600")
        self.configure(padx=10, pady=10)
        master_path = os.path.dirname(os.path.abspath(__file__))
        theme_path = os.path.join(
            master_path, "..", "gui_helpers", "gui_files", "ui_color_theme.json"
        )
        ctk.set_default_color_theme(theme_path)

        # Variables
        self.analysis_type = ctk.StringVar(value="Folder")
        self.export_excel = ctk.BooleanVar(value=False)
        self.input_dir = ctk.StringVar()
        self.output_dir = ctk.StringVar()
        self.muscle = ctk.StringVar(value="Rectus femoris")
        self.outline_strategy = ctk.StringVar(value="Automatic")
        self.sorting = ctk.BooleanVar(value=True)
        self.scaling = ctk.StringVar(value="Automatic")
        self.scan_depth = ctk.DoubleVar(value=5)
        self.flip_horizontal = ctk.BooleanVar(value=False)
        self.flip_vertical = ctk.BooleanVar(value=False)

        self.circles = []
        self.drawing = False

        # UI Setup
        self.create_ui()

    def create_ui(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        ctk.CTkLabel(main_frame, text="Type of Analysis").grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkRadioButton(
            main_frame, text="Folder", variable=self.analysis_type, value="Folder"
        ).grid(row=1, column=0, sticky="w")
        ctk.CTkRadioButton(
            main_frame, text="Image", variable=self.analysis_type, value="Image"
        ).grid(row=1, column=1, sticky="w")

        ctk.CTkCheckBox(
            main_frame, text="Export to Excel", variable=self.export_excel
        ).grid(row=2, column=0, sticky="w")

        ctk.CTkLabel(main_frame, text="Input Directory").grid(
            row=3, column=0, sticky="w"
        )
        ctk.CTkButton(main_frame, text="Browse", command=self.select_input_source).grid(
            row=3, column=1, sticky="w"
        )

        ctk.CTkLabel(main_frame, text="Output Directory").grid(
            row=4, column=0, sticky="w"
        )
        ctk.CTkButton(main_frame, text="Browse", command=self.select_output_dir).grid(
            row=4, column=1, sticky="w"
        )

        ctk.CTkLabel(main_frame, text="Muscle Type").grid(row=5, column=0, sticky="w")
        ctk.CTkOptionMenu(
            main_frame,
            variable=self.muscle,
            values=[
                "Rectus femoris",
                "Vastus lateralis",
                "Quad RF",
                "Quad VL",
                "Quadriceps",
                "Gastro MED",
            ],
        ).grid(row=5, column=1, sticky="w")

        ctk.CTkLabel(main_frame, text="Outline Finder Strategy").grid(
            row=6, column=0, sticky="w"
        )
        ctk.CTkOptionMenu(
            main_frame,
            variable=self.outline_strategy,
            values=["Manual", "Automatic", "Fixed Pixels"],
        ).grid(row=6, column=1, sticky="w")

        ctk.CTkLabel(main_frame, text="Scaling").grid(row=7, column=0, sticky="w")
        ctk.CTkOptionMenu(
            main_frame, variable=self.scaling, values=["Automatic", "Manual"]
        ).grid(row=7, column=1, sticky="w")

        ctk.CTkLabel(main_frame, text="Scan Depth (cm)").grid(
            row=8, column=0, sticky="w"
        )
        ctk.CTkEntry(main_frame, textvariable=self.scan_depth).grid(
            row=8, column=1, sticky="w"
        )

        ctk.CTkCheckBox(
            main_frame, text="Flip Horizontally", variable=self.flip_horizontal
        ).grid(row=9, column=0, sticky="w")
        ctk.CTkCheckBox(
            main_frame, text="Flip Vertically", variable=self.flip_vertical
        ).grid(row=9, column=1, sticky="w")

        ctk.CTkButton(
            main_frame, text="Start Analysis", command=self.start_analysis
        ).grid(row=10, column=0, columnspan=2, pady=10)

        # Canvas for image display
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ne")
        self.canvas = ctk.CTkCanvas(
            self.canvas_frame, width=500, height=500, bg="white"
        )
        self.canvas.grid()

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.bind("<Delete>", self.on_delete_circle)

        for child in main_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)

    def select_input_source(self):
        if self.analysis_type.get() == "Folder":
            directory = filedialog.askdirectory()
            if directory:
                self.input_dir.set(directory)
                self.load_images_from_folder(directory)
        else:
            file_paths = filedialog.askopenfilenames(
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
            )
            if file_paths:
                self.image_list = list(file_paths)
                self.current_image_index = 0
                self.load_image(self.image_list[self.current_image_index])

    def select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def load_images_from_folder(self, folder_path):
        self.image_list = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
        ]
        if self.image_list:
            self.current_image_index = 0
            self.load_image(self.image_list[self.current_image_index])

    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            messagebox.showerror("Error", f"Failed to load image: {image_path}")
            return
        self.image = self.original_image.copy()
        self.display_image(self.image)

    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img)
        img_pil.thumbnail((600, 400))  # Resize to fit within 600x400
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")  # Clear previous image
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk  # Keep reference to avoid garbage collection

    def on_mouse_down(self, event):
        """Start drawing a circle."""
        self.drawing = True
        self.circles.append((event.x, event.y, 0))

    def on_mouse_move(self, event):
        """Update the circle radius as the mouse is dragged."""
        if self.drawing and self.circles:
            cx, cy, _ = self.circles[-1]
            radius = int(np.sqrt((event.x - cx) ** 2 + (event.y - cy) ** 2))
            self.circles[-1] = (cx, cy, radius)

            self.temp_image = self.image.copy()
            cv2.circle(
                self.temp_image, (cx, cy), radius, (255, 0, 255), 2
            )  # Violet circle
            self.display_image(self.temp_image)

    def on_mouse_up(self, event):
        """Finalize the circle drawing."""
        self.drawing = False
        if self.circles:
            cx, cy, radius = self.circles[-1]
            cv2.circle(self.image, (cx, cy), radius, (255, 0, 255), 2)  # Violet circle
            cv2.circle(
                self.mask, (cx, cy), radius, (0, 0, 0), thickness=-1
            )  # Draw filled circle on mask
            self.display_image(self.image)

    def on_delete_circle(self, event):
        """Remove the last drawn circle."""
        if self.circles:
            self.circles.pop()
            self.redraw_circles()

    def redraw_circles(self):
        """Redraw all circles after one has been removed."""
        self.image = self.original_image.copy()
        self.mask = np.ones_like(self.image, dtype=np.uint8) * 255
        for cx, cy, radius in self.circles:
            cv2.circle(self.image, (cx, cy), radius, (255, 0, 255), 2)  # Violet circle
            cv2.circle(
                self.mask, (cx, cy), radius, (0, 0, 0), thickness=-1
            )  # Update mask
        self.display_image(self.image)

    def start_analysis(self):
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        muscle_type = self.muscle.get()
        scaling = self.scaling.get()
        scan_depth = self.scan_depth.get()

        if not input_dir or not output_dir:
            messagebox.showerror(
                "Error", "Please select both input and output directories."
            )
            return

        settings = {"tubeness_sigma": 5, "gaussian_sigma": 2}

        if self.analysis_type.get() == "Image":
            image = cv2.imread(input_dir)
            preprocessed_image = preprocess_image(
                image,
                muscle_type,
                self.flip_horizontal.get(),
                self.flip_vertical.get(),
                settings["tubeness_sigma"],
                settings["gaussian_sigma"],
            )
            starting_points = find_starting_points(
                preprocessed_image, method=self.outline_strategy.get()
            )
            area = measure_area(preprocessed_image, starting_points, scan_depth)
            results = {"Filename": [os.path.basename(input_dir)], "Area (cm²)": [area]}
            cv2.imwrite(
                os.path.join(output_dir, os.path.basename(input_dir)),
                preprocessed_image,
            )

        else:
            results = process_images(
                input_dir,
                output_dir,
                settings,
                muscle_type,
                self.flip_horizontal.get(),
                self.flip_vertical.get(),
                self.outline_strategy.get(),
                scan_depth,
            )

        if self.export_excel.get():
            excel_expo(results, os.path.join(output_dir, "results.xlsx"))

        messagebox.showinfo("Success", "Analysis completed successfully.")


if __name__ == "__main__":
    app = ACSAutoApp()
    app.mainloop()
