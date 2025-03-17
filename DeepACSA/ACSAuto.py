import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import pandas as pd
from PIL import Image, ImageTk

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
        self.geometry("800x600")

        # Variables
        self.analysis_type = tk.StringVar(value="Folder")
        self.export_excel = tk.BooleanVar(value=False)
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.muscle = tk.StringVar(value="Rectus femoris")
        self.outline_strategy = tk.StringVar(value="Automatic")
        self.sorting = tk.BooleanVar(value=True)
        self.scaling = tk.StringVar(value="Automatic")
        self.scan_depth = tk.DoubleVar(value=5)
        self.flip_horizontal = tk.BooleanVar(value=False)
        self.flip_vertical = tk.BooleanVar(value=False)

        self.circles = []
        self.drawing = False

        # UI Setup
        self.create_ui()

    def create_ui(self):
        ctk.CTkLabel(self, text="Type of Analysis").pack()
        ctk.CTkRadioButton(
            self, text="Folder", variable=self.analysis_type, value="Folder"
        ).pack()
        ctk.CTkRadioButton(
            self, text="Image", variable=self.analysis_type, value="Image"
        ).pack()

        ctk.CTkCheckBox(self, text="Export to Excel", variable=self.export_excel).pack()

        ctk.CTkLabel(self, text="Input Directory").pack()
        ctk.CTkButton(self, text="Browse", command=self.select_input_dir).pack()

        ctk.CTkLabel(self, text="Output Directory").pack()
        ctk.CTkButton(self, text="Browse", command=self.select_output_dir).pack()

        ctk.CTkLabel(self, text="Muscle Type").pack()
        ctk.CTkOptionMenu(
            self,
            variable=self.muscle,
            values=[
                "Rectus femoris",
                "Vastus lateralis",
                "Quad RF",
                "Quad VL",
                "Quadriceps",
                "Gastro MED",
            ],
        ).pack()

        ctk.CTkLabel(self, text="Outline Finder Strategy").pack()
        ctk.CTkOptionMenu(
            self,
            variable=self.outline_strategy,
            values=["Manual", "Automatic", "Fixed Pixels"],
        ).pack()

        ctk.CTkCheckBox(self, text="Sort Coordinates", variable=self.sorting).pack()

        ctk.CTkLabel(self, text="Scaling").pack()
        ctk.CTkOptionMenu(
            self, variable=self.scaling, values=["Automatic", "Manual"]
        ).pack()

        ctk.CTkLabel(self, text="Scan Depth (cm)").pack()
        ctk.CTkEntry(self, textvariable=self.scan_depth).pack()

        ctk.CTkCheckBox(
            self, text="Flip Horizontally", variable=self.flip_horizontal
        ).pack()
        ctk.CTkCheckBox(
            self, text="Flip Vertically", variable=self.flip_vertical
        ).pack()

        ctk.CTkButton(self, text="Start Analysis", command=self.start_analysis).pack(
            pady=20
        )

        # Canvas for image display and drawing
        self.canvas_frame = ctk.CTkFrame(self, width=600, height=400)
        self.canvas_frame.pack(pady=10)
        self.canvas = ctk.CTkCanvas(
            self.canvas_frame, width=600, height=400, bg="white"
        )
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.bind("<Delete>", self.on_delete_circle)

    def select_input_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.input_dir.set(directory)
            # self.load_image(directory)

    def select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def load_image(self, image_path):
        # Load and display the image on the canvas
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image = self.original_image.copy()
        self.temp_image = self.image.copy()
        self.mask = np.ones_like(self.image, dtype=np.uint8) * 255
        self.display_image(self.image)

    def display_image(self, img):
        """Display the given image on the canvas."""
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection

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
