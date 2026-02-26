from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from DeepACSA.gui_helpers.image_processing import (
    excel_expo,
    preprocess_image,
    fill_contour_mask,
    area_cm2_from_mask,
    remove_border_scale_bar,
    detect_contour_from_preprocessed_mask,
    find_starting_points,
    measure_area,
)

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

EDGE_SETTINGS = {
    # Muscle CSA: weaker boundaries -> more smoothing, lower Canny
    "Rectus femoris": dict(blur="gauss", blur_k=5, blur_sig=1.4, canny_lo=25, canny_hi=80, dilate=1),
    "Vastus lateralis": dict(blur="gauss", blur_k=5, blur_sig=1.6, canny_lo=20, canny_hi=75, dilate=1),
    "Quad RF": dict(blur="gauss", blur_k=5, blur_sig=1.4, canny_lo=25, canny_hi=80, dilate=1),
    "Quad VL": dict(blur="gauss", blur_k=5, blur_sig=1.6, canny_lo=20, canny_hi=75, dilate=1),
    "Quadriceps": dict(blur="gauss", blur_k=7, blur_sig=1.8, canny_lo=20, canny_hi=70, dilate=1),
    "Gastro MED": dict(blur="median", blur_k=5, blur_sig=None, canny_lo=20, canny_hi=70, dilate=1),

    # Tendon: sharper, brighter lines -> less smoothing, higher Canny
    "Patellar tendon": dict(blur="gauss", blur_k=3, blur_sig=1.0, canny_lo=35, canny_hi=120, dilate=1),
}


# =============================================================================
# Helpers
# =============================================================================

def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def safe_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("safe_gray: img is None")
    if img.ndim == 2:
        return img.astype(np.uint8, copy=False)
    if img.ndim == 3:
        c = img.shape[2]
        if c == 1:
            return img[:, :, 0].astype(np.uint8, copy=False)
        if c == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if c == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"safe_gray: unsupported shape {img.shape}")


def ensure_gray2d(img: np.ndarray) -> np.ndarray:
    g = safe_gray(img)
    if g.ndim != 2:
        raise ValueError(f"ensure_gray2d failed, got shape {g.shape}")
    return g


def as_int(x, *, name="value") -> int:
    try:
        return int(round(float(x)))
    except Exception as e:
        raise ValueError(f"{name} must be numeric, got {x!r}") from e


def to_tk_photo_rgb(rgb: np.ndarray, size: tuple[int, int]) -> ImageTk.PhotoImage:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"to_tk_photo_rgb expects RGB image, got {rgb.shape}")
    pil = Image.fromarray(rgb)
    pil.thumbnail(size)
    return ImageTk.PhotoImage(pil)


def mask_outname(original_name: str) -> str:
    stem, _ = os.path.splitext(original_name)
    return f"{stem}_mask.png"

def _blur_u8(gray: np.ndarray, kind: str, k: int, sig: float | None) -> np.ndarray:
    k = int(k) | 1
    if kind == "median":
        return cv2.medianBlur(gray, k)
    if kind == "gauss":
        return cv2.GaussianBlur(gray, (k, k), float(sig or 0))
    return gray

def build_edge_map(gray_u8: np.ndarray, muscle_type: str) -> np.ndarray:
    s = EDGE_SETTINGS.get(muscle_type, EDGE_SETTINGS["Rectus femoris"])

    g = gray_u8
    g = _blur_u8(g, s["blur"], s["blur_k"], s.get("blur_sig", None))

    edges = cv2.Canny(g, threshold1=int(s["canny_lo"]), threshold2=int(s["canny_hi"]))
    # optional small dilation to connect broken edges
    if int(s.get("dilate", 0)) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, k, iterations=int(s["dilate"]))

    # enforce binary 0/255
    edges = (edges > 0).astype(np.uint8) * 255
    return edges

def rays_hit_edges(edge_u8: np.ndarray,
                   start_points: list[tuple[int, int]],
                   *,
                   max_range_px: int | None = None,
                   n_rays: int = 360) -> list[tuple[int, int]]:
    h, w = edge_u8.shape[:2]
    is_edge = (edge_u8 > 0)

    if max_range_px is None:
        max_range_px = int(np.hypot(w, h))

    hits: list[tuple[int, int]] = []

    for (x0, y0) in start_points:
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))

        for ang in np.linspace(0, 2 * np.pi, int(n_rays), endpoint=False):
            dx, dy = np.cos(ang), np.sin(ang)

            for d in range(1, max_range_px):
                x = int(round(x0 + d * dx))
                y = int(round(y0 + d * dy))
                if x < 0 or y < 0 or x >= w or y >= h:
                    break
                if is_edge[y, x]:
                    hits.append((x, y))
                    break

    return hits

def poly_from_hits(hits: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(hits) < 3:
        return []
    pts = np.array(hits, dtype=np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    return [(int(p[0][0]), int(p[0][1])) for p in hull]

def ray_outline_from_mask(mask_u8: np.ndarray,
                          start_points: list[tuple[int, int]],
                          *,
                          n_rays: int = 360) -> list[tuple[int, int]]:
    """
    Rays start inside foreground (mask>0) and march outward until they leave it.
    Returns a polygon-like point set (convex hull of boundary hits).
    """
    h, w = mask_u8.shape[:2]
    fg = (mask_u8 > 0)

    hits: list[tuple[int, int]] = []

    for (x0, y0) in start_points:
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        if not fg[y0, x0]:
            # If user clicked outside, skip this point (or you could project to nearest fg)
            continue

        max_d = int(np.hypot(w, h))  # safe upper bound

        for ang in np.linspace(0, 2 * np.pi, int(n_rays), endpoint=False):
            dx, dy = np.cos(ang), np.sin(ang)
            last_in = (x0, y0)

            for d in range(1, max_d):
                x = int(round(x0 + d * dx))
                y = int(round(y0 + d * dy))
                if x < 0 or y < 0 or x >= w or y >= h:
                    hits.append(last_in)
                    break
                if fg[y, x]:
                    last_in = (x, y)
                else:
                    # just left foreground
                    hits.append(last_in)
                    break

    if len(hits) < 3:
        return []

    pts = np.array(hits, dtype=np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts)
    return [(int(p[0][0]), int(p[0][1])) for p in hull]
# =============================================================================
# App
# =============================================================================

class ACSAutoApp(ctk.CTk):
    """
    - Removed the contour deletion stage completely (per request).
    - Always shows final overlay automatically at the end (on original image).
    - Toggle to hide overlay completely.
    - Slightly nicer UI layout (grouped sections), background/theme unchanged.
    """
    def __init__(self):
        super().__init__()

        self.title("ACSAuto - Anatomical Cross-Sectional Area Analysis")
        self.geometry("1180x700")
        self.configure(padx=0, pady=0)

        master_path = os.path.dirname(os.path.abspath(__file__))
        theme_path = os.path.join(master_path, "..", "gui_helpers", "gui_files", "ui_color_theme.json")
        ctk.set_default_color_theme(theme_path)

        # close safely
        self._stop_requested = False
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

        # UI vars
        self.analysis_type = ctk.StringVar(value="Folder")
        self.export_excel = ctk.BooleanVar(value=True)
        self.input_dir = ctk.StringVar()
        self.output_dir = ctk.StringVar()

        self.muscle = ctk.StringVar(value="Rectus femoris")
        self.starting_point_mode = ctk.StringVar(value="Automatic")  # Automatic | Manual
        self.scaling = ctk.StringVar(value="None")              # Automatic | Manual
        self.scan_depth = ctk.DoubleVar(value=5.0)

        self.flip_horizontal = ctk.BooleanVar(value=False)
        self.flip_vertical = ctk.BooleanVar(value=False)

        # overlay option (OFF = completely remove overlay)
        self.show_final_overlay = ctk.BooleanVar(value=True)

        # file state
        self.image_list: list[str] = []
        self.selected_image_paths: list[str] = []
        self.cur_path: str | None = None

        # image shown on canvas
        self.img_gray: np.ndarray | None = None
        self._orig_gray_current: np.ndarray | None = None
        self._bin_gray_current: np.ndarray | None = None

        # mapping
        self._disp_w = 1
        self._disp_h = 1
        self._scale = 1.0
        self._canvas_w = 1
        self._canvas_h = 1

        # calibration
        self.px_per_cm: float | None = None
        self.calib_points: list[tuple[int, int]] = []
        self._calib_mode = False

        # modes: VIEW | START_POINTS | EDIT_CONTOUR
        self.mode = "VIEW"

        # starting points stage
        self.sp_points: list[tuple[int, int]] = []
        self._sp_done_var = tk.BooleanVar(value=False)
        self._sp_accept = False

        # edit stage (optional; we keep it so you can still add/move vertices if you want)
        self.edit_poly: list[tuple[int, int]] = []
        self._edit_drag_idx: int | None = None
        self._edit_done_var = tk.BooleanVar(value=False)
        self._edit_accept = False

        # last final contour for automatic overlay
        self._last_final_poly: list[tuple[int, int]] = []

        # detection ray seed activstion
        self.seed_from_rays = ctk.BooleanVar(value=True)

        self.create_ui()

    # -------------------------------------------------------------------------
    # Close handling
    # -------------------------------------------------------------------------
    def _on_close_request(self):
        self._stop_requested = True
        try:
            self._sp_done_var.set(True)
            self._edit_done_var.set(True)
        except Exception:
            pass
        self.quit()

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------
    def create_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        main = ctk.CTkFrame(self)
        main.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        right = ctk.CTkFrame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # --- nicer grouping (no background changes) ---
        io = ctk.CTkFrame(main)
        io.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 6))
        io.grid_columnconfigure(1, weight=1)

        opts = ctk.CTkFrame(main)
        opts.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=6)
        opts.grid_columnconfigure(1, weight=1)

        run = ctk.CTkFrame(main)
        run.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(6, 10))
        run.grid_columnconfigure(0, weight=1)

        # --- I/O ---
        ctk.CTkLabel(io, text="Input").grid(row=0, column=0, sticky="w")
        ctk.CTkRadioButton(io, text="Folder", variable=self.analysis_type, value="Folder",
                           command=self._on_analysis_type_changed).grid(row=0, column=1, sticky="w")
        ctk.CTkRadioButton(io, text="Image", variable=self.analysis_type, value="Image",
                           command=self._on_analysis_type_changed).grid(row=0, column=2, sticky="w")

        ctk.CTkButton(io, text="Browse…", command=self.select_input_source).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.lbl_in = ctk.CTkLabel(io, text="", anchor="w")
        self.lbl_in.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(6, 0))

        ctk.CTkButton(io, text="Output…", command=self.select_output_dir).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.lbl_out = ctk.CTkLabel(io, text="", anchor="w")
        self.lbl_out.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(6, 0))

        ctk.CTkCheckBox(io, text="Export to Excel", variable=self.export_excel).grid(row=3, column=0, sticky="w", pady=(6, 0))

        # --- Options ---
        ctk.CTkLabel(opts, text="Muscle Type").grid(row=0, column=0, sticky="w")
        ctk.CTkOptionMenu(
            opts,
            variable=self.muscle,
            values=[
                "Rectus femoris",
                "Vastus lateralis",
                "Quad RF",
                "Quad VL",
                "Quadriceps",
                "Gastro MED",
                "Patellar tendon",
            ],
        ).grid(row=0, column=1, sticky="ew")

        ctk.CTkLabel(opts, text="Starting Points").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ctk.CTkOptionMenu(opts, variable=self.starting_point_mode,
                          values=["Automatic", "Manual"]).grid(row=1, column=1, sticky="ew", pady=(6, 0))

        ctk.CTkLabel(opts, text="Scaling").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ctk.CTkOptionMenu(opts, variable=self.scaling,
                          values=["None", "Manual"]).grid(row=2, column=1, sticky="ew", pady=(6, 0))
        ctk.CTkButton(opts, text="Calibrate…", command=self.calibrate_scaling).grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(6, 0))

        ctk.CTkLabel(opts, text="Scan Depth (cm)").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ctk.CTkEntry(opts, textvariable=self.scan_depth).grid(row=3, column=1, sticky="ew", pady=(6, 0))

        ctk.CTkCheckBox(opts, text="Flip Horizontally", variable=self.flip_horizontal).grid(row=4, column=0, sticky="w", pady=(6, 0))
        ctk.CTkCheckBox(opts, text="Flip Vertically", variable=self.flip_vertical).grid(row=4, column=1, sticky="w", pady=(6, 0))

        ctk.CTkCheckBox(
            opts,
            text="Show final overlay",
            variable=self.show_final_overlay,
            command=self._render,
        ).grid(row=5, column=0, sticky="w", pady=(6, 0))

        ctk.CTkCheckBox(
            opts,
            text="Seed contour from rays (starting points affect overlay)",
            variable=self.seed_from_rays,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))

        # --- Run + status ---
        ctk.CTkButton(run, text="Start Analysis", command=self.start_analysis).grid(row=0, column=0, sticky="ew")
        btn_row = ctk.CTkFrame(run)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(btn_row, text="Clear vertices", command=self.clear_vertices).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.status = ctk.CTkLabel(run, text="Ready.", justify="left", font=("Verdana", 10), text_color="#FFFFFF")
        self.status.grid(row=3, column=0, sticky="w", pady=(6, 0))

        # Canvas
        self.canvas = tk.Canvas(right, bg="white", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda e: self._render())
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)

        self.bind("<Return>", self.on_enter_key)
        self.bind("<Escape>", self.on_escape_key)

        self._on_analysis_type_changed()

    def _set_status(self, text: str):
        try:
            self.status.configure(text=text)
        except Exception:
            pass

    def _on_analysis_type_changed(self):
        self.input_dir.set("")
        self.selected_image_paths = []
        self.image_list = []
        self.cur_path = None
        self.img_gray = None
        self._orig_gray_current = None
        self._bin_gray_current = None
        self.mode = "VIEW"
        self._last_final_poly = []
        self.canvas.delete("all")
        self._set_status("Ready.")
        self.lbl_in.configure(text="")
        self.lbl_out.configure(text="")

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------
    def select_input_source(self):
        if self.analysis_type.get() == "Folder":
            directory = filedialog.askdirectory()
            if directory:
                self.input_dir.set(directory)
                self.lbl_in.configure(text=directory)
                self._load_images_from_folder(directory)
        else:
            file_paths = filedialog.askopenfilenames(
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
            )
            if file_paths:
                self.selected_image_paths = list(file_paths)
                self.image_list = self.selected_image_paths[:]
                self.lbl_in.configure(text=f"{len(self.image_list)} image(s) selected")
                self._load_image(self.image_list[0])

    def select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)
            self.lbl_out.configure(text=directory)

    def _load_images_from_folder(self, folder_path: str):
        files = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if f.lower().endswith(SUPPORTED_EXT)
        ]
        if not files:
            messagebox.showwarning("No images", "No supported image files found in that folder.")
            return
        self.image_list = files
        self._load_image(self.image_list[0])

    def _load_image(self, image_path: str):
        self.cur_path = os.path.abspath(image_path)
        bgr = load_bgr(image_path)
        self.img_gray = ensure_gray2d(bgr)
        self._orig_gray_current = self.img_gray.copy()
        self.mode = "VIEW"
        self._render()

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------
    def calibrate_scaling(self):
        if self.img_gray is None:
            messagebox.showerror("Error", "Load an image first (so you can click on it).")
            return
        self.calib_points = []
        self._calib_mode = True
        messagebox.showinfo(
            "Manual scaling",
            "Click TWO points with known distance.\nAfter second click you will be asked for cm."
        )

    # -------------------------------------------------------------------------
    # Mapping + Render
    # -------------------------------------------------------------------------
    def _compute_display_mapping(self):
        assert self.img_gray is not None
        h, w = self.img_gray.shape[:2]
        cw = max(1, int(self.canvas.winfo_width()))
        ch = max(1, int(self.canvas.winfo_height()))
        self._scale = min(cw / w, ch / h, 1.0)
        self._disp_w = max(1, as_int(w * self._scale))
        self._disp_h = max(1, as_int(h * self._scale))
        self._canvas_w = cw
        self._canvas_h = ch

    def _canvas_to_image_xy(self, x_canvas: int, y_canvas: int) -> tuple[int, int] | None:
        if self.img_gray is None:
            return None
        if x_canvas < 0 or y_canvas < 0 or x_canvas >= self._disp_w or y_canvas >= self._disp_h:
            return None
        h, w = self.img_gray.shape[:2]
        x_img = int(round(x_canvas / max(self._scale, 1e-9)))
        y_img = int(round(y_canvas / max(self._scale, 1e-9)))
        return int(np.clip(x_img, 0, w - 1)), int(np.clip(y_img, 0, h - 1))

    def _nearest_point_idx(self, pts: list[tuple[int, int]], ix: int, iy: int, max_dist_px: int = 15) -> int | None:
        if not pts:
            return None
        best_i, best_d2 = None, None
        for i, (x, y) in enumerate(pts):
            d2 = (x - ix) ** 2 + (y - iy) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2, best_i = d2, i
        if best_d2 is not None and best_d2 <= max_dist_px ** 2:
            return best_i
        return None

    def _render(self):
        if self.img_gray is None:
            self.canvas.delete("all")
            return

        self._compute_display_mapping()
        rgb = cv2.cvtColor(self.img_gray, cv2.COLOR_GRAY2RGB)

        # scale-aware overlay thickness (prevents vanishing when scaled down)
        t = max(2, int(round(2.0 / max(self._scale, 1e-6))))
        rad = max(3, int(round(4.0 / max(self._scale, 1e-6))))

        # START_POINTS overlay (cyan points)
        if self.mode == "START_POINTS":
            for (x, y) in self.sp_points:
                cv2.circle(rgb, (int(x), int(y)), rad, (0, 255, 255), -1)
                cv2.circle(rgb, (int(x), int(y)), rad + 2, (0, 0, 0), 1)

        # EDIT overlay polygon (live editing)
        if self.mode == "EDIT_CONTOUR" and len(self.edit_poly) >= 2:
            pts = np.array(self.edit_poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(rgb, [pts], True, (0, 255, 0), t)
            for (x, y) in self.edit_poly:
                cv2.circle(rgb, (int(x), int(y)), rad, (255, 0, 255), -1)

        # FINAL OVERLAY (automatic at end, on original image view)
        if self.show_final_overlay.get() and self._last_final_poly and self.mode != "EDIT_CONTOUR":
            if len(self._last_final_poly) >= 2:
                pts = np.array(self._last_final_poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(rgb, [pts], True, (0, 255, 0), t)
                for (x, y) in self._last_final_poly:
                    cv2.circle(rgb, (int(x), int(y)), rad, (255, 0, 255), -1)

        disp = cv2.resize(rgb, (self._disp_w, self._disp_h), interpolation=cv2.INTER_AREA)
        canvas_rgb = np.full((self._canvas_h, self._canvas_w, 3), 255, dtype=np.uint8)
        canvas_rgb[: self._disp_h, : self._disp_w] = disp

        tk_img = to_tk_photo_rgb(canvas_rgb, size=(self._canvas_w, self._canvas_h))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self.canvas.image = tk_img

    def clear_vertices(self):
        """Remove all vertices from the editable polygon (only meaningful in EDIT_CONTOUR)."""
        if self.mode != "EDIT_CONTOUR":
            self._set_status("Clear vertices is only available while editing the contour.")
            return
        self.edit_poly = []
        self._edit_drag_idx = None
        self._render()
        self._set_status("All vertices cleared. Click to add new vertices, Enter to accept, Esc to use auto.")

    def toggle_overlay(self):
        self.show_final_overlay.set(not self.show_final_overlay.get())
        self._render()

    def _poly_from_rays(self, bin_img_u8: np.ndarray, sp: list[tuple[int, int]], scan_depth_cm: float) -> list[tuple[int, int]]:
        """
        Build a polygon seed using the ray method:
        - uses measure_area() which internally shoots rays and collects contour points
        - we re-run a *lightweight* version: ray endpoints -> convex hull
        """
        # We replicate ray endpoints here (fast) so we can build a hull for overlay.
        h, w = bin_img_u8.shape[:2]
        depth_px = max(5, int(round(scan_depth_cm * 50)))  # fallback heuristic if scan_depth isn't pixels
        # NOTE: if you have px/cm, use it:
        if self.scaling.get() == "Manual" and self.px_per_cm:
            depth_px = max(5, int(round(scan_depth_cm * self.px_per_cm)))

        contour_pts: list[tuple[int, int]] = []
        num_rays = 360

        for (x0, y0) in sp:
            for ang in np.linspace(0, 2 * np.pi, num_rays, endpoint=False):
                dx, dy = np.cos(ang), np.sin(ang)
                hit = None
                for d in range(depth_px):
                    x = int(round(x0 + d * dx))
                    y = int(round(y0 + d * dy))
                    if x < 0 or y < 0 or x >= w or y >= h:
                        hit = (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))
                        break
                    if bin_img_u8[y, x] > 0:  # edge hit
                        hit = (x, y)
                        break
                if hit is not None:
                    contour_pts.append(hit)

        if len(contour_pts) < 3:
            return []

        pts = np.array(contour_pts, dtype=np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)  # stable polygon seed
        return [(int(p[0][0]), int(p[0][1])) for p in hull]
    
    # -------------------------------------------------------------------------
    # Keys
    # -------------------------------------------------------------------------
    def on_enter_key(self, event=None):
        if self.mode == "START_POINTS":
            self._sp_accept = True
            self._sp_done_var.set(True)
            return
        if self.mode == "EDIT_CONTOUR":
            self._edit_accept = True
            self._edit_done_var.set(True)
            return

    def on_escape_key(self, event=None):
        if self.mode == "START_POINTS":
            self._sp_accept = False
            self._sp_done_var.set(True)
            return
        if self.mode == "EDIT_CONTOUR":
            self._edit_accept = False
            self._edit_done_var.set(True)
            return
        if self._calib_mode:
            self._calib_mode = False
            self.calib_points = []
            self._set_status("Calibration cancelled.")
            return

    # -------------------------------------------------------------------------
    # Mouse
    # -------------------------------------------------------------------------
    def on_right_click(self, event):
        # START_POINTS: right-click removes nearest point
        if self.mode == "START_POINTS":
            xy = self._canvas_to_image_xy(event.x, event.y)
            if xy is None:
                return
            ix, iy = xy
            j = self._nearest_point_idx(self.sp_points, ix, iy)
            if j is not None:
                self.sp_points.pop(j)
                self._render()
            return

        # EDIT_CONTOUR: right-click deletes nearest vertex
        if self.mode == "EDIT_CONTOUR":
            xy = self._canvas_to_image_xy(event.x, event.y)
            if xy is None:
                return
            ix, iy = xy
            j = self._nearest_point_idx(self.edit_poly, ix, iy)
            if j is not None:
                self.edit_poly.pop(j)
                self._render()
            return

    def on_mouse_down(self, event):
        if self.img_gray is None:
            return

        # calibration first
        if self._calib_mode:
            xy = self._canvas_to_image_xy(event.x, event.y)
            if xy is None:
                return
            self.calib_points.append(xy)
            if len(self.calib_points) == 2:
                (x1, y1), (x2, y2) = self.calib_points
                px = float(np.hypot(x2 - x1, y2 - y1))
                print(f"Calibration points: {self.calib_points}, pixel distance: {px:.2f}")
                self._calib_mode = False

                win = ctk.CTkInputDialog(text="Distance between points (cm):", title="Manual scaling")
                cm_str = win.get_input()
                try:
                    cm = float(cm_str)
                    if cm <= 0:
                        raise ValueError
                except Exception:
                    messagebox.showerror("Error", "Invalid cm value.")
                    self.calib_points = []
                    return

                self.px_per_cm = px / cm
                self.calib_points = []
                self._set_status(f"Manual scaling set: {self.px_per_cm:.2f} px/cm")
            return

        # START_POINTS: add point
        if self.mode == "START_POINTS":
            xy = self._canvas_to_image_xy(event.x, event.y)
            if xy is not None:
                self.sp_points.append(xy)
                self._render()
            return

        # EDIT_CONTOUR: add/move vertex
        if self.mode == "EDIT_CONTOUR":
            xy = self._canvas_to_image_xy(event.x, event.y)
            if xy is None:
                return
            ix, iy = xy
            j = self._nearest_point_idx(self.edit_poly, ix, iy)
            if j is not None:
                self._edit_drag_idx = j
            else:
                self.edit_poly.append((ix, iy))
            self._render()
            return

    def on_mouse_move(self, event):
        if self.img_gray is None:
            return
        if self.mode == "EDIT_CONTOUR" and self._edit_drag_idx is not None:
            xy = self._canvas_to_image_xy(event.x, event.y)
            if xy is None:
                return
            self.edit_poly[self._edit_drag_idx] = xy
            self._render()

    def on_mouse_up(self, event):
        if self.mode == "EDIT_CONTOUR":
            self._edit_drag_idx = None

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    def _resolve_paths(self) -> list[str]:
        if self.analysis_type.get() == "Folder":
            indir = self.input_dir.get().strip()
            if not indir or not os.path.isdir(indir):
                raise ValueError("Please select a valid input folder.")
            return [
                os.path.join(indir, f)
                for f in sorted(os.listdir(indir))
                if f.lower().endswith(SUPPORTED_EXT)
            ]
        return self.selected_image_paths or self.image_list

    def start_analysis(self):
        if self._stop_requested:
            return

        outdir = self.output_dir.get().strip()
        if not outdir:
            messagebox.showerror("Error", "Please select an output directory.")
            return
        os.makedirs(outdir, exist_ok=True)

        if self.scaling.get() == "Manual" and (self.px_per_cm is None or self.px_per_cm <= 0):
            messagebox.showerror("Error", "Manual scaling selected but not calibrated (px/cm missing).")
            return

        try:
            paths = self._resolve_paths()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if not paths:
            messagebox.showerror("Error", "No images selected/found.")
            return

        muscle_type = self.muscle.get()
        scan_depth_cm = float(self.scan_depth.get())
        settings = {"tubeness_sigma": 5, "gaussian_sigma": 2}

        results_rows: list[dict] = []

        for idx, p in enumerate(paths, start=1):
            if self._stop_requested:
                return
            
            # Clear vertices and reset state for each new image to avoid confusion (especially since we removed the deletion stage).
            self.clear_vertices()

            self._set_status(f"Processing {idx}/{len(paths)}: {os.path.basename(p)}")

            bgr = load_bgr(p)
            orig_gray = ensure_gray2d(bgr)
            self._orig_gray_current = orig_gray

            # Build edge map for ray hits (structure-dependent)
            edge_u8 = build_edge_map(orig_gray, muscle_type)
            # also remove scale bar/border artifacts on edge image (optional but helps)
            edge_u8 = remove_border_scale_bar(edge_u8, thr=240)

            # for manual start-point placement show the EDGE image (not the old binary mask)
            self._bin_gray_current = edge_u8

            # preprocess -> mask
            pre_mask = preprocess_image(
                bgr,
                muscle_type,
                self.flip_horizontal.get(),
                self.flip_vertical.get(),
                settings["tubeness_sigma"],
                settings["gaussian_sigma"],
            )
            pre_mask = ensure_gray2d(pre_mask)

            if pre_mask.shape[:2] != orig_gray.shape[:2]:
                pre_mask = cv2.resize(pre_mask, (orig_gray.shape[1], orig_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

            # pre_for_contours = remove_border_scale_bar(pre_mask, thr=240)
            # _, bin_img = cv2.threshold(pre_for_contours, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # bin_img_u8 = bin_img.astype(np.uint8, copy=False)
            # self._bin_gray_current = bin_img_u8

            # ---------------- STARTING POINTS stage (binary display) ----------------
            if self.starting_point_mode.get() == "Manual":
                self.mode = "START_POINTS"
                self.img_gray = self._bin_gray_current
                self.sp_points = []
                self._sp_accept = False
                self._sp_done_var.set(False)
                self._render()
                self._set_status("Starting points: Left=add, Right=remove, Enter=accept, Esc=skip(auto).")
                self.wait_variable(self._sp_done_var)

                if self._stop_requested:
                    return

                if self._sp_accept and len(self.sp_points) > 0:
                    sp = self.sp_points[:]
                else:
                    sp, _ = find_starting_points(self._bin_gray_current.copy(), method="Automatic")
            else:
                sp, _ = find_starting_points(self._bin_gray_current.copy(), method="Automatic")

            area_rays = measure_area(self._bin_gray_current.copy(), sp, scan_depth=as_int(scan_depth_cm, name="scan_depth_cm"))

            # ---------------- Seed polygon (automatic) ----------------
            # Use the DeepACSA helper to get a contour directly from preprocessed mask.
            # This avoids the broken deletion step and gives you a stable auto contour.
            cnt_auto, _ = detect_contour_from_preprocessed_mask(pre_mask.copy())
            poly_seed: list[tuple[int, int]] = []

            if self.seed_from_rays.get() and len(sp) > 0:
                #poly_seed = self._poly_from_rays(bin_img_u8, sp, scan_depth_cm)
                #poly_seed = ray_outline_from_mask(self._bin_gray_current.copy(), sp, n_rays=360)
                hits = rays_hit_edges(edge_u8, sp, n_rays=360)
                poly_seed = poly_from_hits(hits)
                
                if poly_seed:
                    self._set_status("Seeded contour from rays (starting points affect overlay).")
                else:
                    self._set_status("Ray seeding produced no polygon, falling back to auto contour.")

            if not poly_seed:
                cnt_auto, _ = detect_contour_from_preprocessed_mask(pre_mask.copy())
                if cnt_auto is not None and len(cnt_auto) >= 3:
                    cnt_auto = np.asarray(cnt_auto).astype(np.int32)
                    if cnt_auto.ndim == 2 and cnt_auto.shape[1] == 2:
                        cnt_auto = cnt_auto.reshape(-1, 1, 2)
                    poly_seed = [(int(pt[0][0]), int(pt[0][1])) for pt in cnt_auto]

            # ---------------- Optional edit stage on ORIGINAL ----------------
            self.img_gray = self._orig_gray_current
            self.mode = "EDIT_CONTOUR"
            self.edit_poly = poly_seed[:]
            self._edit_drag_idx = None
            self._edit_accept = False
            self._edit_done_var.set(False)
            self._render()
            self._set_status("Edit contour: click add/move, Right=delete vertex, Enter=accept, Esc=use auto.")
            self.wait_variable(self._edit_done_var)

            if self._stop_requested:
                return

            final_poly = self.edit_poly[:] if (self._edit_accept and len(self.edit_poly) >= 3) else poly_seed[:]
            self._last_final_poly = final_poly[:]  # stored for automatic overlay

            # compute filled mask + area
            if len(final_poly) >= 3:
                final_cnt = np.array(final_poly, dtype=np.int32).reshape(-1, 1, 2)
                filled = fill_contour_mask(orig_gray.shape[:2], final_cnt)

                area = area_cm2_from_mask(
                    filled,
                    scan_depth_cm=scan_depth_cm,
                    px_per_cm=self.px_per_cm if self.scaling.get() == "Manual" else None,
                )
            else:
                filled = np.zeros_like(orig_gray, dtype=np.uint8)
                area = 0.0

            out_name = os.path.basename(p)
            cv2.imwrite(os.path.join(outdir, mask_outname(out_name)), filled)

            results_rows.append(
                {
                    "Filename": out_name,
                    "Area (cm²)": float(area),
                    "Area_rays (a.u.)": float(area_rays),
                    "N_start_points": int(len(sp)),
                }
            )

            # --- FINAL: show original + overlay automatically ---
            self.mode = "VIEW"
            self.img_gray = self._orig_gray_current
            self._set_status(f"Done: {out_name} | Area={area:.2f} cm² (overlay={'on' if self.show_final_overlay.get() else 'off'})")
            self._render()

        if self.export_excel.get():
            excel_expo(results_rows, os.path.join(outdir, "results.xlsx"))

        self._set_status("All done.")
        messagebox.showinfo("Success", "Analysis completed successfully.")


if __name__ == "__main__":
    app = ACSAutoApp()
    app.mainloop()