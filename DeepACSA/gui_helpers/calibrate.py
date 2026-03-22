"""
Description
-----------
Image scaling utilities.

This module contains functions to automatically or manually scale
ultrasound images. The automatic method requires scaling bars to be
present on the right side of the image. The manual method allows
scaling based on user-selected points, provided that the real-world
distance between those points is known.
"""

import math
import tkinter as tk

from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2


mlocs = []


def region_of_interest(img: np.ndarray, vertices: np.ndarray):
    """
    Function to crop the images according to a specified
    region of interest. The input image is cropped and
    the region of interest is returned.

    Parameters
    ----------
    img : np.ndarray
        Input image likely to be already processed and
        solely containing edges.
    vertices : np.ndarray
        Numpy array containing the vertices that translate
        to the region / coordinates for image cropping.

    Returns
    -------
    masked_img : np.ndarray
        Image cropped to the pre-specified region of interest.

    Notes
    -----
    - The `img` parameter should be a processed image, typically containing edges,
      where the region of interest needs to be cropped.
    - The `vertices` parameter should be an array of vertices that define a polygon.
      The function will crop the input image based on the shape defined by these vertices.
    - The function uses a mask to retain only the region of interest in the input image,
      based on the defined vertices. Pixels outside the polygon will be set to black (0),
      while pixels inside the polygon will retain their original values.
    - The function does not modify the original image; instead, it returns a cropped
      version containing the region of interest.

    Examples
    --------
    >>> region_of_interest(preprocessed_image,
                        ([(0,1), (0,2), (4,2), (4,7)], np.int32),)
    """
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def mclick(event, x_val, y_val, flags, param):
    """
    Instance method to detect mouse click coordinates in image.

    This instance is used when the image to be analyzed should be
    cropped. Upon clicking the mouse button, the coordinates
    of the cursor position are stored in the instance attribute
    self.mlocs.

    Parameters
    ----------
    event : int
        Event flag specified as Cv2 mouse event left mouse button down.
    x_val : int
        Value of x-coordinate of mouse event to be recorded.
    y_val : int
        Value of y-coordinate of mouse event to be recorded.
    flags : int
        Specific condition whenever a mouse event occurs. This
        is not used here but needs to be specified as input
        parameter.
    param : any
        User input data. This is not required here but needs to
        be specified as input parameter.

    Returns
    -------
    None

    Notes
    -----
    - This function is intended to be used with the `setMouseCallback` function
      from OpenCV. It allows the function to capture mouse click events in an image.
    - The recorded (x, y) coordinates of mouse clicks are stored in the global
      variable `mlocs`. The variable `mlocs` should be defined before using this
      function and should be accessible to other parts of the code.
    - The global variable `mlocs` can accumulate coordinates of multiple mouse
      clicks across different mouse events.

    Examples
    --------
    # Before using the function, define the global variable `mlocs`.
    mlocs = []

    # Set the mouse callback function for an image.
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", mclick)

    # Wait for a key press to exit the image window.
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Now, the `mlocs` variable contains the recorded mouse click coordinates.
    # Note that `mlocs` may contain multiple sets of coordinates if multiple
    # clicks were made.
    """
    # Define global variable for functions to access
    global mlocs
    global img2

    # if the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        mlocs.append((x_val, y_val))
        # Draw a red dot on the image at the clicked position
        cv2.circle(img2, (x_val, y_val), 3, (0, 0, 255), -1)
        cv2.imshow("image", img2)


def draw_the_lines(img: np.ndarray, line: np.ndarray):
    """
    Function to highlight lines on the input images.
    This is used to visualize the lines on the input image.

    Parameters
    ----------
    img : np.ndarray
        Input image where the lines are to be drawn upon.
    line : np.ndarray
        An array of lines wished to be drawn upon the image.

    Returns
    -------
    img : np.ndarray
        Input image now with lines drawn upon.

    Notes
    -----
    - The function creates a blank image with the same dimensions as the input
      image to draw the lines.
    - The lines are drawn using the cv2.line() function from OpenCV, and they are
      colored in green with a thickness of 3 pixels.
    - The original input image is not modified; instead, the lines are overlaid
      on a copy of the input image to preserve the original.

    Example
    -------
    >>> draw_the_lines(Image1.tif, ([[0 738 200 539]))
    """
    img = np.copy(img)
    # Creating empty image to draw lines on
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for x_1, y_1, x_2, y_2 in line:
        cv2.line(blank_image, (x_1, y_1), (x_2, y_2), (0, 255, 0), thickness=3)

    # Overlay image with lines on original images (only needed for plotting)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def calibrate_distance_efov(path_to_image: str, arg_muscle: str):
    """
    Function to calibrate EFOV ultrasonography images automatically.
    This is done by determining the length of the previously detected
    scalining lines present in the image.

    This function is highly specific and limited to EFOV images containing
    lines with scaling bars. Simple scaling bars will not work. For each muscle
    included in DeepACSA, different parameters are used for the Hugh algorithm.

    Parameters
    ----------
    path_to_image : str
        String variable containing the to image that should be analyzed.
    arg_muscle : str
        String variable containing the muscle present in the analyzed image.

    Returns
    -------
    scalingline_lenght : int
        Integer variable containing the length of the detected scaling line in
        pixel units.
    image_with_lines : np.ndarray
        Cropped image containing the drawn lines.

    Notes
    -----
    - The function automatically detects and calibrates EFOV images containing scaling bars.
    - The muscle parameter is used to specify the muscle type, which influences the line detection parameters.
    - The function uses OpenCV's HoughLinesP function for line detection.

    Example
    -------
    >>> calibrate_distance_efov(C:/Desktop/Test/Img1.tif, "RF")
    571
    """
    image = cv2.imread(path_to_image)
    # Transform BGR Image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    width = image.shape[1]
    # Define ROI with scaling lines
    region_of_interest_vertices = [
        (10, height),
        (10, height * 0.1),
        (width, height * 0.1),
        (width, height),
    ]
    # Transform RGB to greyscale for edge detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Edge detecition
    canny_image = cv2.Canny(gray_image, 400, 600)
    cropped_image = region_of_interest(
        canny_image,
        np.array([region_of_interest_vertices], np.int32),
    )

    # For RF
    muscle = arg_muscle
    if muscle == "RF":
        lines = cv2.HoughLinesP(
            cropped_image,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            lines=np.array([]),
            minLineLength=325,
            maxLineGap=3,
        )
        if lines is None:
            return None, None
        # draw lines on image
        image_with_lines = draw_the_lines(image, lines[0])

    # For VL
    elif muscle == "VL":
        lines = cv2.HoughLinesP(
            cropped_image,
            rho=1,  # Distance of pixels in accumulator
            theta=np.pi / 180,  # Angle resolution
            threshold=50,  # Only lines with higher vote
            lines=np.array([]),
            minLineLength=175,
            maxLineGap=3,
        )  # Gap between lines
        if lines is None:
            return None, None
        # draw lines on image
        image_with_lines = draw_the_lines(image, lines[0])

    # For GM / GL
    elif muscle == "GL":
        lines = cv2.HoughLinesP(
            cropped_image,
            rho=1,  # Distance of pixels in accumulator
            theta=np.pi / 180,  # Angle resolution
            threshold=50,  # Only lines with higher vote
            lines=np.array([]),
            minLineLength=250,
            maxLineGap=5,
        )
        if lines is None:
            return None, None
        # draw scaling lines on image
        image_with_lines = draw_the_lines(image, lines[0])

    # For BF
    elif muscle == "BF":
        lines = cv2.HoughLinesP(
            cropped_image,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            lines=np.array([]),
            minLineLength=325,
            maxLineGap=3,
        )
        if lines is None:
            return None, None
        # draw lines on image
        image_with_lines = draw_the_lines(image, lines[0])

    else:
        lines = cv2.HoughLinesP(
            cropped_image,
            rho=1,  # Distance of pixels in accumulator
            theta=np.pi / 180,  # Angle resolution
            threshold=50,  # Only lines with higher vote
            lines=np.array([]),
            minLineLength=250,
            maxLineGap=5,
        )
        if lines is None:
            return None, None
        # draw scaling lines on image
        image_with_lines = draw_the_lines(image, lines[0])

    # Calculate length of the scaling line
    scalingline = lines[0][0]
    point1 = [scalingline[0], scalingline[1]]
    point2 = [scalingline[2], scalingline[3]]
    scalingline_length = math.sqrt(
        ((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2)
    )

    return scalingline_length, image_with_lines


def calibrate_distance_static(nonflipped_img: np.ndarray, spacing: int):
    """
    Function to calibrate an image to convert measurements
    in pixel units to centimeter.

    The function calculates the distance in pixel units between two
    scaling bars on the input image. The bars should be positioned on the
    right side of image. The distance (in milimeter) between two bars must
    be specified by the spacing variable. It is the known distance between two
    bars in milimeter. Then the ratio of pixel / centimeter is calculated.
    To get the distance, the median distance between two detected bars
    is calculated.

    Parameters
    ----------
    nonflipped_img : np.ndarray
        Input image to be analysed as a numpy array. The image must
        be loaded prior to calibration, specifying a path
        is not valid.
    spacing : int
        Integer variable containing the known distance in milimeter
        between the two scaling bars. This can be 5, 10,
        15 or 20 milimeter.

    Returns
    -------
    calib_dist : int
        Integer variable containing the distance between the two
        specified point in pixel units.
    imgscale : np.ndarray
        Cropped region of the input area containing only the
        scaling bars.
    scale_statement : str
        String variable containing a statement how many milimeter
        correspond to how many pixels.

    Notes
    -----
    - The function calibrates the image using the scaling bars present on the right side of the image.
    - The spacing parameter should be provided as an integer, not a string.

    Examples
    --------
    >>> calibrateDistanceStatic(img=([[[[0.22414216 0.19730392 0.22414216] ... [0.2509804  0.2509804  0.2509804 ]]]), 5)
    99, 5 mm corresponds to 99 pixels
    """
    global img2
    # calibrate according to scale at the right border of image
    img2 = np.uint8(nonflipped_img)

    height = img2.shape[0]
    width = img2.shape[1]
    imgscale = img2[int(height * 0.4) : (height), (width - int(width * 0.15)) : width]

    # search for rows with white pixels, calculate median of distance
    calib_dist = np.max(np.diff(np.argwhere(imgscale.max(axis=1) > 150), axis=0))

    if int(calib_dist) < 1:
        return None, None, None

    # calculate calib_dist for 10mm
    if spacing == "5":
        calib_dist = calib_dist * 2
    if spacing == "15":
        calib_dist = calib_dist * (2 / 3)
    if spacing == "20":
        calib_dist = calib_dist / 2

    # scalingline_length = depth * calib_dist
    scale_statement = "10 mm corresponds to " + str(calib_dist) + " pixels"

    return calib_dist, imgscale, scale_statement


# def calibrate_distance_manually(nonflipped_img: np.ndarray, spacing: str):
#     """
#     Function to manually calibrate an image to convert measurements
#     in pixel units to centimeters.

#     The function calculates the distance in pixel units between two
#     points on the input image. The points are determined by clicks of
#     the user; only two points may be selected and clicks outside the
#     visible image are ignored to ensure both calibration points remain
#     on the picture. A right mouse click can be used to undo the last
#     point in case of a mistake. The distance (in milimeters) is
#     determined by the value contained in the spacing variable. Then the
#     ratio of pixel / centimeter is calculated. To get the distance, the
#     euclidean distance between the two points is calculated.

#     Parameters
#     ----------
#     nonflipped_img : np.ndarray
#         Input image to be analysed as a numpy array. The image must
#         be loaded prior to calibration, specifying a path
#         is not valid.
#     spacing : int
#         Integer variable containing the known distance in milimeter
#         between the two placed points by the user. This can be 5, 10,
#         15 or 20 milimeter.

#     Returns
#     -------
#         calib_dist : int
#             Integer variable containing the distance between the two
#             specified point in pixel units.

#     Notes
#     -----
#     - The function displays the image and waits for the user to click on two points to specify the distance.
#     - The spacing parameter should be provided as a string, representing the numeric value of the known distance.
#     - After calibration, the function will return the calibration distance in pixel units.

#     Examples
#     --------
#     >>> calibrateDistanceManually(img=([[[[0.22414216 0.19730392 0.22414216] ... [0.2509804  0.2509804  0.2509804 ]]]), 5)
#     99, 5 mm corresponds to 99 pixels
#     """
#     global mlocs
#     mlocs = []

#     # keep an unmodified copy of the original for zooming calculations
#     orig_img = np.uint8(nonflipped_img)
#     zoom_factor = 1.0

#     def refresh():
#         """Redraw the image window according to the current zoom and points."""
#         nonlocal zoom_factor
#         h, w = orig_img.shape[:2]
#         disp_w = max(1, int(w * zoom_factor))
#         disp_h = max(1, int(h * zoom_factor))
#         display_img = cv2.resize(
#             orig_img, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
#         )
#         for px, py in mlocs:
#             cv2.circle(
#                 display_img,
#                 (int(px * zoom_factor), int(py * zoom_factor)),
#                 3,
#                 (255, 255, 255),
#                 2,
#             )
#         cv2.imshow("image", display_img)

#     def mclick(event, x_val, y_val, flags, param):
#         nonlocal zoom_factor
#         global mlocs

#         # mouse wheel event: adjust zoom
#         if event == cv2.EVENT_MOUSEWHEEL or event == cv2.EVENT_MOUSEHWHEEL:
#             if flags > 0:
#                 zoom_factor *= 1.1
#             else:
#                 zoom_factor /= 1.1
#             zoom_factor = max(0.2, min(5.0, zoom_factor))
#             refresh()
#             return

#         # left click: add a point (only if fewer than 2 points)
#         if event == cv2.EVENT_LBUTTONDOWN:
#             h, w = orig_img.shape[:2]
#             # map to original coordinates
#             ox = int(x_val / zoom_factor)
#             oy = int(y_val / zoom_factor)
#             if not (0 <= ox < w and 0 <= oy < h):
#                 return
#             if len(mlocs) < 2:
#                 mlocs.append((ox, oy))
#             else:
#                 return

#         # right click: remove the last point if present
#         elif event == cv2.EVENT_RBUTTONDOWN and mlocs:
#             mlocs.pop()

#         refresh()

#     # give information for scaling
#     tk.messagebox.showinfo(
#         "Information",
#         "Scale the image before creating a mask."
#         + "\nClick on two scaling bars that are EXACTLY 1 CM APART."
#         + "\nUse the mouse wheel to zoom in/out if necessary."
#         + "\nRight‑click on the last point to remove the last point if you make a mistake."
#         + "\nClick 'q' to continue.",
#     )
#     # Edit image
#     img2 = np.uint8(nonflipped_img)

#     # display the image and wait for a keypress
#     # cv2.imshow("image", img2)
#     import matplotlib.pyplot as plt

#     plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
#     plt.title("image")
#     plt.axis("off")
#     plt.show()
#     cv2.setMouseCallback("image", mclick)
#     key = cv2.waitKey(0)

#     # if the 'q' key is pressed, break from the loop
#     if key == ord("q"):
#         cv2.destroyAllWindows()

#     calib_dist = 0
#     if len(mlocs) == 2:
#         calib_dist = np.abs(
#             math.sqrt(
#                 (mlocs[1][0] - mlocs[0][0]) ** 2 + (mlocs[1][1] - mlocs[0][1]) ** 2
#             )
#         )

#     mlocs = []

#     if calib_dist == 0:
#         tk.messagebox.showerror(
#             "Error",
#             "Calibration failed. Please click on two points to specify the distance.",
#         )
#         return None

#     tk.messagebox.showinfo("Information", f"1 cm corresponds to {calib_dist} pixels")

#     return calib_dist


def calibrate_distance_manually(nonflipped_img: np.ndarray, spacing: str):
    """
    Manually calibrate pixel-to-centimeter scaling using an interactive GUI.

    This function opens a Tkinter-based calibration window that allows the user
    to select two points on an image corresponding to a known physical distance
    (typically 1 cm). The Euclidean distance between the selected points is
    computed in pixel units and returned as the calibration factor.

    The interface supports zooming, point selection, and undo functionality,
    enabling precise placement of calibration points directly within the GUI.

    Parameters
    ----------
    nonflipped_img : np.ndarray
        Input image as a NumPy array. The image must be preloaded and is expected
        to be in either grayscale or BGR format. The image is internally converted
        to RGB for display purposes.

    spacing : str
        String representing the known physical distance between the two selected
        points in millimeters. Typically values such as "5", "10", "15", or "20".
        Note that the current implementation assumes a 1 cm (10 mm) reference for
        interpretation of the returned pixel distance.

    Returns
    -------
    float or None
        The Euclidean distance between the two selected points in pixel units,
        corresponding to 1 cm. Returns None if calibration is aborted or invalid.

    Notes
    -----
    **User interaction:**
    - Left-click: Place up to two calibration points.
    - Right-click: Remove the last placed point.
    - Mouse wheel: Zoom in/out centered around the cursor.
    - Enter: Confirm selection and compute calibration.
    - Escape: Cancel calibration.

    **Behavior:**
    - The function operates within a Tkinter `Toplevel` window and blocks execution
      until the window is closed.
    - The returned value represents pixels per centimeter and can be used to convert
      pixel-based measurements to real-world units.

    **Important:**
    - This implementation replaces OpenCV GUI (`cv2.imshow`) functionality to ensure
      compatibility with macOS and integration with Tkinter/CustomTkinter-based GUIs.
    - The `spacing` parameter is currently not used in scaling and serves as metadata.
      If variable scaling is required (e.g., non-1 cm references), the returned value
      should be adjusted accordingly.

    Examples
    --------
    >>> import cv2
    >>> img = cv2.imread("ultrasound.png")
    >>> px_per_cm = calibrate_distance_manually(img, spacing="10")
    >>> print(px_per_cm)
    98.7

    >>> # Convert pixel measurement to centimeters
    >>> pixel_length = 250
    >>> length_cm = pixel_length / px_per_cm
    >>> print(length_cm)
    2.53

    See Also
    --------
    area_cm2_from_mask : Convert segmented pixel areas to cm² using calibration.
    fill_contour_mask : Utility for generating binary masks from contours.
    """
    if nonflipped_img is None:
        return None

    # Ensure uint8 RGB for display
    img = np.uint8(nonflipped_img)
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    orig_img = img_rgb
    orig_h, orig_w = orig_img.shape[:2]

    messagebox.showinfo(
        "Information",
        "Scale the image before creating a mask."
        "\nClick on two scaling bars that are EXACTLY 1 CM APART."
        "\nUse the mouse wheel to zoom in/out if necessary."
        "\nRight-click to remove the last point."
        "\nPress Enter to confirm, or Esc to cancel.",
    )

    root = tk._default_root
    if root is None:
        root = tk.Tk()
        root.withdraw()

    top = tk.Toplevel(root)
    top.title("Calibration")
    top.geometry("1000x800")
    top.grab_set()

    canvas = tk.Canvas(top, bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    points = []
    zoom_factor = 1.0
    photo_ref = {"img": None}
    result = {"calib_dist": None}
    canvas_img_id = {"id": None}

    def redraw():
        canvas.delete("point")

        disp_w = max(1, int(orig_w * zoom_factor))
        disp_h = max(1, int(orig_h * zoom_factor))

        pil_img = Image.fromarray(orig_img).resize(
            (disp_w, disp_h), Image.Resampling.BILINEAR
        )
        photo = ImageTk.PhotoImage(pil_img)
        photo_ref["img"] = photo

        if canvas_img_id["id"] is None:
            canvas_img_id["id"] = canvas.create_image(0, 0, anchor="nw", image=photo)
        else:
            canvas.itemconfig(canvas_img_id["id"], image=photo)

        canvas.config(scrollregion=(0, 0, disp_w, disp_h))

        for px, py in points:
            x = px * zoom_factor
            y = py * zoom_factor
            r = 4
            canvas.create_oval(
                x - r, y - r, x + r, y + r, outline="white", width=2, tags="point"
            )

    def canvas_to_image_coords(x, y):
        cx = canvas.canvasx(x)
        cy = canvas.canvasy(y)
        ox = int(cx / zoom_factor)
        oy = int(cy / zoom_factor)
        return ox, oy

    def on_left_click(event):
        if len(points) >= 2:
            return
        ox, oy = canvas_to_image_coords(event.x, event.y)
        if 0 <= ox < orig_w and 0 <= oy < orig_h:
            points.append((ox, oy))
            redraw()

    def on_right_click(event):
        if points:
            points.pop()
            redraw()

    def on_mousewheel(event):
        nonlocal zoom_factor

        old_zoom = zoom_factor

        # Windows / macOS
        if event.delta > 0:
            zoom_factor *= 1.1
        else:
            zoom_factor /= 1.1

        zoom_factor = max(0.2, min(8.0, zoom_factor))

        # zoom around cursor
        mouse_x = canvas.canvasx(event.x)
        mouse_y = canvas.canvasy(event.y)

        redraw()

        if old_zoom != 0:
            scale = zoom_factor / old_zoom
            new_x = mouse_x * scale
            new_y = mouse_y * scale

            canvas_w = canvas.winfo_width()
            canvas_h = canvas.winfo_height()

            xview = max(0, new_x - event.x)
            yview = max(0, new_y - event.y)

            total_w = max(1, int(orig_w * zoom_factor))
            total_h = max(1, int(orig_h * zoom_factor))

            canvas.xview_moveto(xview / total_w)
            canvas.yview_moveto(yview / total_h)

    def on_confirm(event=None):
        if len(points) != 2:
            messagebox.showerror(
                "Error",
                "Calibration failed. Please click on two points to specify the distance.",
            )
            return

        dist = math.sqrt(
            (points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2
        )

        result["calib_dist"] = dist
        top.destroy()

    def on_cancel(event=None):
        result["calib_dist"] = None
        top.destroy()

    # Optional scrollbars
    xbar = tk.Scrollbar(top, orient="horizontal", command=canvas.xview)
    ybar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
    canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)

    xbar.pack(side="bottom", fill="x")
    ybar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    canvas.bind("<Button-1>", on_left_click)
    canvas.bind("<Button-3>", on_right_click)
    canvas.bind("<MouseWheel>", on_mousewheel)  # macOS / Windows
    top.bind("<Return>", on_confirm)
    top.bind("<Escape>", on_cancel)

    redraw()
    top.wait_window()

    calib_dist = result["calib_dist"]

    if calib_dist is None or calib_dist == 0:
        return None

    messagebox.showinfo("Information", f"1 cm corresponds to {calib_dist:.1f} pixels")
    return calib_dist
