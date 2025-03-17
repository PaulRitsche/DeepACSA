"""
Description
-----------
This module contains functions to automatically or manually
scale images.
The scope of the automatic method is limited to scaling bars being
present in the right side of the image. The scope of the manual method
is not limited to specific scaling types in images. However, the distance
between two selected points in the image required for the scaling must be known.

Functions scope
---------------
region_of_interest
    Function to crop the images according to a specified
    region of interest.
mclick
    Function to detect mouse click coordinates in image.
draw_the_lines
    Function to mark the detected lines.
calibrate_distance_efov
    Function to calibrate EFOV ultrasonography images automatically. 
calibrate_distance_manually
    Function to manually calibrate an image to convert measurements
    in pixel units to centimeters.
calibrate_distance_static
    Function to calibrate an image to convert measurements
    in pixel units to centimeters.
"""

import math
import tkinter as tk

import cv2
import numpy as np

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

    Example
    -------
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
    if muscle == "VL":
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
    if muscle == "GL":
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
    if muscle == "BF":
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


def calibrate_distance_manually(nonflipped_img: np.ndarray, spacing: str):
    """
    Function to manually calibrate an image to convert measurements
    in pixel units to centimeters.

    The function calculates the distance in pixel units between two
    points on the input image. The points are determined by clicks of
    the user. The distance (in milimeters) is determined by the value
    contained in the spacing variable. Then the ratio of pixel / centimeter
    is calculated. To get the distance, the euclidean distance between the
    two points is calculated.

    Parameters
    ----------
    nonflipped_img : np.ndarray
        Input image to be analysed as a numpy array. The image must
        be loaded prior to calibration, specifying a path
        is not valid.
    spacing : int
        Integer variable containing the known distance in milimeter
        between the two placed points by the user. This can be 5, 10,
        15 or 20 milimeter.

    Returns
    -------
        calib_dist : int
            Integer variable containing the distance between the two
            specified point in pixel units.

    Notes
    -----
    - The function displays the image and waits for the user to click on two points to specify the distance.
    - The spacing parameter should be provided as a string, representing the numeric value of the known distance.
    - After calibration, the function will return the calibration distance in pixel units.

    Examples
    --------
    >>> calibrateDistanceManually(img=([[[[0.22414216 0.19730392 0.22414216] ... [0.2509804  0.2509804  0.2509804 ]]]), 5)
    99, 5 mm corresponds to 99 pixels
    """
    global mlocs
    mlocs = []

    def mclick(event, x_val, y_val, flags, param):
        global mlocs
        img_cop = img2.copy()
        # if the left mouse button was clicked, record the (x, y) coordinates
        if event == cv2.EVENT_LBUTTONDOWN:
            mlocs.append((x_val, y_val))
            # Draw a red dot on the image at the clicked position
            cv2.circle(img_cop, (x_val, y_val), 3, (255, 255, 255), 2)
            cv2.imshow("image", img_cop)

    # give information for scaling
    tk.messagebox.showinfo(
        "Information",
        "Scale the image before creating a mask."
        + "\nClick on two scaling bars that are EXACTLY 1 CM APART."
        + "\nClick 'q' to continue.",
    )
    # Edit image
    img2 = np.uint8(nonflipped_img)

    # display the image and wait for a keypress
    cv2.imshow("image", img2)
    cv2.setMouseCallback("image", mclick)
    key = cv2.waitKey(0)

    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()

    calib_dist = 0
    if len(mlocs) == 2:
        calib_dist = np.abs(
            math.sqrt(
                (mlocs[1][0] - mlocs[0][0]) ** 2 + (mlocs[1][1] - mlocs[0][1]) ** 2
            )
        )

    mlocs = []

    if calib_dist == 0:
        tk.messagebox.showerror(
            "Error",
            "Calibration failed. Please click on two points to specify the distance.",
        )
        return None

    tk.messagebox.showinfo("Information", f"1 cm corresponds to {calib_dist} pixels")

    return calib_dist
