import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, median_filter
from skimage.feature import canny, hessian_matrix, hessian_matrix_eigvals
from skimage.filters import threshold_otsu


def preprocess_image(
    image,
    muscle_type,
    flip_horizontal,
    flip_vertical,
    tubeness_sigma,
    gaussian_sigma,
    min_length_fac=0.1,
):
    """
    Preprocess the image based on the selected muscle type and options.
    :param image: The input image to preprocess.
    :param muscle_type: The type of muscle to process (e.g., "Rectus femoris").
    :param flip_horizontal: Whether to flip the image horizontally.
    :param flip_vertical: Whether to flip the image vertically.
    :param tubeness_sigma: Sigma value for tubeness filtering.
    :param gaussian_sigma: Sigma value for Gaussian blur.
    :param min_length_fac: Minimum length factor to remove small objects.
    :return: The preprocessed image.
    """

    start = time.time()

    # Flip the image if necessary
    if flip_horizontal:
        image = cv2.flip(image, 1)
    if flip_vertical:
        image = cv2.flip(image, 0)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Apply Non-Local Means Denoising
    denoised = cv2.fastNlMeansDenoising(
        equalized, None, h=10, templateWindowSize=7, searchWindowSize=21
    )

    # Apply Bilateral Filtering for edge-preserving smoothing
    bilateral = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply Gaussian blur using scipy (small sigma)
    if gaussian_sigma > 0:
        blurred = gaussian_filter(
            bilateral, sigma=gaussian_sigma
        )  # Adjust sigma as needed

    # Tubeness filtering using OpenCV (using GaussianBlur for demonstration)
    if tubeness_sigma > 0:
        tubeness = cv2.GaussianBlur(
            blurred, (0, 0), tubeness_sigma
        )  # Adjust sigma as needed

    # Apply Canny edge detection using skimage with adjusted thresholds
    edges = canny(tubeness, sigma=1)  # Adjust sigma as needed for edge detection

    # Convert edges from boolean to uint8 for OpenCV compatibility
    edges_uint8 = (edges * 255).astype(np.uint8)

    # Detect contours in the edge-detected image
    contours, _ = cv2.findContours(
        edges_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter out small objects based on min_length_fac
    image_width = image.shape[1]
    min_length_threshold = min_length_fac * image_width
    filtered_contours = [
        cnt for cnt in contours if cv2.arcLength(cnt, True) > min_length_threshold
    ]

    print(f"Number of contours after filtering: {len(filtered_contours)}")

    # Create a mask for the filtered contours and apply it
    mask = np.zeros_like(edges_uint8)
    cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    print(f"Time taken for preprocessing: {time.time() - start} seconds")

    return mask


import cv2
import numpy as np

def remove_border_scale_bar(gray_u8: np.ndarray, *, thr=240, min_area_frac=0.0005) -> np.ndarray:
    """
    Remove bright scale-bar/overlays near borders.
    Works on uint8 grayscale.
    """
    if gray_u8.ndim != 2:
        raise ValueError("remove_border_scale_bar expects 2D grayscale uint8")
    if gray_u8.dtype != np.uint8:
        gray_u8 = gray_u8.astype(np.uint8)

    h, w = gray_u8.shape[:2]
    min_area = int(min_area_frac * h * w)

    # detect very bright pixels
    bin_img = (gray_u8 >= thr).astype(np.uint8) * 255

    # connect fragments
    k = max(3, (min(h, w) // 300) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # contours (OpenCV 3/4 compatible)
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]

    # build mask of border-touching bright rectangles
    remove = np.zeros((h, w), dtype=np.uint8)

    def touches_border(x, y, ww, hh, pad=2):
        return (x <= pad) or (y <= pad) or (x + ww >= w - pad) or (y + hh >= h - pad)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        if touches_border(x, y, ww, hh):
            cv2.rectangle(remove, (x, y), (x + ww, y + hh), 255, thickness=-1)

    # apply: set removed pixels to 0 (or median background if you prefer)
    out = gray_u8.copy()
    out[remove > 0] = 0
    return out

def detect_candidate_contours(bin_img: np.ndarray, *, min_area_px=300) -> list[np.ndarray]:
    """
    Return external contours sorted by descending area.
    bin_img: 0/255 uint8.
    """
    if bin_img.ndim != 2:
        raise ValueError("detect_candidate_contours expects 2D binary")
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    contours = [c for c in contours if cv2.contourArea(c) >= float(min_area_px)]
    contours.sort(key=cv2.contourArea, reverse=True)
    return contours


def _find_contours_compat(bin_img: np.ndarray):
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hier = res
    elif len(res) == 3:
        _, contours, hier = res
    else:
        raise RuntimeError(f"Unexpected findContours return length: {len(res)}")
    return contours, hier

import cv2
import numpy as np


def contours_external(bin_img: np.ndarray):
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hier = res
    else:
        _, contours, hier = res
    return contours, hier


def largest_contour(bin_img: np.ndarray, *, min_area_px: int = 50):
    if bin_img.ndim != 2:
        raise ValueError("largest_contour expects a 2D binary image")
    if bin_img.dtype != np.uint8:
        bin_img = bin_img.astype(np.uint8)

    contours, _ = contours_external(bin_img)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < float(min_area_px):
        return None
    return cnt


def fill_contour_mask(shape_hw: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [contour.astype(np.int32)], 255)
    return m


def area_cm2_from_mask(mask_u8: np.ndarray, *, scan_depth_cm: float, px_per_cm: float | None = None) -> float:
    """
    Convert a filled mask (255 inside) to cm².

    If px_per_cm is provided (manual scaling), use it.
    Else assume scan_depth_cm corresponds to full image height.
    """
    if mask_u8.ndim != 2:
        raise ValueError("area_cm2_from_mask expects 2D mask")
    area_px = int(np.count_nonzero(mask_u8))
    if area_px <= 0:
        return 0.0

    h = int(mask_u8.shape[0])

    if px_per_cm is not None and px_per_cm > 0:
        cm_per_px = 1.0 / float(px_per_cm)
    else:
        # None
        cm_per_px = 1

    return float(area_px) * (cm_per_px ** 2)


def detect_contour_from_preprocessed_mask(pre_mask_u8: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Input: your preprocess_image() output (a filled-ish mask).
    Output: (largest_contour or None, cleaned_binary_mask)
    """
    if pre_mask_u8.ndim == 3:
        pre_mask_u8 = cv2.cvtColor(pre_mask_u8, cv2.COLOR_BGR2GRAY)
    if pre_mask_u8.dtype != np.uint8:
        pre_mask_u8 = pre_mask_u8.astype(np.uint8)

    # Ensure binary
    _, bin_img = cv2.threshold(pre_mask_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Cleanup (small gaps)
    k = max(3, (min(bin_img.shape[:2]) // 200) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnt = largest_contour(bin_img, min_area_px=200)
    return cnt, bin_img

def measure_area(image, starting_points=None, scan_depth=5, num_rays=360, *, return_mask=False):
    """
    Robust area measurement:
    - Converts input to grayscale uint8
    - Creates a binary region mask via Otsu threshold + morphology
    - Takes the largest external contour and fills it
    - Computes area in pixels and converts using scan_depth**2 (keeps your original scaling behavior)

    Parameters
    ----------
    image : np.ndarray
        Preprocessed image (can be grayscale or BGR).
    starting_points : ignored (kept for API compatibility)
        Present only to maintain core interface.
    scan_depth : float|int
        Used for conversion factor (and clamped to int pixels).
    return_mask : bool
        If True, returns (cm_area, filled_mask, contour_points)

    Returns
    -------
    cm_area : float
        Area in "cm^2" using your original conversion scheme.
    """
    scan_depth_px = int(round(float(scan_depth)))
    if scan_depth_px <= 0:
        return (0.0, None, None) if return_mask else 0.0

    # ---- ensure grayscale uint8 ----
    if image is None:
        return (0.0, None, None) if return_mask else 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if gray.dtype != np.uint8:
        gmin, gmax = float(np.min(gray)), float(np.max(gray))
        if gmax > gmin:
            gray = ((gray - gmin) * (255.0 / (gmax - gmin))).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)

    # ---- binarize (Otsu) ----
    # Otsu chooses foreground vs background automatically.
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If Otsu picked the wrong polarity (common), flip so "muscle region" is white
    # Heuristic: prefer fewer white pixels than black pixels? actually muscle often occupies mid/large.
    white = int(np.count_nonzero(bin_img))
    total = int(bin_img.size)
    if white < total * 0.05 or white > total * 0.95:
        bin_img = cv2.bitwise_not(bin_img)

    # ---- cleanup ----
    k = max(3, (min(gray.shape[:2]) // 200) * 2 + 1)  # odd kernel, scale with image size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ---- contours ----
    contours, _ = _find_contours_compat(bin_img)
    if not contours:
        return (0.0, bin_img if return_mask else None, None) if return_mask else 0.0

    # pick largest contour by area
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 1:
        return (0.0, bin_img if return_mask else None, None) if return_mask else 0.0

    # ---- fill contour ----
    filled = np.zeros_like(gray, dtype=np.uint8)
    cv2.fillPoly(filled, [contour.astype(np.int32)], 255)

    area_px = int(np.count_nonzero(filled))
    cm_area = area_px / float(scan_depth_px ** 2)

    if return_mask:
        return float(cm_area), filled, contour
    return float(cm_area)

def _ensure_binary_u8(img: np.ndarray) -> np.ndarray:
    """
    Ensure a single-channel uint8 binary-ish image suitable for findContours.
    - If 3-channel: convert to gray
    - If float: normalize to uint8
    - If non-binary: Otsu threshold to get contours
    """
    if img is None:
        raise ValueError("find_starting_points: image is None")

    if img.ndim == 3:
        # treat as BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        # robust conversion to uint8
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_max > img_min:
            img = ((img - img_min) * (255.0 / (img_max - img_min))).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)

    # ensure we have a binary image for contours
    # (works even if already binary)
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img


def _find_contours_compat(bin_img: np.ndarray):
    """
    OpenCV 3/4 compatibility: findContours may return (img, contours, hierarchy) or (contours, hierarchy)
    """
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hierarchy = res
    elif len(res) == 3:
        _, contours, hierarchy = res
    else:
        raise RuntimeError(f"Unexpected findContours return length: {len(res)}")
    return contours, hierarchy


def find_starting_points(image: np.ndarray, method: str = "Automatic"):
    """
    Find starting points for outline detection.

    Returns
    -------
    starting_points : list[tuple[int,int]]
    overlay : np.ndarray
        RGB overlay image with points drawn (safe for display).
        NOTE: analysis 'image' is NOT modified.
    """
    starting_points: list[tuple[int, int]] = []

    # For contour detection: use a single-channel binary uint8 image
    bin_img = _ensure_binary_u8(image)
    contours, _ = _find_contours_compat(bin_img)

    # Create an RGB overlay for drawing (never draw on analysis input)
    if image.ndim == 2:
        overlay = cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_GRAY2RGB)
    else:
        # assume BGR if 3ch
        overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if method == "Automatic":
        for contour in contours:
            M = cv2.moments(contour)
            if M.get("m00", 0) != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                starting_points.append((cX, cY))
                cv2.circle(overlay, (cX, cY), 5, (255, 0, 255), -1)

    elif method == "Fixed Pixels":
        # Example fallback: center point (adjust if you have a defined rule)
        h, w = bin_img.shape[:2]
        cX, cY = w // 2, h // 2
        starting_points.append((cX, cY))
        cv2.circle(overlay, (cX, cY), 5, (255, 0, 255), -1)

    elif method == "Manual":
        # NOTE: OpenCV GUI; if you want full CustomTkinter integration, don’t use this.
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                starting_points.append((x, y))
                cv2.circle(overlay_bgr, (x, y), 5, (255, 0, 255), -1)
                cv2.imshow("Select Starting Points", overlay_bgr)

        print("Click to select starting points. Press ESC when done.")
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imshow("Select Starting Points", overlay_bgr)
        cv2.setMouseCallback("Select Starting Points", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        raise ValueError(f"Unknown method: {method}")

    return starting_points, overlay


def draw_outline(image, contours):
    """
    Draw the detected outline on the image.
    :param image: The image on which to draw.
    :param contours: The contours detected in the image.
    :return: The image with the outline drawn.
    """
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    return image


def flip_image(image, axis):
    """
    Flip the image horizontally (axis=0) or vertically (axis=1).
    :param image: The image to flip.
    :param axis: The axis to flip on (0 for horizontal, 1 for vertical).
    :return: The flipped image.
    """
    return cv2.flip(image, axis)


def clear_display():
    """
    Placeholder function to mimic the clearing of ROI manager and other displays.
    """
    print("Clearing display...")


def remove_structures(image):
    """
    Allows the user to manually draw violet circles around structures to remove them from the image.
    The circles can be removed by pressing the "Delete" key.
    :param image: The input image from which structures should be removed.
    :return: The image with the specified structures removed.
    """

    # Create a copy of the image for drawing and a mask for removal
    output_image = image.copy()
    temp_image = image.copy()
    mask = np.ones_like(image, dtype=np.uint8) * 255

    circles = []

    def draw_circle(event, x, y, flags, param):
        nonlocal temp_image, circles

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start a new circle with radius 0
            circles.append((x, y, 0))

        elif event == cv2.EVENT_MOUSEMOVE:
            if len(circles) > 0 and flags == cv2.EVENT_FLAG_LBUTTON:
                # Update the radius of the last circle
                cx, cy, _ = circles[-1]
                radius = int(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))
                circles[-1] = (cx, cy, radius)

                # Redraw the image with the new circle
                temp_image = output_image.copy()
                cv2.circle(
                    temp_image, (cx, cy), radius, (255, 0, 255), 2
                )  # Violet circle
                cv2.imshow("Draw Circles", temp_image)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finalize the last circle by drawing it on the output image
            if len(circles) > 0:
                cx, cy, radius = circles[-1]
                cv2.circle(
                    output_image, (cx, cy), radius, (255, 0, 255), 2
                )  # Violet circle
                cv2.circle(
                    mask, (cx, cy), radius, (0, 0, 0), thickness=-1
                )  # Draw filled circle on mask
                temp_image = (
                    output_image.copy()
                )  # Update the temporary image for further drawing
                cv2.imshow("Draw Circles", output_image)

    def remove_last_circle():
        if circles:
            circles.pop()  # Remove the last circle
            redraw_circles()

    def redraw_circles():
        nonlocal temp_image, mask, output_image
        output_image = image.copy()
        mask = np.ones_like(image, dtype=np.uint8) * 255
        for cx, cy, radius in circles:
            cv2.circle(
                output_image, (cx, cy), radius, (255, 0, 255), 2
            )  # Violet circle
            cv2.circle(mask, (cx, cy), radius, (0, 0, 0), thickness=-1)  # Update mask
        temp_image = output_image.copy()
        cv2.imshow("Draw Circles", output_image)

    # Display the image and set up the mouse callback
    cv2.imshow("Draw Circles", output_image)
    cv2.setMouseCallback("Draw Circles", draw_circle)
    print(
        "Draw violet circles around structures to remove them. Press Delete to remove the last circle. Press ESC when done."
    )

    # Wait until ESC is pressed
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == 8:  # Delete key
            remove_last_circle()

    cv2.destroyAllWindows()

    # Apply the mask to remove the selected structures
    cleaned_image = cv2.bitwise_and(image, mask)

    return cleaned_image


def excel_expo(results, output_path):
    """
    Export the results to an Excel file.
    :param results: Dictionary of results to export.
    :param output_path: Path to save the Excel file.
    """
    df = pd.DataFrame(results)
    df.to_excel(output_path, index=False)
    print(f"Results exported to {output_path}")


def process_images(
    input_dir,
    output_dir,
    settings,
    muscle_type,
    flip_horizontal,
    flip_vertical,
    outline_strategy,
    scan_depth,
):
    """
    Process all images in the specified directory.
    :param input_dir: Directory of images to process.
    :param output_dir: Directory to save processed images.
    :param settings: Preprocessing settings.
    :param muscle_type: Muscle type to analyze.
    :param flip_horizontal: Whether to flip the image horizontally.
    :param flip_vertical: Whether to flip the image vertically.
    :param outline_strategy: Outline finder strategy.
    :param scan_depth: Scan depth for measurements.
    """
    file_list = [
        f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png", ".tif"))
    ]
    results = {"Filename": [], "Area (cm²)": []}

    for file_name in file_list:
        image_path = os.path.join(input_dir, file_name)
        image = cv2.imread(image_path)

        cv2.imshow("Processed Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        preprocessed_image = preprocess_image(
            image,
            muscle_type,
            flip_horizontal,
            flip_vertical,
            settings["tubeness_sigma"],
            settings["gaussian_sigma"],
        )
        # preprocessed_image = preprocess_image2(image=image)

        if preprocessed_image is None or preprocessed_image.size == 0:
            print(f"Preprocessing failed for image: {image_path}")
            continue

        # Print debug information
        print(f"Processing image: {file_name}")

        # Final check before saving
        if preprocessed_image is None or preprocessed_image.size == 0:
            print(
                f"Image processing resulted in an empty or invalid image: {file_name}"
            )
            continue
        cv2.imshow("mask", preprocessed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cleaned_image = remove_structures(preprocessed_image)

        starting_points, preprocessed_image = find_starting_points(
            cleaned_image, method=outline_strategy
        )

        # Display the processed image with contours
        cv2.imshow("Processed Image - Final", cleaned_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # draw_outline(image, starting_points)
        area = measure_area(preprocessed_image, starting_points)

        results["Filename"].append(file_name)
        results["Area (cm²)"].append(area)

        output_image_path = os.path.join(output_dir, file_name)
        try:
            cv2.imwrite(output_image_path, preprocessed_image)
            print(f"Saved processed image to: {output_image_path}")
        except Exception as e:
            print(f"Failed to save image {file_name}: {e}")

    return results
