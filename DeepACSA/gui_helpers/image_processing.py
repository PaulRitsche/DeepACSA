"""
Description
-----------
Ultrasound image preprocessing and area measurement pipeline.

This module provides functions for preprocessing muscle ultrasound images,
detecting anatomical contours, and estimating cross-sectional area.
The processing workflow includes contrast enhancement, denoising,
edge detection, contour filtering, and mask-based area computation.

Both automatic and interactive components are supported. Users may
manually remove unwanted structures, select starting points for
outline detection, and process entire directories of images.
Measurement results can be exported to Excel files.

The functions in this module form the core image analysis pipeline
used for muscle cross-sectional area estimation.
"""

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
    Preprocess an ultrasound image and return a binary region mask.

    The preprocessing pipeline includes optional image flipping,
    contrast enhancement (CLAHE), denoising, smoothing, edge detection,
    and contour filtering. Small contours are removed based on a
    minimum length threshold relative to image width. The output is a
    binary mask highlighting the retained regions.

    Parameters
    ----------
    image : np.ndarray
        Input ultrasound image in BGR or grayscale format.
    muscle_type : str
        Name of the muscle being analyzed. Currently not used to alter
        preprocessing logic but retained for API compatibility.
    flip_horizontal : bool
        If True, flip the image horizontally before processing.
    flip_vertical : bool
        If True, flip the image vertically before processing.
    tubeness_sigma : float
        Standard deviation used for Gaussian-based tubeness smoothing.
        If 0 or less, this step is skipped.
    gaussian_sigma : float
        Standard deviation for Gaussian blur applied prior to tubeness
        filtering. If 0 or less, this step is skipped.
    min_length_fac : float, optional
        Minimum contour length threshold expressed as a fraction of
        image width. Contours with arc length below this threshold
        are removed. Default is 0.1.

    Returns
    -------
    np.ndarray
        Binary mask (uint8) where retained regions are 255 and
        background pixels are 0.

    Notes
    -----
    The function assumes the input image is a valid OpenCV-readable image.
    The returned mask can be used for subsequent contour detection or
    area measurement steps.
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
    Remove bright scale bars or overlays touching image borders.

    The function detects very bright connected components in a grayscale
    image, identifies those that touch the image borders, and removes them
    by setting the corresponding pixels to zero. This is primarily intended
    to eliminate scale bars or overlay artifacts commonly present near
    image edges.

    Parameters
    ----------
    gray_u8 : np.ndarray
        Input 2D grayscale image of dtype ``uint8``.
    thr : int, optional
        Intensity threshold used to classify pixels as "bright".
        Pixels with values greater than or equal to this threshold
        are considered candidate overlay regions. Default is 240.
    min_area_frac : float, optional
        Minimum area of detected bright regions expressed as a fraction
        of total image area. Regions smaller than this threshold are
        ignored. Default is 0.0005.

    Returns
    -------
    np.ndarray
        Grayscale image with detected border-touching bright regions
        removed (set to 0).

    Raises
    ------
    ValueError
        If the input image is not a 2D array.

    Notes
    -----
    The function performs morphological closing to connect fragmented
    bright regions before contour detection. Only contours exceeding
    the specified minimum area and touching the image border are removed.
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
    Detect and rank external contours by area.

    Parameters
    ----------
    bin_img : np.ndarray
        2D binary image (uint8) where foreground pixels are 255
        and background pixels are 0.
    min_area_px : int, optional
        Minimum contour area in pixels. Contours with an area
        smaller than this threshold are discarded. Default is 300.

    Returns
    -------
    list of np.ndarray
        List of external contours sorted in descending order
        by contour area.

    Raises
    ------
    ValueError
        If the input image is not 2D.

    Notes
    -----
    Contours are extracted using ``cv2.findContours`` with
    ``RETR_EXTERNAL`` and ``CHAIN_APPROX_SIMPLE``.
    """
    if bin_img.ndim != 2:
        raise ValueError("detect_candidate_contours expects 2D binary")
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = res[0] if len(res) == 2 else res[1]
    contours = [c for c in contours if cv2.contourArea(c) >= float(min_area_px)]
    contours.sort(key=cv2.contourArea, reverse=True)
    return contours


def _find_contours_compat(bin_img: np.ndarray):
    """
    OpenCV-compatible contour extraction.

    Parameters
    ----------
    bin_img : np.ndarray
        2D binary image (uint8).

    Returns
    -------
    contours : list of np.ndarray
        Detected external contours.
    hierarchy : np.ndarray
        Contour hierarchy information returned by OpenCV.

    Raises
    ------
    RuntimeError
        If ``cv2.findContours`` returns an unexpected number of values.

    Notes
    -----
    Handles differences between OpenCV 3 and OpenCV 4 return
    signatures of ``cv2.findContours``.
    """
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
    """
    Extract external contours from a binary image.

    Parameters
    ----------
    bin_img : np.ndarray
        2D binary image (uint8) from which contours should be
        extracted.

    Returns
    -------
    contours : list of np.ndarray
        List of detected external contours.
    hierarchy : np.ndarray
        Contour hierarchy information returned by OpenCV.

    Notes
    -----
    Uses ``cv2.findContours`` with ``RETR_EXTERNAL`` and
    ``CHAIN_APPROX_SIMPLE``. Handles differences in return
    format between OpenCV versions.
    """
    res = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hier = res
    else:
        _, contours, hier = res
    return contours, hier


def largest_contour(bin_img: np.ndarray, *, min_area_px: int = 50):
    """
    Return the largest external contour above a minimum area threshold.

    Parameters
    ----------
    bin_img : np.ndarray
        2D binary image (uint8) from which contours are extracted.
    min_area_px : int, optional
        Minimum contour area in pixels. Contours smaller than this
        threshold are ignored. Default is 50.

    Returns
    -------
    np.ndarray or None
        The largest contour satisfying the area constraint,
        or ``None`` if no valid contour is found.

    Raises
    ------
    ValueError
        If the input image is not 2D.

    Notes
    -----
    The image is converted to ``uint8`` if necessary before
    contour extraction.
    """
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
    """
    Create a filled binary mask from a contour.

    Parameters
    ----------
    shape_hw : tuple of int
        Tuple ``(height, width)`` specifying the mask dimensions.
    contour : np.ndarray
        Contour points defining the region to fill.

    Returns
    -------
    np.ndarray
        Binary mask (uint8) with the contour region filled
        (255 inside, 0 outside).

    Notes
    -----
    The contour is cast to ``int32`` before filling to ensure
    OpenCV compatibility.
    """
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [contour.astype(np.int32)], 255)
    return m


def area_cm2_from_mask(mask_u8: np.ndarray, *, px_per_cm: float | None = None) -> float:
    """
    Compute area in square centimeters from a filled mask.

    Parameters
    ----------
    mask_u8 : np.ndarray
        2D binary mask (uint8) where foreground pixels are 255.
    px_per_cm : float or None, optional
        Pixel-to-centimeter scaling factor (pixels per centimeter).
        If provided and positive, area is converted using this value.
        If ``None`` or non-positive, pixel units are treated as
        centimeters (no scaling applied).

    Returns
    -------
    float
        Estimated area in square centimeters.

    Raises
    ------
    ValueError
        If the input mask is not 2D.

    Notes
    -----
    Area is computed as the number of non-zero pixels multiplied by
    ``(cm_per_px)^2``. If no scaling factor is provided, the function
    assumes one pixel equals one centimeter.
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
    Extract the largest contour from a preprocessed mask.
    
    The function converts the input mask to grayscale (if necessary),
    ensures a binary representation using Otsu thresholding, performs
    morphological cleanup, and returns the largest external contour.
    
    Parameters
    ----------
    pre_mask_u8 : np.ndarray
        Preprocessed mask image (typically the output of
        ``preprocess_image``). Can be grayscale or BGR.
    
    Returns
    -------
    tuple of (np.ndarray or None, np.ndarray)
        - Largest contour satisfying the area threshold, or ``None`` if
          no valid contour is found.
        - Cleaned binary mask used for contour detection.
    
    Notes
    -----
    Morphological closing is applied to reduce small gaps before
    contour extraction. The minimum area threshold for contour
    selection is fixed internally.
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
    Estimate cross-sectional area from a segmented image.

    The function converts the input image to grayscale, applies Otsu
    thresholding and morphological cleanup to obtain a binary mask,
    selects the largest external contour, fills it, and computes the
    area in pixel units. The result is scaled using ``scan_depth``.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or BGR) representing a segmented or
        preprocessed muscle region.
    starting_points : optional
        Present for API compatibility. Not used in the current
        implementation.
    scan_depth : float or int, optional
        Scaling factor used to convert pixel area. The value is
        rounded to an integer and used as ``area_px / scan_depth**2``.
    num_rays : int, optional
        Retained for API compatibility. Not used in the current
        implementation.
    return_mask : bool, optional
        If True, returns a tuple ``(area_cm2, filled_mask, contour)``.
        Otherwise, returns only the computed area.

    Returns
    -------
    float or tuple
        If ``return_mask`` is False:
            Estimated area value.
        If ``return_mask`` is True:
            Tuple containing:
            - Area value (float)
            - Filled binary mask (np.ndarray)
            - Selected contour (np.ndarray)

    Notes
    -----
    The scaling behavior follows the original implementation,
    where area is computed as::

        area_px / scan_depth**2

    This assumes a specific pixel-to-centimeter relationship and
    does not use physical calibration directly.
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
    Convert an image to a binary uint8 representation.

    Parameters
    ----------
    img : np.ndarray
        Input image (grayscale, BGR, or numeric array).

    Returns
    -------
    np.ndarray
        Single-channel binary image (uint8) obtained using
        Otsu thresholding.

    Raises
    ------
    ValueError
        If the input image is None.

    Notes
    -----
    The function converts multi-channel images to grayscale,
    normalizes non-uint8 arrays to uint8, and applies Otsu
    thresholding to produce a contour-ready binary image.
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
    Extract contours with OpenCV version compatibility.

    Parameters
    ----------
    bin_img : np.ndarray
        2D binary image (uint8).

    Returns
    -------
    contours : list of np.ndarray
        Detected external contours.
    hierarchy : np.ndarray
        Contour hierarchy information returned by OpenCV.

    Raises
    ------
    RuntimeError
        If ``cv2.findContours`` returns an unexpected number of values.

    Notes
    -----
    Handles differences between OpenCV 3 and OpenCV 4 return
    signatures of ``cv2.findContours``.
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

    Depending on the selected method, starting points are determined from
    the centroids of detected contours, a fixed pixel location, or manual
    mouse clicks in an OpenCV window. An RGB overlay image is returned with
    the selected points drawn for visualization.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or BGR). The image is used for contour
        detection and for generating an overlay for display.
    method : {'Automatic', 'Fixed Pixels', 'Manual'}, optional
        Strategy used to generate starting points:

        - ``'Automatic'``: compute contour centroids from a binary mask.
        - ``'Fixed Pixels'``: use the image center as a single point.
        - ``'Manual'``: collect points via mouse clicks in an OpenCV window.

    Returns
    -------
    starting_points : list of tuple[int, int]
        List of ``(x, y)`` starting point coordinates in pixel units.
    overlay : np.ndarray
        RGB overlay image with the selected points drawn. The input
        ``image`` is not modified.

    Raises
    ------
    ValueError
        If `method` is not one of the supported options.

    Notes
    -----
    When ``method='Manual'``, this function opens an OpenCV GUI window and
    blocks until a key is pressed. This may not work in headless
    environments.
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
    Draw contours on an image.

    Parameters
    ----------
    image : np.ndarray
        Image on which to draw. This array is modified in place.
    contours : list of np.ndarray
        Contours to draw (as returned by ``cv2.findContours``).

    Returns
    -------
    np.ndarray
        The same image array with contours drawn.

    Notes
    -----
    This function uses ``cv2.drawContours`` and therefore draws directly
    into the provided `image` array.
    """
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    return image


def flip_image(image, axis):
    """
    Flip an image along a specified axis.

    Parameters
    ----------
    image : np.ndarray
        Input image to flip.
    axis : int
        Flip code passed to ``cv2.flip``:

        - ``0``: vertical flip (top/bottom)
        - ``1``: horizontal flip (left/right)
        - ``-1``: both axes

    Returns
    -------
    np.ndarray
        Flipped image.
    """
    return cv2.flip(image, axis)


def clear_display():
    """
    Clear interactive displays (placeholder).

    This function currently acts as a placeholder and only prints a message.
    It is intended to mimic clearing UI elements such as ROI managers or
    display windows.
    """
    print("Clearing display...")


def remove_structures(image):
    """
    Interactively remove unwanted structures by masking user-drawn circles.

    An OpenCV window is opened where the user can draw circles over regions
    to remove. The selected regions are excluded from the image by applying
    a binary mask. The last circle can be removed with the Delete key, and
    the interaction ends when the user presses ESC.

    Parameters
    ----------
    image : np.ndarray
        Input image from which structures should be removed. This function
        does not modify `image` in place.

    Returns
    -------
    np.ndarray
        Image with the user-selected circular regions removed (masked out).

    Notes
    -----
    This function requires an interactive OpenCV GUI environment. It will
    block execution until the user exits the window (ESC).
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
    Export results to an Excel file.

    Parameters
    ----------
    results : dict or pandas.DataFrame
        Results to export. If a dict is provided, it must be compatible with
        ``pandas.DataFrame(results)`` (e.g., keys as column names and values
        as equal-length lists).
    output_path : str or os.PathLike
        File path where the Excel file will be written.

    Returns
    -------
    None

    Notes
    -----
    The file is written using ``pandas.DataFrame.to_excel`` with
    ``index=False``.
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
    Process a directory of images and compute cross-sectional area estimates.
    
    Images with extensions ``.jpg``, ``.png``, and ``.tif`` are loaded from
    `input_dir`, preprocessed into a mask, optionally cleaned via an
    interactive removal step, and analyzed to estimate area. Intermediate
    results are displayed using OpenCV windows. The resulting mask images are
    written to `output_dir`, and a summary dictionary of filenames and area
    values is returned.
    
    Parameters
    ----------
    input_dir : str or os.PathLike
        Directory containing input images.
    output_dir : str or os.PathLike
        Directory where processed output images will be saved.
    settings : dict
        Preprocessing settings. Must contain keys ``'tubeness_sigma'`` and
        ``'gaussian_sigma'``.
    muscle_type : str
        Muscle type label passed through to preprocessing.
    flip_horizontal : bool
        If True, flip images horizontally before preprocessing.
    flip_vertical : bool
        If True, flip images vertically before preprocessing.
    outline_strategy : {'Automatic', 'Fixed Pixels', 'Manual'}
        Strategy passed to ``find_starting_points``.
    scan_depth : float or int
        Scan depth passed to ``measure_area`` for scaling.
    
    Returns
    -------
    dict
        Dictionary with keys ``'Filename'`` and ``'Area (cm²)'`` containing
        per-image results.
    
    Notes
    -----
    This function uses multiple OpenCV GUI windows and may block execution
    awaiting user input. It also calls ``remove_structures``, which is
    interactive and requires a display environment.
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
