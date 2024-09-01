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


def measure_area(image, starting_points, scan_depth=5, num_rays=360):
    """
    Measure the area of the muscle in the preprocessed image by shooting rays from starting points.
    :param image: The preprocessed binary image (edges detected).
    :param starting_points: Initial points for outline detection.
    :param scan_depth: The maximum distance a ray can travel.
    :param num_rays: Number of rays to shoot outwards from each starting point.
    :return: The measured area in cm².
    """

    # Initialize a list to store detected contour points
    contour_points = []

    # Loop through each starting point
    for start_point in starting_points:
        x0, y0 = start_point

        # Shoot rays in multiple directions
        for angle in np.linspace(0, 2 * np.pi, num_rays):
            dx = np.cos(angle)
            dy = np.sin(angle)

            for d in range(scan_depth):
                x = int(x0 + d * dx)
                y = int(y0 + d * dy)

                # Check if the ray hits an edge
                if (
                    x < 0
                    or y < 0
                    or x >= image.shape[1]
                    or y >= image.shape[0]
                    or image[y, x] > 0
                ):
                    contour_points.append((x, y))
                    break

    # Convert the list of points to a numpy array
    contour_points = np.array(contour_points, dtype=np.int32)

    if len(contour_points) == 0:
        print("No contour points detected.")
        return 0.0

    # Create a blank mask and draw the contour on it
    mask = np.zeros_like(image)
    if len(contour_points) > 2:  # Need at least 3 points to form a contour
        cv2.polylines(mask, [contour_points], isClosed=True, color=255, thickness=1)
        cv2.fillPoly(mask, [contour_points], color=255)

        # Measure the area in pixels
        area_in_pixels = np.sum(mask > 0)

        # Convert pixel area to real area based on scan depth (assuming 1 pixel = 1 unit length)
        cm_area = area_in_pixels / (scan_depth**2)
    else:
        cm_area = 0.0

    # Plot the contour on the original image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.polylines(
        output_image, [contour_points], isClosed=True, color=(0, 255, 0), thickness=2
    )

    plt.figure(figsize=(10, 10))
    plt.title("Detected Muscle Contour")
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    print(f"Measured area: {cm_area:.2f} cm²")
    return cm_area


def find_starting_points(image, method="Automatic"):
    """
    Find starting points for outline detection and draw them on the image.
    :param image: The preprocessed image.
    :param method: The method for finding starting points ("Manual", "Automatic", "Fixed Pixels").
    :return: A list of starting points and the image with the points drawn.
    """
    starting_points = []

    if method == "Automatic":
        # Automatically detect points using some heuristic (e.g., contour centroids)
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                starting_points.append((cX, cY))
                # Draw the starting point on the image
                cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)

    elif method == "Manual":
        # Manually select starting points using mouse clicks
        def click_event(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDOWN:
                starting_points.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Select Starting Points", image)

        print("Click to select starting points. Press ESC when done.")
        cv2.imshow("Select Starting Points", image)
        cv2.setMouseCallback("Select Starting Points", click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif method == "Fixed Pixels":
        # Return fixed points
        starting_points = [(50, 50), (150, 150), (200, 200)]
        for point in starting_points:
            cv2.circle(image, point, 5, (0, 255, 0), -1)

    return starting_points, image


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
