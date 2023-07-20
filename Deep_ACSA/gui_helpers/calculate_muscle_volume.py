"""
Description
-----------
This module contains a function to automatically calculate the volume
of the analyzed muscle. Note that this can only be done across several
images of the same muscle, at least 3. The truncated cone formula is used,
modelling the muscle as a cone. The higher the number of images per muscle,
the higher the accuracy of the volume calculation, as distances between images
are extrapolated. Thus, the distance must be known and must be equal.

Functions scope
---------------
muscle_volume_calculation
    Function to calcualte the echo intensity of muscle area.
"""

import math

def muscle_volume_calculation(scan_area: list, dist: float):
    """
    Fuction to calculate muscle volume on the basis of the predicted
    muscle areas. Several images of the same muscle are needed to
    calculate the muscle muscle volume with known distances between
    the images.

    The truncated cone formula is used to calculate the volume,
    where the muscle is modelled as a cone.

    Parameters
    ----------
    scan_area : list
        List variable containing the predicted muscle areas.
        Should be at least 3 predicted areas.
    dist : float
        Float variable contianing the distance between the images
        in cm. Needs to be constant.

    Returns
    -------
    total_volume : float
        Float variable containing the total predicted muscle
        volume in cm^3.

    Notes
    -----
    - The `scan_area` list should contain at least three elements representing
      the predicted muscle areas from different scans or images of the same
      muscle.
    - The `dist` parameter should be the known distance between the images in
      centimeters (cm). It must be constant for accurate volume calculation.
    - The returned volume is in cubic centimeters (cm^3).

    Example
    -------
    >>>muscle_volume = muscle_volume_calculation([2, 3, 5, 4, 2, 2], 3.75)
    11.3276276145058
    """
    total_volume = 0

    for scan_nr in range(len(scan_area)-1):

        partial_volume = 0

        # a and b are ACSA of 2 scans
        a = scan_area[scan_nr]
        b = scan_area[scan_nr + 1]

        partial_volume = 1/3 * (a + math.sqrt((a * b) + b)) * dist
        total_volume = total_volume + partial_volume

    return total_volume
