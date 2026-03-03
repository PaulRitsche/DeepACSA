"""
Description
-----------
Utilities for estimating muscle volume from ultrasound data.

Muscle volume is computed from predicted anatomical cross-sectional
areas (ACSA) using the truncated cone formula between consecutive
scans with constant spacing. At least three images of the same muscle
are required, and the inter-scan distance must be known and provided
in centimeters.
"""

import math


def muscle_volume_calculation(scan_area: list, dist: float):
    """
    Estimate total muscle volume from predicted cross-sectional areas.

    The function calculates muscle volume by applying the truncated cone
    formula between consecutive anatomical cross-sectional areas (ACSA)
    measured from ultrasound images. The individual segment volumes are
    summed to obtain the total muscle volume.

    Parameters
    ----------
    scan_area : list of floats
        List variable containing the predicted muscle areas as floats.
    dist : float
        Float variable containing the distance between the images
        in cm.

    Returns
    -------
    total_volume : float
        Float variable containing the total predicted muscle
        volume in cubic centimeters (cm^3).

    Notes
    -----
    The truncated cone formula is given by:

    .. math::

        V = \\frac{h}{3} (A1 + A2 + \\sqrt{A1 A2})

    where `h` is the distance between scans. `A1` and `A2` are the predicted areas
    of two adjacent scans.

    The function makes a few assumptions without checking for them.

    - The `scan_area` list should contain at least three elements representing
      the predicted muscle areas from 3 different scans or images of the same
      muscle.
    - The `dist` parameter should be the known distance between the images in
      centimeters (cm). It must be constant and positive for accurate volume 
      calculation.

    Examples
    --------
    >>> muscle_volume_calculation([2, 3, 5, 4, 2, 2], 3.75)
    59.52879521092045
    """
    total_volume = 0

    for scan_nr in range(len(scan_area)-1):

        partial_volume = 0

        # a and b are ACSA of 2 scans
        a = scan_area[scan_nr]
        b = scan_area[scan_nr + 1]

        partial_volume = 1/3 * (a + b + math.sqrt((a * b))) * dist
        total_volume = total_volume + partial_volume

    return total_volume
