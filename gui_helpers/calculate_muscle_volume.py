"""Python module which provides functions to caluculate muscle volume."""

import math

def muscle_volume_calculation(scan_area: list, dist: float):

	"""Calculates muscle volume on the basis of the predicted muscle areas.

	Arguments:
		List with the predicted muscle areas. Should be at least 3 predicted areas.
		Distance between the slices in cm. Needs to be constant.

	Returns:
		Predicted muscle volume (cm^3).

	Example:
		>>>scan area = [2, 3, 5, 4, 2, 2]
		muscle_volume = muscle_volume_calculation(scan_area, dist)
	Result:
		>>>muscle_volume = 11.3276276145058
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
