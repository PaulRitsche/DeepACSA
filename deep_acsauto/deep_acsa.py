"""Anatomical cross-sectional area evalutaion in Ultrasound images."""

import sys
import argparse

from predict_muscle_area import calculate_batch, calculate_batch_efov


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
    )

    parser.add_argument(
        '-rp', '--rootpath',
        type=str,
        required=True,
        default=None,
        help="path to root directory of images",
    )
    parser.add_argument(
        '-mp', '--modelpath',
        type=str,
        required=True,
        default=None,
        help="file path to .h5 file containing model used for prediction"
    )
    parser.add_argument(
        '-ft', "--filetype",
        type=str,
        required=True,
        default="/**/*.tif",
        help="specify image type as: /**/*.tif, /**/*.tiff, /**/*.png, /**/*.bmp, /**/*.jpeg, /**/*.jpg"
    )
    parser.add_argument(
        '-d', "--depth",
        type=float,
        required=True,
        default=4,
        help="ultrasound scanning depth (cm)"
    )
    parser.add_argument(
        '-sp', "--spacing",
        type=float,
        default=5,
        help="distance (mm) between detetec vertical scaling lines"
    )
    parser.add_argument(
        '-m', "--muscle",
        type=str,
        required=True,
        default="RF",
        help="muscle that is analyzed",
    )
    parser.add_argument(
        '-s', "--scaling",
        type=str,
        required=True,
        default="EFOV",
        help="scaling type present in ultrasound image"
    )

    args = parser.parse_args()
    print(args)
    results = []

    if args.scaling == "EFOV":
        calculate_batch_efov(
            args.rootpath,
            args.filetype,
            args.modelpath,
            args.depth,
            args.muscle
            )
    else:
        calculate_batch(
            args.rootpath,
            args.filetype,
            args.modelpath,
            args.spacing,
            args.muscle,
            args.scaling
            )
