# Copyright (C) 2023 Paul Ritsche

from pathlib import Path

try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import setup

INSTALL_REQUIRES = [
    "customtkinter==5.2.2",
    "CTKToolTip=0.8",
    "Keras==2.10.0",
    "matplotlib==3.6.1",
    "numpy==1.23.4",
    "opencv-contrib-python==4.6.0.66",
    "openpyxl==3.0.10",
    "pandas==1.5.1",
    "Pillow==9.2.0",
    "pre-commit==2.17.0",
    "scikit-image==0.19.3",
    "scikit-learn==1.1.2",
    "sewar==0.4.5",
    "shapely==2.0.5",
    "tensorflow==2.10.0",
    "tqdm==4.64.1",
]

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.10",
    "Operating System :: Microsoft :: Windows",
]


this_directory = Path(__file__).parent
long_descr = (this_directory / "README_Pypi.md").read_text()

if __name__ == "__main__":
    setup(
        name="DeepACSA",
        version="0.3.2",
        author="Paul Ritsche",
        author_email="paul.ritsche@unibas.ch",
        long_description=long_descr,
        long_description_content_type="text/markdown",
        url="https://github.com/PaulRitsche/DeepACSA",
        packages=find_packages(),
        package_data={"DeepACSA": ["*"]},
        classifiers=CLASSIFIERS,
        python_requires=">=3.10",
        install_requires=INSTALL_REQUIRES,
    )
