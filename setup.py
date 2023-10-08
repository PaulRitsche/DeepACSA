# Copyright (C) 2023 Paul Ritsche

from pathlib import Path

try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import setup

INSTALL_REQUIRES = [
    "Keras==2.9.0",
    "matplotlib==3.5.2",
    "numpy==1.21.2",
    "opencv-contrib-python==4.5.3.56",
    "pandas==1.3.3",
    "Pillow==8.3.2",
    "scikit-image==0.18.3",
    "scikit-learn==0.24.2",
    "tensorflow==2.9.0",
    "tqdm==4.62.2",
    "openpyxl==3.0.9",
    "h5py==3.4.0",
]

CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.9",
    "Operating System :: Microsoft :: Windows",
    "License :: OSI Approved :: Apache Software License",
]


this_directory = Path(__file__).parent
long_descr = (this_directory / "README_Pypi.md").read_text()

if __name__ == "__main__":
    setup(
        name="DeepACSA",
        version="0.3.1",
        author="Paul Ritsche",
        author_email="paul.ritsche@unibas.ch",
        long_description=long_descr,
        long_description_content_type="text/markdown",
        url="https://github.com/PaulRitsche/DeepACSA",
        packages=find_packages(),
        package_data={"DeepACSA": ["*"]},
        classifiers=CLASSIFIERS,
        python_requires=">=3.9",
        install_requires=INSTALL_REQUIRES,
    )
