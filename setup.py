import setuptools

with open("README_Pypi.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepACSA",
    version="0.3.0",
    author="Paul Ritsche",
    author_email="paul.ritsche@unibas.ch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PaulRitsche/DeepACSA",
    packages=setuptools.find_packages(),
    package_data={"": [".Deep_ACSA/gui_helpers/*.ico"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
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
    ],
)
