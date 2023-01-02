import setuptools

setuptools.setup(
    name="DeepACSA",
    version="0.2.0",
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: MSK Ultrasobography :: Deep Learning'
    ],
    install_requires=[
                      "jupyter==1.0.0",
                      "Keras==2.4.3",
                      "matplotlib==3.3.4",
                      "numpy==1.19.5",
                      "opencv-contrib-python==4.5.1.48",
                      "pandas==1.1.2",
                      "Pillow==8.1.0",
                      "scikit-image==0.18.1",
                      "scikit-learn==0.24.1",
                      "tensorflow==2.9.3",
                      "tqdm==4.56.2",
                      "openpyxl==3.0.6",
                      "h5py==2.10.0",
    ],
    python_requires="==3.8.13"
)
