![Title_Image](https://user-images.githubusercontent.com/71383228/148084248-5e169761-a075-4c86-836a-e9b2cec56ef6.jpg)

# DeepACSA

DeepACSA is an open-source tool to evaluate the anatomical cross-sectional area of muscles in ultrasound images using deep learning.
All following commands should be entered in your command prompt / terminal.
More information about the usage of DeepACSA can be found in the instructional video (https://youtu.be/It9CqVSNc9M). 
Our trained models can be accessed at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6953924.svg)](https://doi.org/10.5281/zenodo.6953924).
Anonymized panoramic ultrasound images are available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5799204.svg)](https://doi.org/10.5281/zenodo.5799204).
If you find this work useful, please remember to cite the corresponding paper (https://journals.lww.com/acsm-msse/Abstract/9900/DeepACSA__Automatic_Segmentation_of.87.aspx), where more information about the model architecture and performance can be found as well. 

## Installation Windows

1. Git setup (Optional): 

This step is redundant if you download the repository as a .zip file and not use Git functionalities. Git is not required to run DeepACSA.
If you want to use Git, install it https://git-scm.com/download/win and set it up according to the instructions. 

2. Clone the Github repository:

Create and navigate to the folder where you want to save the project. For example:

```sh
mkdir DeepACSA
```
```sh
cd DeepACSA
```

When you have navigated to your preferred folder, clone the git repository.
```sh
git clone https://github.com/PaulRitsche/DeepACSA
```

3. Anaconda setup (only before first usage)

Install Python / Anaconda: https://www.anaconda.com/distribution/ (click ‘Download’ and be sure to choose ‘Python 3.X Version’ (where the X represents the latest version being offered. IMPORTANT: Make sure you tick the ‘Add Anaconda to my PATH environment variable’ box).
Open an Anaconda prompt window and create a virtual environment (This may take some time):

```sh
conda env create -f environment.yml 
```
Change the directory to where you want to saved the project folder, e.g. by typing:

```sh
cd c:/Users/Paul/Desktop/DeepACSA
```

Activate the virtual environment by typing:

```sh
conda activate DeepACSA
```

...And you are ready to go!

4. GPU-Setup (Optional): 

If you are using a GPU and want to train your own models, make sure your CUDA version complies with our tensorflow version (which is 2.4.0). 

## Usage

Open an Anaconda prompt window.
Activate your virtual environment and change to the directory containing the code, as done above. 

Type the here presented command into the prompt window and enter the required parameters in the GUI. 

```sh
python deep_acsa_gui.py

```

## Installation MacOS

Please be aware that the installation using the supplied environment.yml and requirements.txt are not functional for the new Apple M1/M2 chips. There are some support issues for older versions of tensorflow. At the moment, we do not provide any installation guidelines for the latest versions of MacOS with the M1/M2 chips. Sorry for any inconveniences, we aim to provide this in the near future.

## Descriptive figure of the model used

![Figure2_VGG16Unet](https://user-images.githubusercontent.com/71383228/182554020-2c8bad75-7f08-4194-8f5f-0a521a70781c.jpg)

DeepACSA workflow. aFigure 2: DeepACSA workflow. a) Original ultrasound image of the m. rectus femoris (RF) at 50% of femur length that serves as input for the model. b) Detailed U-net CNN architecture (modified from Ronneberger et al. (19) and Cronin et al. (15)) with a VGG16 encoder (left path). c) Model prediction of muscle area following post-processing (shown as a binary image). 


## Results of comparing DeepACSA analysis to manual analysis

![Figure3_BAP](https://user-images.githubusercontent.com/71383228/182554096-c5fde7cd-a137-4cc4-ad73-a819368d13ec.jpg)

Bland-Altman plots of all muscles plotting the difference between manual and DeepACSA with incorrect predictions removed (rm), manual and DeepACSA as well as manual and ACSAuto area segmentation measurements against the mean of both measures. Dotted and solid lines illustrate 95% limits of agreement and bias. M. rectus femoris (RF) and m. vastus lateralis (VL), mm. gastrocnemius medialis (GM), and lateralis (GL).

## Examples

This is an example command making use of the implemented GUI:
```sh
python deep_acsa_gui.py 
```

## Literature
Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. ArXiv150504597 Cs (2015).

Cronin, N. J., Finni, T. & Seynnes, O. Fully automated analysis of muscle architecture from B-mode ultrasound images with deep learning. ArXiv200904790 Cs Eess (2020).
