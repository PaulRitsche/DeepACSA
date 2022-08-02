![Title_Image](https://user-images.githubusercontent.com/71383228/148084248-5e169761-a075-4c86-836a-e9b2cec56ef6.jpg)

# DeepACSA

DeepACSA is an open-source tool to evaluate the anatomical cross-sectional area of muscles in ultrasound images using deep learning.
All following commands should be entered in your command prompt / terminal.
More information about the usage of DeepACSA can be found in the instructional video (https://youtu.be/It9CqVSNc9M). 
Our trained models can be accessed at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6953924.svg)](https://doi.org/10.5281/zenodo.6953924).
Anonymized panoramic ultrasound images are available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5799204.svg)](https://doi.org/10.5281/zenodo.5799204).
If you find this work useful, please remember to cite the corresponding paper (https://medrxiv.org/cgi/content/short/2021.12.27.21268258v1), where more information about the model architecture and performance can be found as well. 

## Installation 

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

## Descriptive figure of the model used

![Unet_Graphic]
DeepACSA workflow. a) Original ultrasound image of the m. rectus femoris (RF) at 50% of femur length with automatic scaling (green line). b) The original image is then preprocessed with contrast-limited adaptive histogram equalization and inputted to the model. c) Detailed U-net CNN architecture (modified from Ronneberger et al. (2015) and Cronin et al. (2020). Multi-channel feature maps are represented by the blue boxes with number of channels displayed on top of the respective box. Copied feature maps from the convolutional (left) side that are concatenated with the ones from the expanding (right) side are represented by the white boxes. The different operations are marked by the arrows. d) Model prediction of muscle area following post-processing (shown as a binary image). 


## Results of comparing DeepACSA analysis to manual analysis

![BA_plots]
Bland-Altman plots of all muscles plotting the difference between manual and DeepACSA with incorrect predictions removed (rm), manual and DeepACSA as well as manual and ACSAuto area segmentation measurements against the mean of both measures. Dotted and solid lines illustrate 95% limits of agreement and bias. M. rectus femoris (RF) and m. vastus lateralis (VL), mm. gastrocnemius medialis (GM), and lateralis (GL).

## Examples

This is an example command making use of the implemented GUI:
```sh
python deep_acsa_gui.py 
```

## Literature
Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. ArXiv150504597 Cs (2015).

Cronin, N. J., Finni, T. & Seynnes, O. Fully automated analysis of muscle architecture from B-mode ultrasound images with deep learning. ArXiv200904790 Cs Eess (2020).
