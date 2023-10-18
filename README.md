![Title_Image](https://user-images.githubusercontent.com/71383228/148084248-5e169761-a075-4c86-836a-e9b2cec56ef6.jpg)

# DeepACSA
[![Documentation Status](https://readthedocs.org/projects/deepacsa/badge/?version=latest)](https://deepacsa.readthedocs.io/en/latest/?badge=latest)

*Automatic analysis of human lower limb ultrasonography images*

DeepACSA is an open-source tool to evaluate the anatomical cross-sectional area of muscles in ultrasound images using deep learning.
More information about the installtion and usage of DeepACSA can be found in the online [documentation](https://deepacsa.readthedocs.io/en/latest/index.html). You can find information about contributing, issues and bug reports there as well.
Our trained models, training data, an executable as well as example files can be accessed at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8419487.svg)](https://doi.org/10.5281/zenodo.8419487).
If you find this work useful, please remember to cite the corresponding [paper](https://journals.lww.com/acsm-msse/Abstract/9900/DeepACSA__Automatic_Segmentation_of.87.aspx), where more information about the model architecture and performance can be found as well. 

## Whats new?

With version 0.3.1, we included new models for the m. vastus lateralis (VL) and m. rectus femoris (RF) and added manual image labelling and mask inspection to the GUI. Take a look at our [documentation](https://deepacsa.readthedocs.io/en/latest/news.html) to see more details and the result of the model comparisons.

## Quickstart

To quickly start the DeepACSA either open the executable or type 

``python -m Deep_ACSA``

in your prompt once the package was installed locally with

``python -m pip install -e .``

when navigated at the DeepACSA/DeepACSA folder. 
Irrespective of the way the software was started, the GUI should open and is ready to be used.

![main_gui](https://github.com/PaulRitsche/DeepACSA/assets/71383228/b3a48daf-58ea-4971-badd-dcd6387000b7)


## Descriptive figure of the model used

![Figure2_VGG16Unet](https://user-images.githubusercontent.com/71383228/182554020-2c8bad75-7f08-4194-8f5f-0a521a70781c.jpg)

DeepACSA workflow. a) Original ultrasound image of the m. rectus femoris (RF) at 50% of femur length that serves as input for the model. b) Detailed U-net CNN architecture with a VGG16 encoder (left path). c) Model prediction of muscle area following post-processing (shown as a binary image). 

## Results of comparing DeepACSA analysis to manual analysis

![Figure3_BAP](https://user-images.githubusercontent.com/71383228/182554096-c5fde7cd-a137-4cc4-ad73-a819368d13ec.jpg)

Bland-Altman plots of all muscles plotting the difference between manual and DeepACSA with incorrect predictions removed (rm), manual and DeepACSA as well as manual and ACSAuto area segmentation measurements against the mean of both measures. Dotted and solid lines illustrate 95% limits of agreement and bias. M. rectus femoris (RF) and m. vastus lateralis (VL), mm. gastrocnemius medialis (GM), and lateralis (GL).
