![Title_Image](https://user-images.githubusercontent.com/71383228/148084248-5e169761-a075-4c86-836a-e9b2cec56ef6.jpg)

# DeepACSA
[![Documentation Status](https://readthedocs.org/projects/deepacsa/badge/?version=latest)](https://deepacsa.readthedocs.io/en/latest/?badge=latest)

*Automatic analysis of human lower limb ultrasonography images*

DeepACSA is an open-source tool to evaluate the anatomical cross-sectional area of muscles in ultrasound images using deep learning.
More information about the installtion and usage of DeepACSA can be found in the online [documentation](https://deepacsa.readthedocs.io/en/latest/index.html). You can find information about contributing, issues and bug reports there as well.
Our trained models, training data, an executable as well as example files can be accessed at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19130694.svg)](https://doi.org/10.5281/zenodo.19130694).
If you find this work useful, please remember to cite the corresponding [paper](https://journals.lww.com/acsm-msse/Abstract/9900/DeepACSA__Automatic_Segmentation_of.87.aspx), where more information about the model architecture and performance can be found as well. 

## Quickstart

To quickly start the DeepACSA either open the executable or type 

``python -m Deep_ACSA``

in your prompt once the package was installed locally with

``pip install DeepACSA==0.3.1.``

when the DeepACSA environment is activated 

``conda create -n DeepACSA python=3.9``

``conda activate DeepACSA``

Irrespective of the way the software was started, the GUI should open and is ready to be used.

⚠️ v0.3.2 is not yet available as a python package. The newest release can be run by cloning the github repository

``git clone https://github.com/PaulRitsche/DeepACSA.git``

then creating the DeepACSA0.3.2 environment, i.e. with

``conda env create -f environment.yml``

in the root directory with the `environment.yml` file. 
Subsequently, install the package locally with 

``python -m pip instell -e .``

then run the module or the UI by 

``python -m DeepACSA`` 

or 

``cd DeepACSA``

``python deep_acsa_gui.py``

# Whats new?

## V0.3.2

With version 0.3.2, we included new models for the 

- patella tendon (taken from [Guzzi et al. 2026](https://link.springer.com/article/10.1007/s10278-026-01846-x))
- vastus medialis (taken from [Tayfur et al. 2025](https://www.sciencedirect.com/science/article/abs/pii/S0301562924004319))

Below you can find some overview tables. All models and the newest installer can be found here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19130694.svg)](https://doi.org/10.5281/zenodo.19130694).

## Patella Tendon Models (Unet3+ best performing)
We provide a model for the automatic segmentation of the patellar tendon anatomical cross-sectional area (ACSA) at 25%, 50%, and 75% of tendon length in healthy subjects (**UNet 3+**). 
We evaluated two model architectures for patellar tendon segmentation: *UNet-VGG16* and *UNet 3+*. Their performance was assessed by comparing automated predictions with manual segmentations. Overall, both models demonstrated good agreement with manual analysis, with **UNet 3+** showing the most consistent performance. Detailed methodology and results are reported in our publication [Guzzi et al. 2026](https://link.springer.com/article/10.1007/s10278-026-01846-x).

## Vatus medialis model (only VGG16Unet trained)
We provide a model for the automatic segmentation of the vastus medialis cross-sectional area (ACSA) in healthy participants as well participants with ACL injuries. 
A *UNet-VGG16* model was evaluated and compared to manual analysis. Comparability calculations and detailled methodology can be found at [Tayfur et al. 2025](https://www.sciencedirect.com/science/article/abs/pii/S0301562924004319)


## V0.3.1

### Hamstring models
In collaboration with the [ORB Michigan](https://www.kines.umich.edu/research/labs-centers/orthopedic-rehabilitation-biomechanics-laboratory), we developed models for the automatic segmentation of the biceps femoris. The dataset consisted of approximately 900 images from around 150 participants. Participants included were youth and adult soccer players, adult endurance runners, adult track and field athletes as well as adults with a recent ACL tear (in total 30% women). Images were captured across different muscle regions including 33%, 50% and 66% of muscle length. We compared the performance of different models to manual analysis of the images. We used similar training procedures as decribed in our DeepACSA [paper](https://journals.lww.com/acsm-msse/Abstract/9900/DeepACSA__Automatic_Segmentation_of.87.aspx), however, we evaluated the models unsing 5-fold cross-validation to check for overfitting. We provide the model with the highest IoU scores for ACSA segmentation. We compared the different model architectures [VGG16-Unet](https://journals.lww.com/acsm-msse/Abstract/9900/DeepACSA__Automatic_Segmentation_of.87.aspx), [Unet2+](https://arxiv.org/abs/1912.05074) and [Unet3+](https://arxiv.org/abs/2004.08790). Below we have outlined the analysis results and the trained models can be found [here](https://osf.io/a3u4v/). 

*Table 1. Comparison of model architectures throughout validation folds.*

![image](https://github.com/PaulRitsche/DeepACSA/assets/71383228/3844d5c7-8376-4016-9c2e-a36eb42f301e)

*Table 2. Comparison of model architectures to manual evaluation on external test set. all -> all Testsets; 1/2/3 -> only Testset 1/2/3; p -> panoramic; s -> single image; 1+2 -> without device 2 images (fewer images in training set), only Testset 1+2; rm -> with visual inspection; n -> number of images.*		

![image](https://github.com/PaulRitsche/DeepACSA/assets/71383228/f33e687a-3343-445d-be45-4be997d03a81)


## Descriptive figure of the model used

![Figure2_VGG16Unet](https://user-images.githubusercontent.com/71383228/182554020-2c8bad75-7f08-4194-8f5f-0a521a70781c.jpg)

DeepACSA workflow. a) Original ultrasound image of the m. rectus femoris (RF) at 50% of femur length that serves as input for the model. b) Detailed U-net CNN architecture with a VGG16 encoder (left path). c) Model prediction of muscle area following post-processing (shown as a binary image). 
