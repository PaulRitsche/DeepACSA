![Title_Image](https://user-images.githubusercontent.com/71383228/148084248-5e169761-a075-4c86-836a-e9b2cec56ef6.jpg)

# DeepACSA
[![Documentation Status](https://readthedocs.org/projects/deepacsa/badge/?version=latest)](https://deepacsa.readthedocs.io/en/latest/?badge=latest)

*Automatic analysis of human lower limb ultrasonography images*

DeepACSA is an open-source tool to evaluate the anatomical cross-sectional area of muscles in ultrasound images using deep learning.
More information about the installtion and usage of DeepACSA can be found in the online [documentation](https://deepacsa.readthedocs.io/en/latest/index.html). You can find information about contributing, issues and bug reports there as well.
Our trained models, training data, an executable as well as example files can be accessed at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8419487.svg)](https://doi.org/10.5281/zenodo.8419487).
If you find this work useful, please remember to cite the corresponding [paper](https://journals.lww.com/acsm-msse/Abstract/9900/DeepACSA__Automatic_Segmentation_of.87.aspx), where more information about the model architecture and performance can be found as well. 

## Whats new?

With version 0.3.1, we included new models for the m. vastus lateralis (VL) and m. rectus femoris (RF) and added manual image labelling and mask inspection to the GUI. Take a look at our [documentation](https://deepacsa.readthedocs.io/en/latest/index.html) to see more details and the result of the model comparisons.

### Hamstring models
In collaboration with the [ORB Michingan (https://www.kines.umich.edu/research/labs-centers/orthopedic-rehabilitation-biomechanics-laboratory)], we developed models for the automatic segmentation of the biceps femoris. The dataset consited of ... images from aroung 150 participants. Participants included were youth and adult soccer players, 
adult endurance runners, adult track and field athletes as well as adults with a recent ACL tear (in total 30% women). We compared the performance of different models to manual analysis of the images. We used similar training procedures as decribed in our DeepACSA [paper](https://journals.lww.com/acsm-msse/Abstract/9900/DeepACSA__Automatic_Segmentation_of.87.aspx), however, we evaluated the models unsing 5-fold cross-validation to counteract overfitting. We provide the model with the highest IoU scores for ACSA segmentation. We compared the same model architectures as described previously in the "What's new" [section (https://deepacsa.readthedocs.io/en/latest/news.html)]. Below we have outlined the analysis results and the trained models can be found [here ()]. 

*Table 1. Comparison of model architectures throughout validation folds.*

![image](https://github.com/PaulRitsche/DeepACSA/assets/71383228/3844d5c7-8376-4016-9c2e-a36eb42f301e)

*Table 2. Comparison of model architectures to manual evaluation on external test set. all -> all Testsets; 1/2/3 -> only Testset 1/2/3; p -> panoramic; s -> single image; 1+2 -> without device images (fewer images in training set), only Testset 1+2; rm -> with visual inspection; n -> number of images.*		

![image](https://github.com/PaulRitsche/DeepACSA/assets/71383228/f33e687a-3343-445d-be45-4be997d03a81)


## Quickstart

To quickly start the DeepACSA either open the executable or type 

``python -m Deep_ACSA``

in your prompt once the package was installed locally with

``pip install DeepACSA==0.3.1.``

when the DeepACSA environment is activated. 
Irrespective of the way the software was started, the GUI should open and is ready to be used.

![main_gui](https://github.com/PaulRitsche/DeepACSA/assets/71383228/b3a48daf-58ea-4971-badd-dcd6387000b7)


## Descriptive figure of the model used

![Figure2_VGG16Unet](https://user-images.githubusercontent.com/71383228/182554020-2c8bad75-7f08-4194-8f5f-0a521a70781c.jpg)

DeepACSA workflow. a) Original ultrasound image of the m. rectus femoris (RF) at 50% of femur length that serves as input for the model. b) Detailed U-net CNN architecture with a VGG16 encoder (left path). c) Model prediction of muscle area following post-processing (shown as a binary image). 

## Results of comparing DeepACSA analysis to manual analysis

![Figure3_BAP](https://user-images.githubusercontent.com/71383228/182554096-c5fde7cd-a137-4cc4-ad73-a819368d13ec.jpg)

Bland-Altman plots of all muscles plotting the difference between manual and DeepACSA with incorrect predictions removed (rm), manual and DeepACSA as well as manual and ACSAuto area segmentation measurements against the mean of both measures. Dotted and solid lines illustrate 95% limits of agreement and bias. M. rectus femoris (RF) and m. vastus lateralis (VL), mm. gastrocnemius medialis (GM), and lateralis (GL).
