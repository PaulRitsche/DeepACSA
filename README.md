![Title_Image](https://user-images.githubusercontent.com/71383228/148084248-5e169761-a075-4c86-836a-e9b2cec56ef6.jpg)

# DeepACSA

DeepACSA is an open-source tool to evaluate the anatomical cross-sectional area of muscles in ultrasound images using deep learning.
More information about the usage of DeepACSA can be found in the instructional [video](https://youtu.be/It9CqVSNc9M) 
Our trained models can be accessed at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6953924.svg)](https://doi.org/10.5281/zenodo.6953924).
Anonymized panoramic ultrasound images are available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5799204.svg)](https://doi.org/10.5281/zenodo.5799204).
If you find this work useful, please remember to cite the corresponding [paper](https://journals.lww.com/acsm-msse/Abstract/9900/DeepACSA__Automatic_Segmentation_of.87.aspx), where more information about the model architecture and performance can be found as well. 

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
