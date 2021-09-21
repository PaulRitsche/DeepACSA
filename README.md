# Deep-ACSAuto

Deep-ACSAuto is an open-source tool to evaluate the anatomical cross-sectional area of muscles in ultrasound images using deep learning.

## Installation 

1. Clone the Github repository. Type the following code in your command window:

```sh
git clone https://github.com/PaulRitsche/DeepACSA.git
```

2. Anaconda setup (only before first usage)

Install Python / Anaconda: https://www.anaconda.com/distribution/ (click ‘Download’ and be sure to choose ‘Python 3.X Version’ (where the X represents the latest version being offered. IMPORTANT: Make sure you tick the ‘Add Anaconda to my PATH environment variable’ box).
Open an Anaconda prompt window and create a virtual environment using the following as an example (you can choose the name of the environment freely): 

```sh
conda create --name DeepACSA python=3.8
```

Activate the virtual environment by typing:

```sh
conda activate DeepACSA 
```
Change the directory to where you have saved the project folder, e.g. by typing:

```sh
cd c:/Users/Paul/Desktop/DeepACSA
```

Then, type the following command to install all requirements for the code (takes some time): 

```sh
pip install -r requirements.txt
```

3. GPU setup

When you use a GPU or want to train your own models, remember to check whether your CUDA setup matches the tensorflow version of this project.

## Usage

Open an Anaconda prompt window.
Activate your virtual environment and change to the directory containing the code, as done above. So far, we have implemented two ways to use DeepACSA. 

1. Make use of the implemented GUI:

Type the here presented command into the prompt window and enter the required parameters in the GUI. 

```sh
python deep_asca_gui.py

```

2. Run DeepACSA from the command promt entirely:

Type the here presented command into the prompt window, while entering the required parameters. 

```sh
deep_acsa [-h] -rp ROOTPATH [-fp FLIP_FLAG_PATH] -mp MODELPATH -d DEPTH [-sp SPACING] -m MUSCLE -s SCALING

```

## Parameters

```console
required arguments:
  -rp, --rootpath 
      path to root directory of images
  -mp, modelpath
  	  file path to .h5 file containing model used for prediction
  -d, --depth
  	  Ultrasound scanning depth (cm)
  -m, --muscle
  	  muscle that is analyzed
  -s, --scaling
  	  scaling type present in ultrasound image

optional arguments:
  -sp, --spacing
  	  distance (mm) between detetec vertical scaling lines
  -h, --help
      show this help message and exit
```

## Examples

This is an example command making use of the implemented GUI:
```sh
python deep_acsa_gui.py 
```

This is an example command for an extended-field-of-view ultrasound image containing a continuous scaling line (see Image):
```sh
python deep_acsa.py -rp "C:\Users\Paul\Desktop\Test_image" -mp "C:\Users\Paul\Desktop\Test_image\model\model.h5" -d 6 -m "RF" -s "EFOV"
```
![RF_EFOV_CONT](https://user-images.githubusercontent.com/71383228/110342363-9a8fda80-802b-11eb-93ec-c643c499449a.jpg)

This is an example command for an extended-field-of-view ultrasound image containing scaling bars:
```sh
python deep_acsa.py -rp "C:\Users\Paul\Desktop\Test_image" -mp "C:\Users\Paul\Desktop\Test_image\model\model.h5" -d 6 -sp 5 -m "RF" -s "Static"
```
This is an example command for an extended-field-of-view ultrasound image where manual scaling is used:
```sh
python deep_acsa.py -rp "C:\Users\Paul\Desktop\Test_image" -mp "C:\Users\Paul\Desktop\Test_image\model\model.h5" -d 6 -sp 5 -m "RF" -s "Manual"
```
Please note that optional parameters can be used for "Static" and "Manual" scaling options. 
