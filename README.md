# DeepACSA

DeepACSA is an open-source tool to evaluate the anatomical cross-sectional area of muscles in ultrasound images using deep learning.
All following commands should be entered in your command prompt / terminal.
More information about DeepACSA can be found in our publication (LINK) and the instructional video (LINK). 
If you find this project helpful, please remember to cite us. 

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

You can also download the .zip file of the repository and save it to a specific folder. Navigate to the folder and continue with the Anaconda setup.

3. Anaconda setup (only before first usage)

Install Python / Anaconda: https://www.anaconda.com/distribution/ (click ‘Download’ and be sure to choose ‘Python 3.X Version’ (where the X represents the latest version being offered. IMPORTANT: Make sure you tick the ‘Add Anaconda to my PATH environment variable’ box).
Open an Anaconda prompt window and create a virtual environment (This may take some time):

```sh
conda env create -f environment.yml 
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
Activate your virtual environment and change to the directory containing the code, as done above. So far, we have implemented two ways to use DeepACSA. 

1. Make use of the implemented GUI:

Type the here presented command into the prompt window and enter the required parameters in the GUI. 

```sh
python deep_asca_gui.py

```

2. Run DeepACSA from the command promt entirely:

Type the here presented command into the prompt window, while entering the required parameters. 

```sh
deep_acsa [-h] -rp ROOTPATH -mp MODELPATH -d DEPTH [-sp SPACING] -m MUSCLE -s SCALING

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
