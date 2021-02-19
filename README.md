# Deep-ACSAuto

Deep-ACSAuto is an open-source tool to evaluate the anatomical cross-sectional area of muscles in ultrasound images using deep learning.

## Installation 

1. Clone the Github repository. Type the following code in your command window:

```sh
git clone https://github.com/PaulRitsche/ACSAuto_DeepLearning
```

2. Anaconda setup (only before first usage)

Install Python / Anaconda: https://www.anaconda.com/distribution/ (click ‘Download’ and be sure to choose ‘Python 3.X Version’ (where the X represents the latest version being offered. IMPORTANT: Make sure you tick the ‘Add Anaconda to my PATH environment variable’ box)
Open an Anaconda prompt window (in windows, click ‘start’ then use the search window), then create a virtual environment using the following as an example (here we use "Deep_ACSAuto" as the virtual environment name, but this can be anything):
conda create --name Deep_ACSAuto python=3.8
(If prompted, type y to confirm that the relevant packages can be installed)
Activate the virtual environment by typing activate Deep_ACSAuto (where Deep_ACSAuto is replaced by the name you chose).
cd to where you have saved the project folder, e.g. by typing cd c:/Deep_ACSAuto
type the following command: pip install -r requirements.txt
(this step may take some time)
type jupyter notebook and Jupyter notebooks should load in your browser

3. Usage

Open an Anaconda prompt window
Activate your virtual environment, as done in step 4 above
cd to the folder where you have the tracking software, e.g. cd c:/Deep_ACSAuto
Type jupyter notebook in the prompt window
Now you should see the different Jupyter notebooks that allow you to train a model or to run inference on single images or videos (each labelled accordingly)
Open the notebook you need. Within each notebook, use ctrl-enter to run a cell
