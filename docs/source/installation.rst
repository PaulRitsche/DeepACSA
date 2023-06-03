Installation
============

**Attention: The installation procedure differes for macOS and windows users!**

We offer two possible installation approaches for our DeepACSA software. The first option is to download the DeepACSA executable file. The second option we describe is DeepACSA package installation via Github. We want to inform you that there are more ways to install the package. However, we do not aim to be complete and rather demonstrate an (in our opinion) user friendly way for the installation of DeepACSA. Moreover, we advise users with less programming experience to make use of the first option and download the executable file.

Download the DeepACSA executable
---------------------------------

1. Got to the Zenodo webpage containing the DeepACSA executable, the pre-trained models and the example files using this `link <>`_.
2. Download the DeepACSA_example.zip
3. Find the DeepACSA.exe Executable in the DeepACSA_example/executable folder.
4. Create a specified DeepACSA directory and put the DeepACSA.exe, the model files and the example file in seperate subfolders (for example "Executable", "Models" and "Example"). Moreover, unpack the DeepACSA_example.zip file.
5. Open the DeepACSA GUI by double clicking the DeepACSA.exe file and start with the testing procedure to check that everything works properly.

Install DeepACSA via Github
---------------------------

In case you want to use this way to install and run DeepACSA, we advise you to setup conda (see step 1) and download the environment.yml file from the repo (see steps 5-8). If you want to actively contribute to the project or customize the code, it might be usefull to you to do all of the following steps (for more information see :ref:`contributelabel`).

*Step 1.* Anaconda setup (only before first usage and if Anaconda/minicoda is not already installed).

Install `Anaconda <https://www.anaconda.com/distribution/>`_ (click 'Download' and be sure to choose 'Python 3.X Version' (where the X represents the latest version being offered. IMPORTANT: Make sure you tick the 'Add Anaconda to my PATH environment variable' box).

*Step 2.* **(Only required for MacOS users, contributing or development)** Git setup (only before first usage and if Git is not already installed). This is optional and only required when you want to clone the whole DeepACSA Github repository.

In case you have never used Git before on you computer, please install it using the instructions provided `here <https://git-scm.com/download>`_.

*Step 3.* **(Only required for MacOS users, contributing or development)** Create a directory for DeepACSA.

On your computer create a specific directory for DeepACSA (for example "DeepACA") and navigate there. You can use Git as a version control system. Once there open a git bash with right click and then "Git Bash Here". In the bash terminal, type the following:

``git init``

This will initialize a git repository and allows you to continue. If run into problems, check this `website <https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository>`_.

*Step 4.* **(Only required for MacOS users, contributing or development)** Clone the DeepACSA Github repository into a pre-specified folder (for example "DeepACSA") by typing the following code in your bash window:

``git clone https://github.com/PaulRitsche/DeepACSA.git``

This will clone the entire repository to your local computer. To make sure that everything worked, see if the files in your local directory match the ones you can find in the Github DeepACSA repository. If you run into problem, check this `website <https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository>`_.

Alternatively, you can only download the environment.yml file from the `DeepACSA repo <https://github.com/PaulRitsche/DeepACSA.git>`_ and continue to the next step.

*Step 5.* Create the virtual environment required for DeepACSA.

DeepACSA is bound to a specific python version (3.8.13). To create an environment for DeepACSA, type the following command in your Git bash terminal:

``conda create -n DeepACSA python=3.8``

*Step 6.* Activate the environment for usage of DeepACSA.

You can now activate the virtual environment by typing:

``conda activate DeepACSA``

An active conda environment is visible in () brackets befor your current path in the bash terminal. In this case, this should look something like (DeepACSA) C:/user/.../DeepACSA.Then, download the DeepACSA package by typing:

*Step 7.* Install the DeepACSA package.

Navigate into the folder that you cloned from Github (DeepACSA) with the bash terminal. You can do that by typing "cd" followed by the path to the folder containing the requirements.txt file. This should look something like:

``cd /.../.../DeepACSA/DeepACSA``

Then you can install the requirements of DeepACSA with: 

``pip install -r requirements.txt``

Install the DeepACSA package locally to make use of its functionalities with:

``python -m pip install -e .``

*Step 8.* The First option of running DeepACSA is using the installed DeepACSA package. You do not need the whole cloned repository for this, only the active DeepACSA environment. You do moreover not need be any specific directory. Type in your bash terminal:

``python -m Deep_ACSA``

The main GUI should now open. If you run into problems, open a discussion in the Q&A section of `DeepACSA discussions <https://github.com/PaulRitsche/DeepACSA/discussions/categories/q-a>`_ and assign the label "Problem".  For usage of DeepACSA please take a look here :ref:`usagelabel`.

*Step 9.* The second option of running DeepACSA is using the deep_acsa_gui python script. This requires you to clone the whole directory and navigate to the directory where the deep_acsa_gui.py file is located. Moreover, you need the active DeepACSA environment.

The deep_acsa_gui.py file is located at the `DeepACSA/Deep_ACSA` folder. To execute the module type the following command in your bash terminal.

``python deep_acsa_gui.py``

The main GUI should now open. If you run into problems, open a discussion in the Q&A section of `DeepACSA discussions <https://github.com/PaulRitsche/DeepACSA/discussions/categories/q-a>`_ and assign the label "Problem". You can find an example discussion there. For usage of DeepACSA please take a look here :ref:`usagelabel`.


GPU setup
---------

**Attention: The next section is only relevant for windows users!**

The processing speed of a single image or video frame analyzed with DeepACSA is highly dependent on computing power. While possible, model inference and model training using a CPU only will decrese processing speed and prolong the model training process. Therefore, we advise to use a GPU whenever possible. Prior to using a GPU it needs to be set up. Firstly the GPU drivers must be locally installed on your computer. You can find out which drivers are right for your GPU `here <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_. Subsequent to installing the drivers, you need to install the interdependant CUDA and cuDNN software packages. To use DeepACSA with tensorflow version 2.10 you need to install CUDA version 11.2 from `here <https://developer.nvidia.com/cuda-11.2.0-download-archive>`_ and cuDNN version 8.5 for CUDA version 11.x from `here <https://developer.nvidia.com/rdp/cudnn-archive>`_ (you may need to create an nvidia account). As a next step, you need to be your own installation wizard. We refer to this `video <https://www.youtube.com/watch?v=OEFKlRSd8Ic>`_ (up to date, minute 9 to minute 13) or this `video <https://www.youtube.com/watch?v=IubEtS2JAiY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=2>`_ (older, entire video but replace CUDA and cuDNN versions). There are procedures at the end of each video testing whether a GPU is detected by tensorflow or not. If you run into problems with the GPU/CUDA setup, please open a discussion in the Q&A section of `DeepACSA discussions <https://github.com/PaulRitsche/DeepACSA/discussions/categories/q-a>`_ and assign the label "Problem".

**Attention : The next section is only relevant for MacOS users!**

In case you want to make use of you M1 / M2 chips for model training and / or inference, we refer you to this `tutorial <https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706>`_. There you will find a detailed description of how to enable GPU support for tensorflow. It is not strictly necessary to do that for model training or inference, but will speed up the process.
