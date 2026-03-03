.. _installation:

Installation
============

We offer two possible installation approaches for our DeepACSA software. The first option is to download the DeepACSA executable file. The second option we describe is DeepACSA package installation via Github. We want to inform you that there are more ways to install the package. However, we do not aim to be complete and rather demonstrate an (in our opinion) user friendly way for the installation of DeepACSA. Moreover, we advise users with less programming experience to make use of the first option and download the executable file.

Download the DeepACSA executable
---------------------------------

1. Got to the Zenodo webpage containing the DeepACSA executable, the pre-trained models and the example files using this `link <https://doi.org/10.5281/zenodo.8419487>`_.
2. Download the ``DeepACSA_example_v0.3.2.7z`` folder and extract its content on your Desktop. If your system does not support .7z files natively, use a program such as 7-Zip to extract the contents.
3. Navigate to ``C:/Users/your_user/Desktop/DeepACSA_example_v0.3.2/executable/`` and open the DeepACSA GUI by double clicking the ``DeepACSA_v0.3.2.exe``. Start with the :ref:`testing procedure <testlabel>` to check that everything works properly.

.. _installlabel:

Install DeepACSA via Github
---------------------------

In case you want to use this way to install and run DeepACSA, we advise you to setup conda (see step 1) and download the environment.yml file from the repo (see steps 5-8). If you want to actively contribute to the project or customize the code, it might be usefull to you to do all of the following steps (for more information see :ref:`contributelabel`).

**Step 1: Anaconda setup**. *This is necessary only before first usage and if Anaconda/minicoda is not already installed*.

    Install `Anaconda <https://www.anaconda.com/distribution/>`_ (click 'Download' and be sure to choose 'Python 3.X Version', where the X represents the latest version being offered. IMPORTANT: Make sure you tick the 'Add Anaconda to my PATH environment variable' box).

**Step 2: Git setup**. *This is optional and only required when you want to clone the whole DeepACSA Github repository. It is only required for MacOS users, who are intending to contribute or develop and is only necessary before first usage and if Git is not already installed.*

    In case you have never used Git before on you computer, please install it using the instructions provided in the `Git guide <https://git-scm.com/download>`_.

**Step 3: Create a desktop directory for DeepACSA**. *Just as for step 2, this is only required for MacOS users, contributing or developing.*

    On your computer create a specific desktop directory for DeepACSA named ``DeepACSA``.

**Step 4: Clone the DeepACSA Github repository into the pre-specified folder "DeepACSA"**. *Just as for step 2 and 3, this is only required for MacOS users, contributing or developing.*
    
    You can use Git as a version control system. Open "Git Bash" on your computer and navigate to the folder you just created with the following command:
        .. code-block:: bash
            
            cd ~/Desktop/DeepACSA
    Now you are in the right working directory and you can type the following command in your bash window to clone the repository:
        .. code-block:: bash
            
            git clone https://github.com/PaulRitsche/DeepACSA.git

    This will clone the entire repository to your local computer. To make sure that everything worked, see if the files in your local directory match the ones you can find in the Github DeepACSA repository. If you run into problem, check this `website <https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository>`_.

    Alternatively, you can only download the environment.yml file from the `DeepACSA repository <https://github.com/PaulRitsche/DeepACSA.git>`_ and continue to the next step.

**Step 5: Create the virtual environment required for DeepACSA**.

    DeepACSA is bound to a specific python version (3.10). To create an environment for DeepACSA, type the following command in your Git bash terminal:
        .. code-block:: bash
            
            conda create -n DeepACSA python=3.10
            
    You will have to accept the conda environment conditions by entering "y" into the bash terminal when prompted.

**Step 6: Activate the environment for usage of DeepACSA**.

    You can now activate the virtual environment by typing:
        .. code-block:: bash
            
            conda activate DeepACSA
    
    An active conda environment is visible in () brackets before your current path in the bash terminal. In this case, this should look something like (DeepACSA) C:/user/Desktop/DeepACSA.

.. _step7:

**Step 7: Install the DeepACSA package**. *For Windows*.

    You can downlaod and install the DeepACSA package by typing in your bash terminal:
        .. code-block:: bash

            pip install DeepACSA==0.3.2

**Step 7: Install the DeepACSA package**. *For MacOS*.

    Navigate into the folder that you cloned from Github named ``DeepACSA`` with the Git bash terminal. You can do that by typing "cd" followed by the path to the folder containing the requirements.txt file. 
    
    This should look something like:
        .. code-block:: bash

            cd ~/Desktop/DeepACSA/DeepACSA
    
    Then you can install the requirements of DeepACSA with:
        .. code-block:: bash

            pip install -r requirements.txt
    
    Install the DeepACSA package locally to make use of its functionalities with:
        .. code-block:: bash

            python -m pip install -e .

**Step 8: Check proper packages installation**.

    To verify that all packages were installed correctly in your virtual environment, run the following command in your Bash terminal:
        .. code-block:: bash

            conda list

    This command displays all installed packages. Check that *deepacsa* appears in the list. If it is not present, return to :ref:`step 7<step7>` and repeat the installation.

**Step 9: Run DeepACSA - first option**. 
    
    The first option of running DeepACSA is using the installed DeepACSA package (either by pip or locally installed). You do not need the whole cloned repository for this, only the active DeepACSA environment. Moreover, to start the GUI this way the location of you prompt is irrelevant, as long as the DeepACSA conda environment is activated. 
    
    Type in your bash terminal:
        .. code-block:: bash

            python -m Deep_ACSA
        
**Step 9: Run DeepACSA - second option**.

    The second option of running DeepACSA is using the ``deep_acsa_gui.py`` python script.
    
    This requires you to clone the whole directory and navigate to the directory where the deep_acsa_gui.py file is located (``C:\Users\your_user\Desktop\DeepACSA\DeepACSA\deep_acsa_gui.py``). Moreover, you need the active DeepACSA environment.

    To execute the module type the following command in your terminal:
        .. code-block:: bash

            python deep_acsa_gui.py

Whichever option you used, the main GUI should now open. 

.. figure:: main.png
    :scale: 75 %
    :alt: main_gui_figure

    Main GUI Window


*If you run into problems, open a discussion in the Q&A section of* `DeepACSA discussions <https://github.com/PaulRitsche/DeepACSA/discussions/categories/q-a>`_ *and assign the label "Problem". You can find an example discussion there. For usage of DeepACSA please take a look at* :ref:`usagelabel`.

.. _gui_setup_ref:

GPU setup
---------

The processing speed of a single image or video frame analyzed with DeepACSA is highly dependent on computing power. While possible, model inference and model training using a CPU only will decrese processing speed and prolong the model training process. Therefore, we advise to use a GPU whenever possible. 

**The following instructions are relevant only for Windows users!**

Before using a GPU it needs to be properly configured. First, install the appropriate GPU drivers on your system. The correct drivers for your GPU can be found on the `official NVIDIA website <https://www.nvidia.com/Download/index.aspx?lang=en-us>`_.

After installing the drivers, you must install the required CUDA and cuDNN software packages. To use DeepACSA with TensorFlow version 2.10, install:

* CUDA version 11.2 from the `CUDA 11.2 download archive <https://developer.nvidia.com/cuda-11.2.0-download-archive>`_.
* cuDNN version 8.5 for CUDA version 11.x from the `cuDNN archive <https://developer.nvidia.com/rdp/cudnn-archive>`_. 

You may need to create an NVIDIA account to access these downloads.

As a next step, you need to be your own installation wizard. We refer to this `video <https://www.youtube.com/watch?v=OEFKlRSd8Ic>`_ (up to date, watch from minute 9 to minute 13) or this `older video <https://www.youtube.com/watch?v=IubEtS2JAiY&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=2>`_ (watch the entire video, but substitute CUDA and cuDNN versions accordingly). There are procedures at the end of each video testing whether a GPU is detected by tensorflow or not. 

**The following instructions are relevant only for MacOS users!**

If you are using an Apple Silicon device (M1 or M2) and would like to enable GPU acceleration for model training and/or inference, please refer to this `tutorial <https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706>`_. It provides a detailed guide on enabling GPU support for TensorFlow on Apple Silicon devices.


*If you run into problems with the GPU/CUDA setup, please open a discussion in the Q&A section of* `DeepACSA discussions <https://github.com/PaulRitsche/DeepACSA/discussions/categories/q-a>`_ *and assign the label "Problem".*
