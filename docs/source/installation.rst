.. _installation:

Installation
============

We offer two possible installation approaches for our DeepACSA software. The first option is to download the DeepACSA Installer. The second option we describe is DeepACSA package installation via Github. We want to inform you that there are more ways to install the package. However, we do not aim to be complete and rather demonstrate an (in our opinion) user friendly way for the installation of DeepACSA. Moreover, we advise users with less programming experience to make use of the first option and download the Installer.

Download the DeepACSA Installer
---------------------------------

1. Got to the Zenodo webpage containing the DeepACSA executable, the pre-trained models and the example files using this `link <https://doi.org/10.5281/zenodo.19130694>`_.
2. Download the ``DeepACSAv0.3.2_Installer`` folder and extract its content on your Desktop. If your system does not support .7z files natively, use a program such as 7-Zip to extract the contents.
3. Navigate to ``C:/Users/your_user/Desktop/Downloads`` and install DeepACSA v0.3.2 by double clicking the ``DeepACSAv0.3.2_Installer.exe``.
4. Navigate to ``C:/Users/your_user/Desktop/DeepACSA_example_v0.3.2/``. Start with the :ref:`testing procedure <testlabel>` to check that everything works properly.

.. _installlabel:

DeepACSA Installation Guide (v0.3.2)
------------------------------------

.. warning::

   Version **0.3.2 is NOT YET available via pip**.
   The newest release must be installed from GitHub.

Overview
--------

DeepACSA can be installed using a Conda environment and a local editable installation.
This is the recommended approach for both users and developers.

Step 1: Install Anaconda (required)
-----------------------------------

If you do not have Anaconda or Miniconda installed:

- Download: https://www.anaconda.com/distribution/
- Choose a Python 3.X version
- Ensure that *"Add Anaconda to PATH"* is enabled during installation

Step 2: Install Git (recommended)
---------------------------------

Git is required to clone the repository:

- https://git-scm.com/download

Step 3: Clone the DeepACSA Repository
-------------------------------------

Open a terminal (or Git Bash) and run:

.. code-block:: bash

   git clone https://github.com/PaulRitsche/DeepACSA.git
   cd DeepACSA

Step 4: Create the Conda Environment
------------------------------------

DeepACSA provides an environment file with all dependencies:

.. code-block:: bash

   conda env create -f environment.yml

This will create the environment (typically named ``DeepACSA0.3.2``).

Step 5: Activate the Environment
--------------------------------

.. code-block:: bash

   conda activate DeepACSA0.3.2

Step 6: Install DeepACSA Locally
--------------------------------

Install the package in editable mode:

.. code-block:: bash

   python -m pip install -e .

This allows local development and updates.

Step 7: Verify Installation
---------------------------

.. code-block:: bash

   conda list

Ensure that ``deepacsa`` appears in the list of installed packages.

Step 8: Run DeepACSA
---------------------

Option 1: Run as module (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m DeepACSA

This works from any directory as long as the environment is active.

Option 2: Run GUI script directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd DeepACSA
   python deep_acsa_gui.py

Alternative: Older Versions (not recommended)
---------------------------------------------

If you explicitly need an older version (e.g. 0.3.1):

.. code-block:: bash

   pip install DeepACSA==0.3.1

.. warning::

   Older versions may not include recent updates and fixes.
   The older version runs in python 3.9.18. Please consider this when installing the older version.

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
