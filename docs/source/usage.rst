DL_Track Usage
==============

We have provided detailed usage instructions and examples for DL_Track in the DL_Track_tutorial.pdf file. The file is located in the `docs <https://github.com/PaulRitsche/DLTrack/tree/main/docs/usage>`_ directory in our Github repository. In general, should the graphical user interface freeze and you can't interact with it, close it and restart. Pay attention that all parameters in the interface are specified correctly.

**Attention MacOS users:**
The DL_Track package is only fully functional on windows OS. However, with restricted functionality macOS users can employ the DL_Track as well. With macOS, the manual scaling option for video and image analysis is not functional. Therefore, images cannot be scaled this way in the GUI. A possible solution is to scale the analysis results subsequent to completion of the analysis. Therefore, the pixel per centimeter must be calculated elsewhere. One option is to use `FIJI <https://imagej.net/software/fiji/downloads>`_. By drawing a line on the image, it is possible to see the length of the line in pixel units. Open the respective image in FIJI by drag and drop. Draw a line on the image with a known distance of one centimetre, click `cmd + m` and get the length of the line in pixel unit from the result window. Do that for every image with varying scanning depth. Divide the analysis results for muscle thickness and fascicle length by the linelength in pixel units. The result will be the muscle thickness and fascicle length in centimeter units.

