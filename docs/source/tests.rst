.. _testlabel:

Tests
=====

Image analaysis
"""""""""""""""
So far, we have not included unit testing in DeepACSA. However, in the DeepACAS_example that you can download `here <https://zenodo.org/record/8007009>`_  you will find an *images_test* folder. 
There, we have included three rectus femoris images as well as an *Original_Results.xlsx* file. Analyse these images using the ``RF`` muscle option,  ``Line`` scaling option and an ``Imagedepth`` of **4.5**. 
Tick the ``Calculate Volume`` checkbox. Compare your results with the one in the *Original_Results.xlsx*. Should the identical, DeepACSA works fine. 

Model training
""""""""""""""
In order to test wheter the model training option included in DeepACSA is functional, open the ``Train Model`` model
by clicking the respective button. Then select the **image** folder in the "DeepACSA_example/training_test" folder clicking the ``Images``
button and the **mask** folder in the "DeepACSA_example/training_test" folder clicking the ``Mask`` button. Subsequently, select the "DeepACSA_example/training_test" as 
output directory by clicking the ``Output`` button. Leave all training parameters as specified and click ``Start Training``. 
Once the training process is completed you should have a results plot, a .h5 model file and a .xlsx weights file in the output directory.
If this is the case, the training process worked!
