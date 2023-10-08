.. DeepACSA documentation master file, created by
   sphinx-quickstart on Wed Dec 28 16:24:03 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepACSA's documentation!
====================================
*DeepACSA: Automated analysis of human lower limb ultrasonography images*

So, what is DeepACSA all about? The DeepACSA algorithm was developed in 2022 by `Paul Ritsche <https://twitter.com/ritpau>`_ & `Philipp Wirth <https://twitter.com/phippli>`_. Moreover, `Neil Cronin <https://twitter.com/NeilJCronin84>`_, `Fabio Sarto <https://twitter.com/FabioSarto3>`_, Marco Narici, `Oliver Faude <https://twitter.com/OliverFaude>`_ and `Martino Franchi <https://twitter.com/MVFranchi>`_ supported the project during development and provided images, materials and insight. The algorithm makes extensive use of fully convolutional neural networks trained on a fair amount of ultrasonography images of the human lower limb. Specifically, the dataset included transversal ultrasonography images from the human gastrocnemius medialis and lateralis, vastus lateralis and rectus femoris. The algorithm is able to analyse muscle anatomical cross-sectional area, echo intensity and muscle volume. 

Why use DeepACSA?
=================

Using the DeepACSA algorithm to analyze muscle morphological parameters in human lower limb muscle ultrasonography images hase two main advantages. The analysis is objectified when using the automated analysis types for images because no user input is required during the analysis process. Secondly, the required analysis time for image analysis is drastically reduced compared to manual analysis. Whereas an image manual analysis takes about one minute, DeepACSA analyzes images in less than one second. This allows users to analyze large amounts of images without supervision during the analysis process in relatively short amounts of time.

Limitations
===========

Currently, we have not provided unit testing for the functions and modules included in the DeepACSA algorithm. Moreover, the muscles included in the training data set are limited to the lower extremities. Although we included images from as many ultrasonography devices as possible, we were only able to include images from four different devices. Therefore, users aiming to analyze images from different muscles or different ultrasonography devices might be required to train their own models because the provided pre-trained models result in bad segmentations. Lastly, even though  DeepACSA objectifies the analysis of ultrasonography images when using the automated analysis types, we labeled the images manually. Therefore, we introduced some subjectivity into the datasets.

.. toctree::
   :caption: Contents
   :hidden:

   News
   installation
   usage
   contribute
   tests
   Documentation




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
