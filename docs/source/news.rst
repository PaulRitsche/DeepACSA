News
==========

v0.3.1: 
-------

1. Code:
    - **Upgraded to Python 3.9.18** and adapted compatibility
    - Restructuring of the GUI 
    - Inclusion of Mask creation inside GUI (see :ref:`datalabel`)
    - Inclusion of Mask inspection inside GUI (see :ref:`masklabel`)
    - Removal of the need to specify image types 

2. Data:
    We included a new model architecture as pre-trained models in the DeepACSA dataset for the RF and VL muscles. The new dataset can be accessed `here <https://doi.org/10.5281/zenodo.8419487>`_
    These model architectures were: 

    - **UNet3+** `(Huang et al., 2020) <https://doi.org/10.48550/arXiv.2004.08790>`_ for RF and VL: It uses full-scale skip connections, causing increased skip connections. These additional skip connections in between layers help to preserve data that is needed for further training. Full-scale skip connections incorporate low-level details with high-level semantics from feature maps in different scales, which is beneficial for medical image segmentation.
    
    To summarise the results for the rectus femoris, for the visually inspected (segmented masks were visually checked for plausibility) rectus femoris images, the original architecture was slightly exceeded by the UNet 3+. Moving towards the vastus lateralis, the UNet 3+ closely followed by TransUNet outperformed the original architecture on the uninspected data. For the visually inspected data of the vastus lateralis, the UNet 3+ outperformed the other models.
    We provide two more models and compared them to manual analysis (Fig. 1-4). Users are adivsed to select the performing models for VL and RF. However, we still advise to test the selected model prior to usage. 
    All models were trained using K-fold (5 folds used) cross-validation to ensure that overfitting is reduced. 

    .. figure:: RF_without_inspection.png
        :scale: 50 %
        :alt: inspect_figure

        Figure 1: Comparison of RF images to manual analysis with uninspected results.

    .. figure:: RF_with_inspection.png
        :scale: 50 %
        :alt: inspect_figure

        Figure 2: Comparison of RF images to manual analysis with inspected results.

    .. figure:: VL_without_inspection.png
        :scale: 50 %
        :alt: inspect_figure

        Figure 2: Comparison of VL images to manual analysis with inspected results.

    .. figure:: VL_with_inspection.png
        :scale: 50 %
        :alt: inspect_figure

        Figure 2: Comparison of VL images to manual analysis with inspected results.
