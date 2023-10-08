Whats new?
==========

v0.3.1: 
-------

1. Code:
    - **Upgraded to Python 3.9.18** and adapted compatibility
    - Restructuring of the GUI 
    - Inclusion of Mask creation inside GUI (see :ref:`trainlabel`)
    - Inclusion of Mask inspection inside GUI (see :ref:`trainlabel`)
    - Removal of the need to specify image types 

2. Data:
    We included three new model architectures as pre-trained models in the DeepACSA dataset for the RF and VL muscles.
    These model architectures were: 

    - **UNet++** `(Zhou et al., 2018) <https://link.springer.com/chapter/10.1007/978-3-030-00889-5_1>`_ for RF and VL: It is designed to improve the performance of the original U-Net architecture, while using nested dense skip connections between the encoding and decoding block. While applying multi-scale context aggregation and up-sampling with concatenation, the semantic gap between the encoder and decoder part tends to decrease. The semantic gap describes the correlation between the pixel information and the interpretation of the pixel information. Pretrained weights from ImageNet and a VGG16 backbone are used. Without deep supervision, UNet++ theoretically achieves a significant performance gain over both U-Net and wide U-Net. These techniques help the network to extract more detailed features and focus on more relevant information, leading to better segmentation performance than the original U-Net.
    - **UNet3+** `(Huang et al., 2020) <https://doi.org/10.48550/arXiv.2004.08790>`_ for RF and VL: It uses full-scale skip connections, causing increased skip connections. These additional skip connections in between layers help to preserve data that is needed for further training. Full-scale skip connections incorporate low-level details with high-level semantics from feature maps in different scales, which is beneficial for medical image segmentation.
    - **TransUnet** `(Chen et al., 2021) <https://arxiv.org/abs/2102.04306>`_ for VL: The TransUNet architecture combines U-Net and transformer architecture to produce an effective semantic segmentation network. The network captures long-range dependencies more effectively and extracts high-level representations using patch extraction and patch embedding layers such as multi-head, self-attention, and transformer-encoder blocks. Skip connections and bilinear interpolation also help maintain spatial information and generate high-resolution feature maps to enable precise object localisation.

    To summarise the results for the rectus femoris, the UNet++ outperformed the original architecture on the uninspected data (segmented masks were not visually checked for plausibiliy). On the other hand, for the visually inspected (segmented masks were visually checked for plausibility) rectus femoris, the original architecture was slightly exceeded by the UNet 3+. Moving towards the vastus lateralis, the UNet 3+ closely followed by TransUNet outperformed the original architecture on the uninspected data. For the visually inspected data of the vastus lateralis, the UNet 3+ and the TransUNet outperformed the other models.
    Although this is a rather untypical approach, the model training data was quite similar and not statistically different. Thus, we provide all three models and compared them to manual analysis (Fig. 1-4). Users are adivsed to select the performing models for VL and RF. However, we still advise to test the selected model prior to usage. 
    All models were trained using K-fold (5 folds used) cross-validation to ensure that overfitting is reduced. 

    .. figure:: ..\\gui_files\\RF_without_inspection.png
        :scale: 50 %
        :alt: inspect_figure

        Figure 1: Comparison of RF images to manual analysis with uninspected results.

    .. figure:: ..\\gui_files\\RF_with_inspection.png
        :scale: 50 %
        :alt: inspect_figure

        Figure 2: Comparison of RF images to manual analysis with inspected results.

    .. figure:: ..\\gui_files\\VL_without_inspection.png
        :scale: 50 %
        :alt: inspect_figure

        Figure 2: Comparison of VL images to manual analysis with inspected results.

    .. figure:: ..\\gui_files\\VL_with_inspection.png
        :scale: 50 %
        :alt: inspect_figure

        Figure 2: Comparison of VL images to manual analysis with inspected results.
