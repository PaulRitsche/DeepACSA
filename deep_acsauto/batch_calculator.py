#from apo_model import ApoModel
#from calibrate import calibrate_distance_efov
#from calibrate import calibrate_distance_manually
#from calibrate import calibrate_distance_static
#from echo_int import calculate_echo_int
from predict_muscle_area import get_list_of_files
from predict_muscle_area import import_image_efov, import_image
from predict_muscle_area import get_flip_flags_list
from predict_muscle_area import plot_image
from predict_muscle_area import calc_area
from predict_muscle_area import compile_save_results

import os
import glob
import pandas as pd
import numpy as np
from skimage.transform import resize
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use("ggplot")

class BatchCalculator: 

    def __init__(self): 

        self.is_running = False
        
    def calculate_batch_efov(rootpath: str, modelpath: str, depth: float,
                             muscle: str, is_running):
        """Calculates area predictions for batches of EFOV US images
            containing continous scaling line.

        Arguments:
            Path to root directory of images,
            path to model used for predictions,
            ultrasound scanning depth,
            analyzed muscle.
        """
        list_of_files = glob.glob(rootpath + '/**/*.tif', recursive=True)

        apo_model = ApoModel(modelpath)

        with PdfPages(rootpath + '/Analyzed_images.pdf') as pdf:

            dataframe = pd.DataFrame(columns=["File", "Muscle", "Area_cm²"])
            for imagepath in list_of_files:

                if is_running == True:

                    # load image
                    imported = import_image_efov(imagepath, muscle)
                    filename, img_copy, img, height, width = imported

                    # find length of the scalingline
                    scalingline_length = calibrate_distance_efov(imagepath, muscle)

                    # predict area
                    pred_apo_t, fig = apo_model.predict_t(img, width, height)
                    echo = calculate_echo_int(img_copy, pred_apo_t)
                    area = calc_area(depth, scalingline_length, pred_apo_t)

                    # append results to dataframe
                    dataframe = dataframe.append({"File": filename,
                                                  "Muscle": muscle,
                                                  "Area_cm²": area,
                                                  "Echo_intensity": echo},
                                                 ignore_index=True)

                    # save figures
                    pdf.savefig(fig)
                    plt.close(fig)

                else: 
                    break

            # save predicted area values
            compile_save_results(rootpath, dataframe)


    def calculate_batch(rootpath: str, flip_file_path: str, modelpath: str,
                        depth: float, spacing: int, muscle: str, scaling: str, 
                        is_running):
        """Calculates area predictions for batches of (EFOV) US images
            not containing a continous scaling line.

            Arguments:
                Path to root directory of images,
                path to txt file containing flipping information for images,
                path to model used for predictions,
                ultrasound scanning depth,
                distance between (vertical) scaling lines (mm),
                analyzed muscle,
                scaling type.
        """
        list_of_files = glob.glob(rootpath + '/**/*.tif', recursive=True)
        flip_flags = get_flip_flags_list(flip_file_path)

        apo_model = ApoModel(modelpath)
        dataframe = pd.DataFrame(columns=["File", "Muscle", "Area_cm²"])

        with PdfPages(rootpath + '/Analyzed_images.pdf') as pdf:

            if len(list_of_files) == len(flip_flags):

                for imagepath in list_of_files:
                    
                    if is_running == True:
                        # load image
                        flip = flip_flags.pop(0)
                        imported = import_image(imagepath, muscle)
                        filename, img, nonflipped_img, height, width = imported

                        if scaling == "Static":
                            calibrate_fn = calibrate_distance_static
                            # find length of the scaling line
                            scalingline_length = calibrate_fn(
                            nonflipped_img, spacing, depth, flip
                            )
                        else:
                            calibrate_fn = calibrate_distance_manually
                        
                        # predict area
                        pred_apo_t, fig = apo_model.predict_t(img, width, height)
                        echo = calculate_echo_int(nonflipped_img, pred_apo_t)
                        area = calc_area(depth, scalingline_length, pred_apo_t)

                        # append results to dataframe
                        dataframe = dataframe.append({"File": filename,
                                                      "Muscle": muscle,
                                                      "Area_cm²": area,
                                                      "Echo_intensity": echo},
                                                      ignore_index=True)

                        # save figures
                        pdf.savefig(fig)
                        plt.close(fig)

                    else: 
                        break

                # save predicted area results
                compile_save_results(rootpath, dataframe)

            else:
                print("Warning: number of flipFlags and images doesn\'t match! " +
                      "Calculations aborted.")

