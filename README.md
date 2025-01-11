# SPOTER Sign Language Recognition

This project implements a sign language recognition system using the SPOTER (Sign POse-based TransformER) architecture. The system processes skeletal data extracted from video recordings of sign language and uses a transformer-based model to classify the signs.

## Project Structure

*   `spoter.ipynb`: Jupyter Notebook containing the complete implementation of the sign language recognition system.
*   `out-checkpoints/`: Directory where model checkpoints are saved during training.
*   `out-img/`: Directory where training and validation plots are saved.
*   `spoter.csv`: CSV file containing the preprocessed data.

## Overview

The `spoter.ipynb` notebook performs the following steps:

1.  **Imports Libraries:** Imports necessary libraries such as `numpy`, `pandas`, `os`, `cv2`, `mediapipe`, `torch`, `matplotlib`, and `sklearn`.
2.  **Data Loading and Preprocessing:**
    *   Loads data from a CSV file (`spoter.csv`).
    *   Defines constants for body and hand landmarks.
    *   Includes functions for extracting pose data from videos (commented out).
    *   Includes functions for data cleaning and label mapping.
3.  **Data Normalization:**
    *   Implements the Bohacek-normalization algorithm for normalizing hand and body pose data.
    *   Includes functions for normalizing both full dataframes and single data dictionaries.
4.  **Data Augmentation:**
    *   Implements various data augmentation techniques, including rotation, shearing, and arm joint rotation.
    *   Includes functions for applying Gaussian noise to the data.
5.  **Dataset Creation:**
    *   Defines a custom dataset class `CzechSLRDataset` for loading and processing the data.
    *   Includes functions for converting between tensors and dictionaries.
6.  **Model Architecture:**
    *   Implements the SPOTER model architecture, which is a transformer-based model for sign language recognition.
    *   Includes a custom `SPOTERTransformerDecoderLayer` that omits the self-attention operation.
7.  **Training and Evaluation:**
    *   Includes functions for training and evaluating the model.
    *   Includes functions for calculating training loss, accuracy, and top-k accuracy.
8.  **Training Loop:**
    *   Sets up the training parameters, including learning rate, number of epochs, and batch size.
    *   Trains the model using the training data and evaluates it on the validation data.
    *   Saves model checkpoints during training.
9.  **Plotting Results:**
    *   Includes functions for plotting training and validation metrics, as well as learning rate progress.

## How to Use

1.  **Set up the environment:** Install the required libraries using `pip install -r requirements.txt` (located in the same directory as the notebook).
2.  **Prepare the data:** Ensure that the `spoter.csv` file is in the same directory as the notebook.
3.  **Run the notebook:** Execute the `spoter.ipynb` notebook to train and evaluate the SPOTER model.
4.  **View results:** The training and validation plots will be saved in the `out-img/` directory, and model checkpoints will be saved in the `out-checkpoints/` directory.

## Requirements

*   Python 3.6+
*   PyTorch
*   Pandas
*   Numpy
*   OpenCV
*   Mediapipe
*   Scikit-learn
*   Matplotlib
*   Torchvision

## Notes

*   The paths to the data and model checkpoints may need to be adjusted based on your local setup.
*   The training parameters can be adjusted to improve the model's performance.
*   The notebook includes various data augmentation techniques that can be enabled or disabled.
*   The notebook includes functions for plotting training and validation metrics, which can be used to visualize the model's performance.

This project provides a starting point for building a sign language recognition system using the SPOTER architecture.
