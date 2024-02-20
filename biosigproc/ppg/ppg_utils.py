# ppg_utils.py
"""
Module for PPG signal processing.

This module provides functions for processing PPG signals.

Functions:
- load_random_ppg_signal
-
"""
import os
import numpy as np
import random


def find_files_with_extension(folder_path, extension=".csv"):
    """
    Find all files in the specified folder with a given file extension.

    Parameters:
    - folder_path (str): The path to the folder.
    - extension (str): The file extension to search for (default: ".csv").

    Returns:
    - List[str]: A list of file paths matching the criteria.
    """
    matching_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                matching_files.append(os.path.join(root, file))
    return matching_files


def load_random_ppg_signal(seed=None):
    """
    Loads a random PPG signal from a specified folder.

    Args:
        seed (int, optional): Seed for the random number generator.
            If provided, ensures reproducibility in selecting the random file.
            Defaults to None.

    Returns:
        numpy.ndarray: The loaded PPG signal.
    """

    if seed:
        # Set the seed for the random number generator
        random.seed(seed)

    # Find all files with the specified extension in the folder
    ppg_files = find_files_with_extension("BioSigProc/biosigproc/ppg/Data", extension=".npy")

    # Choose a random file from the list of available files
    ppg_file = random.choice(ppg_files)

    # Load the PPG signal from the chosen file using numpy.load
    ppg_signal = np.load(ppg_file)

    return ppg_signal

