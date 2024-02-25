# egg_utils.py
"""
Module for EEG signal processing.

This module provides functions for processing EEG signals.

Functions:
-
-
"""
import os
import numpy as np
import random


def reject_electrodes(eeg_signals, electrodes, labels=None):
    """
    Rejects specified electrodes from EEG signals and optionally their corresponding labels.

    Args:
        eeg_signals (numpy.ndarray): The input EEG signals with shape (n_electrodes, n_samples).
        electrodes (list): A list of electrode indices to reject.
        labels (list, optional): Labels for each electrode, if available. Defaults to None.

    Returns:
        tuple:
            - eeg_signals_new (numpy.ndarray): The new EEG signals with rejected electrodes removed.
            - labels_new (list, optional): The new labels corresponding to the retained electrodes, if labels were provided.
    """

    # Validate input
    if not isinstance(electrodes, list):
        raise ValueError("Electrodes must be provided as a list.")

    # Calculate the number of electrodes to keep
    nb_electrodes = eeg_signals.shape[0]
    nb_electrodes_new = nb_electrodes - len(electrodes)

    # Initialize the new arrays
    eeg_signals_new = np.zeros((nb_electrodes_new, eeg_signals.shape[1]))
    if labels is not None:
        labels_new = []

    # Iterate over electrodes and copy remaining ones
    i_new = 0
    removed_electrodes = []
    for i in range(nb_electrodes):
        if i not in electrodes:
            eeg_signals_new[i_new, :] = eeg_signals[i, :]
            if labels:
                labels_new.append(labels[i])
            i_new += 1
        else:
            removed_electrodes.append(i)

    # Print information about removed electrodes and new size
    print("Removed electrodes:", removed_electrodes)
    print("Original size:", nb_electrodes, "New size:", nb_electrodes_new)

    # Return the updated data with optional labels
    if labels:
        return eeg_signals_new, labels_new
    else:
        return eeg_signals_new


def average_eeg_signals(eeg_signals, axis=0):
    """
    Calculates the average of EEG signals along a specified axis.

    Args:
        eeg_signals (np.ndarray): The input EEG signals with shape (n_electrodes, n_samples).
        axis (int, optional): The axis along which to average. Defaults to 0 (averaging across channels).

    Returns:
        np.ndarray: The averaged EEG signal.
    """

    # Check input and dimensions
    if not isinstance(eeg_signals, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if len(eeg_signals.shape) != 2:
        raise ValueError("Input must have 2 dimensions (channels, samples).")

    # Detrending option (optional)
    if np.any(np.isnan(eeg_signals)):
        print("Warning: Input contains NaN values. Detrending might be ineffective.")

    # Perform averaging along the specified axis
    average_eeg = np.mean(eeg_signals, axis=axis)

    return average_eeg
