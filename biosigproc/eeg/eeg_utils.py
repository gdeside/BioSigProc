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
from scipy.signal import welch


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


def extract_eeg_features(eeg_signals, fs, noverlap=0, features_to_extract=None):
    """
    Extracts features from EEG signals within specified frequency bands.

    Args:
        eeg_signals (np.ndarray): Input EEG signals (n_samples, n_channels).
        fs (float): Sampling frequency of the EEG signals.
        noverlap (int, optional): The number of overlapping samples between segments. Defaults to 0.
        features_to_extract (list, optional): A list of feature names to extract.
            Defaults to None (all features will be extracted).

    Returns:
        np.ndarray: Feature matrix (n_samples, n_features).
    """

    frequency_bands = [
        ("Delta", (0.5, 4)),
        ("Theta", (4, 8)),
        ("Alpha", (8, 13)),
        ("Low Beta", (13, 21)),
        ("High Beta", (21, 28)),
        ("Beta", (13, 28)),  # Combining Low Beta and High Beta
        ("Low Gamma", (28, 38)),
        ("Gamma", (28, 48.5)),  # Combining Low Gamma and High Gamma
    ]

    # List of implemented features
    feature_list = [
        ("mean", np.nanmean),
        ("median", np.nanmedian),
        ("std", np.nanstd),
        ("max-min", lambda x: np.ptp(x)),
        ("end-start", lambda x: x[-1] - x[0]),
        ("abs_slope", lambda x: np.abs(np.gradient(x)).mean()),
        ("percentile_diff", lambda x: np.percentile(x, 95) - np.percentile(x, 5)),
        ("num_greater_than_previous", lambda x: np.sum(np.diff(x) > 0)),
        ("binned_entropy", _binned_entropy),
        ("number_mean_crossing", _number_mean_crossing),
        ("ratio_beyond_r_sigma", _ratio_beyond_r_sigma),
        ("rvalue", _rvalue),
        ("intercept", _intercept),
        ("stderr", _stderr),
    ]

    # Define the parameters for the Welch method
    nperseg = int(fs)  # Window size (in case of 1 second)

    # Extract features
    feature_matrix = []
    for eeg_signal in eeg_signals:
        feature_vector = []
        for channel_data in eeg_signal:
            channel_features = []

            # Compute power spectral density (PSD)
            f, psd = welch(channel_data, fs=fs, nperseg=nperseg, noverlap=noverlap)

            for band_name, freq_band in frequency_bands:
                # Extract PSD in the band
                freq_band_psd = psd[(f >= freq_band[0]) & (f <= freq_band[1])]

                # Calculate features based on selected features
                selected_features = feature_list if features_to_extract is None else [
                    f for f in feature_list if f[0] in features_to_extract
                ]
                for feature_name, feature_function in selected_features:
                    feature_value = feature_function(freq_band_psd)
                    channel_features.append(feature_value)

                    # Check and handle NaN values (optional)
                    # if np.isnan(feature_value):  # Uncomment to handle NaN values
                    #     print(f"Warning: Feature value is NaN for feature '{feature_name}'")

            feature_vector += channel_features
        feature_matrix.append(feature_vector)

    return np.stack(feature_matrix)
