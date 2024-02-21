# ecg_utils.py
"""
Module for ECG signal processing.

This module provides functions for processing ECG signals.

Functions:
-
-
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, filtfilt, find_peaks
from scipy.interpolate import interp1d


def pan_tompkin(ecg, fs, show_fig=False):
    """
    Perform Pan-Tompkins QRS detection algorithm on an ECG signal.

    Parameters:
        - ecg (numpy.ndarray): ECG signal.
        - fs (float): Sampling frequency.
        - show_fig (bool): Whether to display intermediate processing figures (default: False).

    Returns:
        - numpy.ndarray: Detected QRS peaks.
    """

    # Remove mean of the signal
    ecg -= np.mean(ecg)

    if show_fig:
        fig, axs = plt.subplots(7)
        axs[0].plot(ecg)

    ########################## Band Pass ########################################
    if fs == 200:
        # Low Pass Filter (12 Hz)
        Wn_low = 12 * 2 / fs
        N_low = 3
        b_low, a_low = butter(N_low, Wn_low, btype='low')
        ecg_l = filtfilt(b_low, a_low, ecg)
        ecg_l = ecg_l / max(np.abs(ecg_l))

        # High Pass Filter (5 Hz)
        Wn_high = 5 * 2 / fs
        N_high = 3
        b_high, a_high = butter(N_high, Wn_high, btype='high')
        ecg_h = filtfilt(b_high, a_high, ecg_l)
        ecg_h = ecg_h / max(np.abs(ecg_h))

        if show_fig:
            axs[1].plot(ecg_h)
    else:
        # Bandpass filter for other sampling frequencies (5-15 Hz)
        f1, f2 = 5, 15
        Wn_bp = [f1 * 2 / fs, f2 * 2 / fs]
        N_bp = 3
        b_bp, a_bp = butter(N_bp, Wn_bp, btype='band')
        ecg_h = filtfilt(b_bp, a_bp, ecg)
        ecg_h = ecg_h / max(np.abs(ecg_h))

        if show_fig:
            axs[1].plot(ecg_h)

    ############################### derivative filter ##########################

    if fs != 200:
        int_c = (5 - 1) / (fs * 1 / 40)
        b = interp1d(np.arange(1, 6), [1, 2, 0, -2, -1] * (1 / 8) * fs, kind='linear')(np.arange(1, int_c * 5 + 1))
    else:
        b = np.array([1, 2, 0, -2, -1]) * (1 / 8) * fs

    # Derivative filter
    ecg_d = filtfilt(b, 1, ecg_h)
    ecg_d = ecg_d / max(np.abs(ecg_d))

    if show_fig:
        axs[2].plot(ecg_d)

    ############################# Squaring #####################################

    ecg_s = ecg_d ** 2
    if show_fig:
        axs[3].plot(ecg_s)

    ############################# Moving average #####################################
    window_size = int(round(0.150 * fs))
    conv_kernel = np.ones(window_size) / window_size
    ecg_m = np.convolve(ecg_s, conv_kernel, mode='same')

    if show_fig:
        axs[4].plot(ecg_m)

    ############################## Find peaks ########################################
    # Define the minimum peak distance in samples
    min_peak_distance = round(0.25 * fs)

    # Find peaks
    pks, locs = find_peaks(ecg_m, distance=min_peak_distance)

    # Filter peaks based on a condition
    selected_indices = np.where(ecg_m[pks] > 0.7 * np.max(ecg_m))[0]
    pks = pks[selected_indices]

    if show_fig:
        axs[5].plot(ecg)
        axs[5].plot(pks, ecg[pks], "x")

    ############################## Correction peaks ########################################

    for i, pk in enumerate(pks):
        print(pk)
        idx = [max(0, pk - 20), min(pk + 20, len(ecg))]
        print(idx)
        new_pk = np.argmax(ecg[idx[0]:idx[1]])+idx[0]
        print(pk, new_pk)
        pks[i] = new_pk

    if show_fig:
        axs[6].plot(ecg)
        axs[6].plot(pks, ecg[pks], "x")

    if show_fig:
        plt.tight_layout()
        plt.show()

    return pks
