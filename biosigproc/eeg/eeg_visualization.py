# eeg_visualization.py
"""
Module for eeg signal visualization.

This module provides functions for plotting eeg signals, offering options
to customize the appearance of the plot and save it to a file.

Functions:

"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_eeg(eeg_signals, fs, label, show_fig=True, file_path=None):
    """
    Plot EEG signals with specified labels.

    Parameters:
        eeg_signals (numpy.ndarray): Array containing EEG signals.
        fs (float): Sampling frequency.
        label (list): List of labels for each EEG channel.
        show_fig (bool, optional): Whether to display the figure (default: True).
        file_path (str, optional): File path to save the figure (default: None).

    Returns:
        None
    """
    # Same y scale for all channels
    bottom = np.amin(eeg_signals[0:eeg_signals.shape[0]])
    top = np.amax(eeg_signals[0:eeg_signals.shape[0]])

    # One big figure to frame the whole
    fig = plt.figure(figsize=(12, 8))
    ax0 = fig.add_subplot(111)
    plt.subplots_adjust(hspace=0.5)  # Adjust space between subplots
    ax0.tick_params(labelcolor='black', top=False, bottom=False, left=False, right=False)

    time = np.arange(0, eeg_signals.shape[1] * 1 / fs, 1 / fs)

    # Plot each channel
    for idx in range(0, eeg_signals.shape[0]):
        if idx == 0:
            _ax = fig.add_subplot(eeg_signals.shape[0], 1, idx + 1)
            ax = _ax
        else:
            ax = fig.add_subplot(eeg_signals.shape[0], 1, idx + 1, sharex=_ax)
        if idx == eeg_signals.shape[0] - 1:
            ax.tick_params(labelcolor='black', top=False, bottom=True, left=False, right=False)
            ax.patch.set_alpha(0)
            ax.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Time (sec)')
        else:
            ax.axis('off')

        ax.plot(time, eeg_signals[idx], linewidth=0.5)
        ax.set_ylim(bottom, top)
        ax.text(0, np.mean(ax.get_ylim()), label[idx], fontsize=8)  # Add label text

    ax0.get_yaxis().set_visible(False)
    ax0.get_xaxis().set_visible(False)

    # Save file
    if file_path:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)

    # Display graph on screen
    if show_fig:
        plt.show()
    plt.close()

    return


def plot_average_eeg_signals(eeg_signals, fs=None, plot_confidence_interval=False, show_fig=True, file_path=None):
    """
    Plots the average EEG signal, optionally with a confidence interval.

    Args:
        eeg_signals (np.ndarray): The input EEG signals with shape (n_electrodes, n_samples).
        fs (float, optional): The sampling frequency of the EEG signals.
        plot_confidence_interval (bool, optional): Whether to plot a confidence interval. Defaults to False.
        show_fig (bool, optional): Whether to display the plot. Defaults to True.
        file_path (str, optional): Path to save the plot as a file. Defaults to None.
    """

    if fs:
        time = np.arange(0, eeg_signals.shape[1] * 1 / fs, 1 / fs)
    else:
        time = np.arange(0, eeg_signals.shape[1])

    if not isinstance(eeg_signals, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if len(eeg_signals.shape) != 2:
        raise ValueError("Input must have 2 dimensions (channels, samples).")

        # Detrending option (optional)
    if np.any(np.isnan(eeg_signals)):
        print("Warning: Input contains NaN values. Detrending might be ineffective.")

        # Perform averaging along the specified axis
    average_eeg = np.mean(eeg_signals, axis=0)
    std_eeg = np.std(eeg_signals, axis=0)
    fig, ax = plt.subplots()
    # Plot the average EEG and confidence interval with a legend

    ax.plot(time, average_eeg, color="blue", label='Average EEG')
    if plot_confidence_interval:
        color = 'red'  # Set a consistent color for confidence interval
        ax.plot(time, average_eeg + 1.96 * std_eeg, color=color, linestyle='--', label='Confidence Interval')
        ax.plot(time, average_eeg - 1.96 * std_eeg, color=color, linestyle='--')

    # Add labels and legend
    ax.set_xlabel('Time (s)' if fs else 'Samples')
    ax.set_ylabel('Average EEG Amplitude')
    ax.legend()  # Add the legend

    # Save file
    if file_path:
        if not os.path.exists(Path(file_path).parent):
            os.makedirs(Path(file_path).parent)
        plt.savefig(file_path)

    # Display graph on screen
    if show_fig:
        plt.show()
    plt.close()

    return
