# ppg_visualization.py
"""
Module for visualizing Photoplethysmogram (PPG) signals.

This module provides functions for plotting PPG signals, offering options
to customize the appearance of the plot and save it to a file.

Functions:
- plot_ppg_signal: Plot PPG signal.

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm


def plot_ppg_signal(signal, fs=None, xlim=None, ylim=None, title=None, show_fig=True, file_path=None, figsize=(20, 8),
                    color='blue'):
    """
    Plot PPG signal.

    Parameters:
        - signal (numpy.ndarray): The PPG signal data.
        - fs (float, optional): Sampling frequency in Hz (default: None).
        - xlim (tuple, optional): x-axis limits for the plot (default: None).
        - ylim (tuple, optional): y-axis limits for the plot (default: None).
        - title (str, optional): Title of the plot (default: None).
        - show_fig (bool, optional): Whether to display the figure (default: True).
        - file_path (str, optional): File path to save the figure (default: None).
        - figsize (tuple, optional): Figure size (default: (20, 8)).
        - color (str, optional): Matplotlib line color (default: 'blue').

    Returns:
        None
    """

    if not isinstance(signal, np.ndarray):
        raise TypeError("`signal` must be a numpy array.")

    fig, ax = plt.subplots(figsize=figsize)

    if fs is not None:
        time = np.arange(0, len(signal) * 1 / fs, 1 / fs)
        ax.set_xlabel('Time [s]')
    else:
        time = np.arange(0, len(signal))

    if xlim:
        ax.set_xlim(*xlim)

    if ylim:
        ax.set_ylim(*ylim)

    ax.plot(time, signal, color=color)

    ax.set_ylabel('PPG Signal')
    if title:
        ax.set_title(title)

    if file_path:
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(file_path)
            print(f"Figure saved to {file_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")

    if show_fig:
        plt.show()
    plt.close()

    return


def plot_ppg_peaks_signal(signal, peaks, fs, xlim=None, show_fig=True, save_fig=None):
    """
    Plots the PPG signal with its corresponding peaks highlighted.

    Args:
        signal (np.ndarray): PPG signal (1D array).
        peaks (list): List of peak indices in the signal.
        fs (float): Sampling frequency of the signal.
        xlim (tuple, optional): A tuple of (min, max) values for the x-axis limits. Defaults to None.
        show_fig (bool, optional): Whether to display the plot on screen. Defaults to True.
        save_fig (str, optional): File path to save the plot. Defaults to None.
    """

    plt.figure(figsize=(12, 8))  # Set figure size

    # Calculate time axis
    time = np.arange(len(signal)) / fs

    # Plot signal and peaks
    plt.plot(time, signal, label="PPG Signal")
    plt.scatter(time[peaks], signal[peaks], label="PPG Peaks", color='red')

    # Set labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("PPG Amplitude")
    plt.title("PPG Signal with Peaks")

    # Add legend
    plt.legend()

    # Set x-axis limits if provided
    if xlim:
        plt.xlim(xlim[0], xlim[1])

    # Save the figure if a file path is provided
    if save_fig:
        # Ensure directory exists before saving
        save_dir = Path(save_fig).parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        plt.savefig(save_fig)

    # Display the plot if requested
    if show_fig:
        plt.show()

    # Close the figure to avoid memory leaks
    plt.close()


def plot_ppg_peaks_labels_signal(signal, peaks, labels, fs, xlim=None, ylim=(-10, 10), show_fig=True, save_fig=None):
    """
    Plots the PPG signal with its corresponding peaks and labels.

    Args:
        signal (np.ndarray): PPG signal (1D array).
        peaks (list): List of peak indices in the signal.
        labels (list): List of labels corresponding to each peak.
        fs (float): Sampling frequency of the signal.
        xlim (tuple, optional): A tuple of (min, max) values for the x-axis limits. Defaults to None.
        ylim (tuple, optional): A tuple of (min, max) values for the y-axis limits. Defaults to (-10, 10).
        show_fig (bool, optional): Whether to display the plot on screen. Defaults to True.
        save_fig (str, optional): File path to save the plot. Defaults to None.
    """

    plt.figure(figsize=(12, 8))  # Set figure size

    # Calculate time axis
    time = np.arange(len(signal)) / fs

    # Plot signal and scatter peaks with distinct colors
    plt.plot(time, signal, label="PPG Signal")
    cmap = cm.get_cmap('tab10')  # Use colormap for multiple peak labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    for i, label in enumerate(unique_labels):
        peak_indices = peaks[labels == label]
        plt.scatter(time[peak_indices], signal[peak_indices], label=label, c=cmap(i % len(cmap.colors)))

    # Add labels, title, and legend
    plt.xlabel("Time (seconds)")
    plt.ylabel("PPG Amplitude")
    plt.title("PPG Signal with Peaks and Labels")
    plt.legend()

    # Set axis limits if provided
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])

    # Save the figure if a file path is provided
    if save_fig:
        # Ensure directory exists before saving
        save_dir = Path(save_fig).parent
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        plt.savefig(save_fig)

    # Display the plot if requested
    if show_fig:
        plt.show()

    # Close the figure to avoid memory leaks
    plt.close()
