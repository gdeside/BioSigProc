# common_visualization.py
"""
Module for common signal visualization.

This module provides functions for plotting various signals, offering options
to customize the appearance of the plot and save it to a file.

Functions:
- plot_signal: Plot a signal.
- plot_fft: Plot the Fourier Transform of a signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import freqz


def plot_signal(signal, fs=None, xlim=None, ylim=None, title=None, show_fig=True, file_path=None, figsize=(20, 8),
                color='blue'):
    """
    Plot signal.

    Parameters:
        - signal (numpy.ndarray): The signal data.
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


def plot_fft(signal, fs, xlim=None, ylim=None, title=None, show_fig=True, file_path=None, figsize=(20, 8),
             color='blue'):
    """
    Plot the Fourier Transform of a signal.

    Parameters:
        - signal (numpy.ndarray): The input signal.
        - fs (float): Sampling frequency.
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

    # Compute the Fourier Transform of the signal
    signal_fft = np.fft.fftshift(np.fft.fft(signal))
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(len(signal), d=1 / fs))

    # Check if the input signal is a numpy array
    if not isinstance(signal, np.ndarray):
        raise TypeError("`signal` must be a numpy array.")

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the Fourier Transform
    ax.plot(frequency_axis[len(frequency_axis) // 2:], np.abs(signal_fft[len(frequency_axis) // 2:]), color=color)

    # Set x-axis limits if provided
    if xlim:
        ax.set_xlim(*xlim)

    # Set y-axis limits if provided
    if ylim:
        ax.set_ylim(*ylim)

    # Set x and y axis labels
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Amplitude')

    # Set plot title if provided
    if title:
        ax.set_title(title)

    # Save the figure to a file if file_path is provided
    if file_path:
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(file_path)
            print(f"Figure saved to {file_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")

    # Display the plot if show_fig is True
    if show_fig:
        plt.show()
    plt.close()

    return


def plot_signals_length(signals, fs, title=None, show_fig=True, save_fig=None):
    """
    Plots the distribution of signal lengths in seconds.

    Args:
        signals (list): A list of EEG signals (each element is a 1D array).
        fs (float): Sampling frequency of the signals.
        title (str, optional): Title for the plot. Defaults to None.
        show_fig (bool, optional): Whether to display the plot on screen. Defaults to True.
        save_fig (str, optional): File path to save the plot. Defaults to None.
    """

    plt.figure(figsize=(12, 8))  # Set figure size

    # Calculate signal lengths in seconds
    signal_lengths = [len(signal) / fs for signal in signals]

    # Plot the histogram
    plt.hist(signal_lengths)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Number of signals")

    if title:
        plt.title(title)

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


def plot_frequency_response(b, a, fs, show_fig=True, save_fig=None):
    """
    Plots the magnitude and phase response of a filter.

    Args:
        b (np.ndarray): Numerator coefficients of the filter.
        a (np.ndarray): Denominator coefficients of the filter.
        fs (float): Sampling frequency.
        show_fig (bool, optional): Whether to display the plot on screen. Defaults to True.
        save_fig (str, optional): File path to save the plot. Defaults to None.
    """

    plt.figure(figsize=(8, 6))  # Set figure size

    # Calculate frequency response using SciPy
    freq, H = freqz(b, a, fs=fs)

    # Plot magnitude (dB)
    plt.subplot(2, 1, 1)
    plt.semilogx(freq, 20 * np.log10(abs(H)), label="Magnitude (dB)")
    plt.title("Frequency Response")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, fs / 2)  # Set appropriate x-axis limits
    plt.grid(True)

    # Plot phase (degrees)
    plt.subplot(2, 1, 2)
    plt.semilogx(freq, np.unwrap(np.angle(H)) * 180 / np.pi, label="Phase (degrees)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (degrees)")
    plt.xlim(0, fs / 2)  # Set appropriate x-axis limits
    plt.yticks([-90, -60, -30, 0, 30, 60, 90])
    plt.ylim([-90, 90])
    plt.grid(True)

    plt.legend()  # Add legend

    # Save the figure if a file path is provided
    if save_fig:
        # Ensure file extension is included
        if not save_fig.endswith(".png"):
            save_fig += ".png"
        plt.savefig(save_fig)

    # Display the plot if requested
    if show_fig:
        plt.show()

    # Close the figure to avoid memory leaks
    plt.close()


def plot_dft(signals, fft, fs, legend=None, show_fig=True, save_fig=None):
    """
    Plots the time-domain signals and their corresponding frequency spectra (DFT).

    Args:
        signals (np.ndarray): Array of time-domain signals (shape: (n_channels, n_samples)).
        fft (np.ndarray): Array of frequency-domain signals (DFT) after FFT (shape: (n_channels, n_frequencies)).
        fs (float): Sampling frequency of the signals.
        legend (list, optional): List of labels for the frequency channels (length should match n_channels-1). Defaults to None.
        show_fig (bool, optional): Whether to display the plot on screen. Defaults to True.
        save_fig (str, optional): File path to save the plot. Defaults to None.
    """

    plt.figure(figsize=(12, 6))  # Set figure size

    # Calculate time axis for signals
    t = np.arange(signals.shape[1]) / fs

    # Plot time-domain signals
    ax1 = plt.subplot(211)
    for i, color in enumerate(plt.cm.tab10.colors):
        if i == 0:
            label = "Reference"  # Assuming the first channel is reference
        else:
            label = legend[i - 1] if legend else f"Channel {i}"
        ax1.plot(t, signals[i], label=label, color=color)
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()

    # Calculate frequency axis for DFT
    f = np.linspace(0, fs / 2, int(fft.shape[1] / 2) + 1)

    # Plot frequency-domain signals (half spectrum due to real signals)
    ax2 = plt.subplot(212)
    for i, color, label in zip(range(1, len(legend) + 1), plt.cm.tab10.colors, legend):
        ax2.plot(f, np.abs(fft[i, :len(f)]), label=label, color=color)  # Plot only positive frequencies
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (absolute)")
    ax2.set_xlim(0, fs / 2)  # Set x-axis limit to half sampling frequency
    ax2.legend()

    # Improve readability (optional)
    plt.tight_layout()

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
