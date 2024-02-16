# ppg_visualization.py
"""
Module for visualizing Photoplethysmogram (PPG) signals.

This module provides functions for plotting PPG signals, offering options
to customize the appearance of the plot and save it to a file.

Functions:
- plot_PPG_signal: Plot PPG signal.

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_ppg_signal(signal, fs, xlim=None, title='PPG Signal', show_fig=True, file_path=None):
    """
    Plot PPG signal.

    Parameters:
    - signal: PPG signal.
    - fs: Sampling frequency.
    - xlim: (Optional) x-axis limits for the plot (default: None).
    - title: Title of the plot (default: 'PPG Signal').
    - show_fig: Whether to display the figure (default: True).
    - file_path: File path to save the figure (default: None).

    Returns:
    None
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    time = np.arange(0, len(signal) * 1 / fs, 1 / fs)

    if xlim:
        ax.set_xlim(*xlim)

    ax.plot(time, signal)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('PPG Signal')
    ax.set_title(title)

    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path)
        print(f"Figure saved to {file_path}")

    if show_fig:
        plt.show()
    plt.close()

    return
