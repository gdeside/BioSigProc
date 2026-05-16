![BioSigProc Logo](logo.png)

# BioSigProc

[![PyPI version](https://img.shields.io/pypi/v/BioSigProc.svg)](https://pypi.org/project/BioSigProc/)
[![Python versions](https://img.shields.io/pypi/pyversions/BioSigProc.svg)](https://pypi.org/project/BioSigProc/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**BioSigProc** is a Python library for processing and analyzing biomedical signals. It provides tools for signal processing, feature extraction, noise simulation, wavelet analysis, and visualization for EEG, ECG, and PPG signals.

## Installation

```bash
pip install BioSigProc
```

**Dependencies:** `numpy`, `matplotlib`, `scipy`, `PyWavelets`

## Package Structure

```
biosigproc/
├── __init__.py              # plot_signal, plot_fft, add_noise, wavelet_signal
├── common_utils.py          # Noise addition, wavelet decomposition
├── common_visualization.py  # Generic signal plotting utilities
├── eeg/
│   ├── eeg_utils.py         # EEG processing and feature extraction
│   └── eeg_visualization.py # EEG plotting
├── ecg/
│   └── ecg_utils.py         # ECG QRS detection (Pan-Tompkins)
└── ppg/
    ├── ppg_utils.py         # PPG signal loading
    └── ppg_visualization.py # PPG plotting
```

## Quick Start

```python
import numpy as np
import biosigproc

# Generate a synthetic signal
fs = 256  # Hz
t = np.arange(0, 5, 1/fs)
signal = np.sin(2 * np.pi * 10 * t)

# Plot the signal
biosigproc.plot_signal(signal, fs=fs, title="10 Hz sine wave")

# Plot its frequency spectrum
biosigproc.plot_fft(signal, fs=fs, title="FFT")

# Add Gaussian noise
noisy = biosigproc.add_noise(signal, noise_type='gaussian', noise_params={'std': 0.1})
```

## API Reference

### Top-level (`biosigproc`)

#### Visualization

**`plot_signal(signal, fs=None, xlim=None, ylim=None, title=None, show_fig=True, file_path=None, figsize=(20,8), color='blue')`**
Plot a 1-D signal. If `fs` is provided the x-axis is in seconds, otherwise in samples.

**`plot_fft(signal, fs, xlim=None, ylim=None, title=None, show_fig=True, file_path=None, figsize=(20,8), color='blue')`**
Plot the one-sided magnitude spectrum of a signal.

#### Noise

**`add_noise(signal, noise_type='gaussian', noise_params=None)`**
Add noise to a signal. Supported `noise_type` values:

| `noise_type`       | `noise_params` keys              |
|--------------------|----------------------------------|
| `'gaussian'`       | `mean` (0), `std` (0.05)         |
| `'uniform'`        | `level` (0.1)                    |
| `'salt_and_pepper'`| `salt_prob` (0.01), `pepper_prob` (0.01) |
| `'poisson'`        | `noise_lam` (0.02)               |

Individual noise functions are also available directly:
- `add_gaussian_noise(signal, noise_mean=0, noise_std=0.05)`
- `add_uniform_noise(signal, noise_level=0.01)`
- `add_salt_and_pepper_noise(signal, salt_prob=0.001, pepper_prob=0.001)`
- `add_poisson_noise(signal, noise_lam=0.02)`

#### Wavelets

**`wavelet_signal(signal, wavelet='db4', level=4, fs=None)`**
Perform multi-level wavelet decomposition (wraps `pywt.wavedec`). Returns a list of coefficients `[cA_n, cD_n, ..., cD_1]`. Pass `fs` to print the frequency range of each sub-band.

```python
coeffs = biosigproc.wavelet_signal(signal, wavelet='db4', level=4, fs=256)
```

---

### EEG (`biosigproc.eeg`)

```python
from biosigproc.eeg import reject_electrodes, average_eeg_signals, extract_eeg_features
from biosigproc.eeg import plot_eeg, plot_average_eeg_signals
```

**`reject_electrodes(eeg_signals, electrodes, labels=None)`**
Remove channels by index from an `(n_electrodes, n_samples)` array.

**`average_eeg_signals(eeg_signals, axis=0)`**
Compute the mean across channels (or samples).

**`extract_eeg_features(eeg_signals, fs, noverlap=0, features_to_extract=None)`**
Extract spectral features per channel per frequency band (Delta, Theta, Alpha, Beta, Gamma). Available features: `mean`, `median`, `std`, `max-min`, `end-start`, `abs_slope`, `percentile_diff`, `num_greater_than_previous`, `binned_entropy`, `number_mean_crossing`, `ratio_beyond_r_sigma`, `rvalue`, `intercept`, `stderr`.

```python
from biosigproc.eeg import extract_eeg_features

# eeg_data shape: (n_epochs, n_channels, n_samples)
features = extract_eeg_features(eeg_data, fs=256)
```

**`plot_eeg(eeg_signals, fs, label, show_fig=True, file_path=None)`**
Plot multi-channel EEG with one sub-panel per channel.

**`plot_average_eeg_signals(eeg_signals, fs=None, plot_confidence_interval=False, show_fig=True, file_path=None)`**
Plot the mean signal across channels with an optional 95 % confidence interval band.

---

### ECG (`biosigproc.ecg`)

```python
from biosigproc.ecg import pan_tompkin
```

**`pan_tompkin(ecg, fs, show_fig=False)`**
Detect QRS complexes using the Pan-Tompkins algorithm. Returns an array of R-peak sample indices.

```python
import numpy as np
from biosigproc.ecg import pan_tompkin

# ecg: 1-D numpy array, fs: sampling frequency
r_peaks = pan_tompkin(ecg_signal, fs=360)
print("R-peaks at samples:", r_peaks)
```

---

### PPG (`biosigproc.ppg`)

```python
from biosigproc.ppg import load_random_ppg_signal, plot_ppg_signal
```

**`load_random_ppg_signal(seed=None)`**
Load a random example PPG recording from the bundled BIDMC dataset. Pass `seed` for reproducibility.

**`plot_ppg_signal(signal, fs=None, xlim=None, ylim=None, title=None, show_fig=True, file_path=None, figsize=(20,8), color='blue')`**
Plot a PPG signal.

```python
from biosigproc.ppg import load_random_ppg_signal, plot_ppg_signal

ppg = load_random_ppg_signal(seed=42)
plot_ppg_signal(ppg, fs=125, title="Example PPG")
```

Additional PPG visualization functions (importable from `biosigproc.ppg.ppg_visualization`):
- `plot_ppg_peaks_signal(signal, peaks, fs, ...)` — overlay detected peaks
- `plot_ppg_peaks_labels_signal(signal, peaks, labels, fs, ...)` — colour-coded peak labels

---

### Common Visualization (`biosigproc.common_visualization`)

Additional utilities importable directly from the module:

| Function | Description |
|---|---|
| `plot_signals_length(signals, fs, ...)` | Histogram of signal durations |
| `plot_frequency_response(b, a, fs, ...)` | Magnitude and phase response of a filter |
| `plot_dft(signals, fft, fs, ...)` | Side-by-side time-domain and frequency-domain plots |

---

## Full Example

```python
import numpy as np
import biosigproc
from biosigproc.eeg import plot_eeg, reject_electrodes

# Simulate 4-channel EEG at 256 Hz for 5 seconds
fs = 256
n_samples = fs * 5
n_channels = 4
eeg = np.random.randn(n_channels, n_samples) * 50  # µV

# Remove a noisy channel
eeg_clean = reject_electrodes(eeg, electrodes=[2])

# Add Gaussian noise to a single channel for testing
noisy_ch = biosigproc.add_noise(eeg_clean[0], noise_type='gaussian', noise_params={'std': 5})

# Wavelet decomposition
coeffs = biosigproc.wavelet_signal(eeg_clean[0], wavelet='db4', level=4, fs=fs)

# Plot
labels = [f"Ch {i}" for i in range(eeg_clean.shape[0])]
plot_eeg(eeg_clean, fs=fs, label=labels)
```

## Contributing

Contributions are welcome. Please open an issue or pull request on [GitHub](https://github.com/gdeside/BioSigProc).

## License

This project is licensed under the [Apache Software License 2.0](LICENSE).
