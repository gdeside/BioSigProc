import unittest
import numpy as np
from egg_utils import reject_electrodes, average_eeg_signals, extract_eeg_features


class TestEggUtils(unittest.TestCase):

    def test_reject_electrodes_valid_input(self):
        """
        Tests reject_electrodes with valid input data and labels.
        """
        # Sample EEG signals and labels
        eeg_signals = np.random.rand(10, 100)
        labels = ["channel_" + str(i) for i in range(10)]
        electrodes_to_reject = [2, 5]

        # Call the function
        eeg_signals_new, labels_new = reject_electrodes(eeg_signals, electrodes_to_reject, labels)

        # Expected output shapes and values
        expected_shape = (8, 100)
        expected_labels = ["channel_" + str(i) for i in range(10) if i not in electrodes_to_reject]

        # Assert shapes and values
        self.assertEqual(eeg_signals_new.shape, expected_shape)
        self.assertEqual(labels_new, expected_labels)

    def test_reject_electrodes_invalid_electrodes(self):
        """
        Tests reject_electrodes with invalid electrodes (out of range).
        """
        eeg_signals = np.random.rand(10, 100)
        electrodes_to_reject = [12, 15]

        # Expect an error with invalid electrodes
        with self.assertRaises(ValueError):
            reject_electrodes(eeg_signals, electrodes_to_reject)

    def test_average_eeg_signals_valid_input(self):
        """
        Tests average_eeg_signals with valid input data.
        """
        eeg_signals = np.random.rand(5, 100)

        # Call the function
        average_eeg = average_eeg_signals(eeg_signals)

        # Expected output shape and dimensions
        expected_shape = (100,)

        # Assert shape
        self.assertEqual(average_eeg.shape, expected_shape)

    def test_average_eeg_signals_invalid_input(self):
        """
        Tests average_eeg_signals with invalid input data (not a 2D array).
        """
        invalid_input = np.random.rand(10)

        # Expect an error with invalid input
        with self.assertRaises(ValueError):
            average_eeg_signals(invalid_input)

    def test_extract_eeg_features_valid_input(self):
        """
        Tests extract_eeg_features with valid input data.
        """
        eeg_signals = np.random.rand(100, 10)
        fs = 250
        features_to_extract = ["mean", "std"]

        # Call the function
        feature_matrix = extract_eeg_features(eeg_signals, fs, features_to_extract=features_to_extract)

        # Expected output shape
        expected_shape = (100, len(features_to_extract))

        # Assert shape
        self.assertEqual(feature_matrix.shape, expected_shape)

    def test_extract_eeg_features_invalid_input(self):
        """
        Tests extract_eeg_features with invalid input data (not a 2D array).
        """
        invalid_input = np.random.rand(10)
        fs = 250

        # Expect an error with invalid input
        with self.assertRaises(ValueError):
            extract_eeg_features(invalid_input, fs)


if __name__ == "__main__":
    unittest.main()
