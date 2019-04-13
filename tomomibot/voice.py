import json
import os

import numpy as np

from tomomibot.audio import pca
from tomomibot.const import GENERATED_FOLDER, ONSET_FILE
from tomomibot.utils import make_wav_path


KERNEL_SIZE = 8


class Voice:

    def __init__(self, name, **kwargs):
        self.name = name

        # Load data file from voice
        onset_path = os.path.join(GENERATED_FOLDER, name, ONSET_FILE)
        with open(onset_path) as file:
            data = json.load(file)

            # Extract informations from data
            self.mfccs = [wav['mfcc'] for wav in data]
            self.wavs = [make_wav_path(name, wav['id']) for wav in data]
            self.positions = [[wav['start'], wav['end']] for wav in data]

            # Do we have information about the sample volumes?
            if 'rms' in data[0]:
                rms_data = [wav['rms'] for wav in data]

                # Calculate average volume
                kernel = np.array(np.full((KERNEL_SIZE,), 1)) / KERNEL_SIZE
                rms_avg_data = np.convolve(rms_data, kernel, 'same')

                # Normalize rms values and store them
                self.rms = rms_data / np.max(rms_data)
                self.rms_avg = rms_avg_data / np.max(rms_avg_data)

            self.fit()

    def fit(self, reference_voice=None):
        if reference_voice is None:
            reference_voice = self

        # Calculate PCA
        _, pca_instance, pca_scaler = pca(reference_voice.mfccs)
        self.points = pca_instance.transform(self.mfccs)
        self._pca_instance = pca_instance
        self._pca_scaler = pca_scaler

    def project(self, vectors):
        """Project a new mfcc vector into given PCA space"""
        points = self._pca_instance.transform(vectors)
        return self._pca_scaler.transform(points)

    def find_wav(self, point):
        """Find closest point and return its wav file path"""
        deltas = self.points - point
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        index = np.argmin(dist_2)
        return self.wavs[index]
