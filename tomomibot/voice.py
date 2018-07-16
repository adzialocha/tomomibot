import json
import os

import numpy as np

from tomomibot.audio import pca
from tomomibot.const import GENERATED_FOLDER, DATA_FILE
from tomomibot.utils import make_wav_path


class Voice:

    def __init__(self, name, **kwargs):
        self.path = os.path.join(GENERATED_FOLDER, name, DATA_FILE)
        self.name = name

        # Load data file from voice
        with open(self.path) as file:
            data = json.load(file)

            # Extract informations from data
            self.mfccs = [wav['mfcc'] for wav in data]
            self.wavs = [make_wav_path(name, wav['id']) for wav in data]
            self.positions = [[wav['start'], wav['end']] for wav in data]

            # Calculate PCA
            pca_points, pca_instance, pca_scaler = pca(self.mfccs)
            self._pca_points = pca_points
            self._pca_instance = pca_instance
            self._pca_scaler = pca_scaler

    def project(self, mfcc):
        """Project a new mfcc vector into given PCA space"""
        point = self._pca_instance.transform([mfcc])
        return self._pca_scaler.transform(point)[0]

    def find_wav(self, point):
        """Find closest point and return its wav file path"""
        deltas = self._pca_points - point
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        index = np.argmin(dist_2)
        return self.wavs[index]

    def generate_sequence(self):
        # @TODO Implement this
        pass
