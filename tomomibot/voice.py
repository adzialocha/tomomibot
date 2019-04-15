import json
import os

import numpy as np

from tomomibot.audio import pca
from tomomibot.const import GENERATED_FOLDER, ONSET_FILE
from tomomibot.utils import make_wav_path


class Voice:

    def __init__(self, name, **kwargs):
        self.name = name

        # Load data file from voice
        onset_path = os.path.join(GENERATED_FOLDER, name, ONSET_FILE)
        with open(onset_path) as file:
            data = json.load(file)

            if 'version' in data:
                self.version = data['version']
            else:
                self.version = 1  # Add version for legacy releases

            # Extract informations from data
            if self.version == 1:
                self.sequence = data
                self.meta = {}
            elif self.version == 2:
                self.sequence = data['sequence']
                self.meta = data['meta']

            self.wavs = [
                make_wav_path(name, wav['id']) for wav in self.sequence]

            self.fit()

    def fit(self, reference_voice=None):
        if reference_voice is None:
            reference_voice = self

        # Get MFCC data from voice sequence
        mfccs = [wav['mfcc'] for wav in reference_voice.sequence]

        # Calculate PCA
        _, pca_instance, pca_scaler = pca(mfccs)
        self.points = pca_instance.transform(mfccs)
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
