import json
import os

import numpy as np

from tomomibot.audio import pca
from tomomibot.const import GENERATED_FOLDER, ONSET_FILE
from tomomibot.utils import (make_wav_path,
                             encode_duration_class,
                             encode_dynamic_class)


def convert_positions(sequence):
    # We store frame indexes as strings since they are too large
    new_sequence = []
    for step in sequence:
        step['start'] = int(step['start'])
        step['end'] = int(step['end'])
        new_sequence.append(step)
    return new_sequence


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
                self.sequence = convert_positions(data)
                self.meta = {}
                self.rms_max = 1
            elif self.version == 2:
                self.sequence = convert_positions(data['sequence'])
                self.meta = data['meta']

                # Get RMS maximum for normalization
                self.rms_max = np.max([wav['rms'] for wav in self.sequence])

            # Prepare wav file informations for playback
            wavs = []
            for wav in self.sequence:
                wav_entry = {
                    'path': make_wav_path(name, wav['id']),
                }

                if self.version == 2:
                    sr = self.meta['samplerate']
                    duration = (wav['end'] - wav['start']) / sr * 1000

                    wav_entry['class_dynamic'] = encode_dynamic_class(
                        None, wav['rms'] / self.rms_max)
                    wav_entry['class_duration'] = encode_duration_class(
                        duration)

                wavs.append(wav_entry)

            self.wavs = np.array(wavs)

            self.fit()

    def fit(self, reference_voice=None):
        if reference_voice is None:
            reference_voice = self

        # Get MFCC data from voice sequence
        mfccs = [wav['mfcc'] for wav in reference_voice.sequence]

        # Calculate PCA
        n_components = 6 if self.version == 2 else 2
        _, pca_instance, pca_scaler = pca(mfccs, components=n_components)
        self.points = pca_instance.transform(mfccs)
        self._pca_instance = pca_instance
        self._pca_scaler = pca_scaler

    def project(self, vectors):
        """Project a new mfcc vector into given PCA space"""
        points = self._pca_instance.transform(vectors)
        return self._pca_scaler.transform(points)

    def find_wav(self, point_classes,
                 class_sound, class_dynamic, class_duration):
        """Return wav path depending on sound, duration and dynamic"""
        indices = point_classes[class_sound]
        possible_wavs = self.wavs[indices]

        if len(possible_wavs) == 0:
            return None

        if self.version == 2:
            # Filter by dynamic class
            if class_dynamic:
                possible_wavs = list(
                    filter(lambda x: x['class_dynamic'] == class_dynamic,
                           possible_wavs))

            # Filter by duration class
            if class_duration:
                possible_wavs = list(
                    filter(lambda x: x['class_duration'] == class_duration,
                           possible_wavs))

        # Fallback when we could not find anything
        if len(possible_wavs) == 0:
            possible_wavs = self.wavs[indices]

        # Pick a random sound from that group
        wav = np.random.choice(possible_wavs)

        return wav['path']
