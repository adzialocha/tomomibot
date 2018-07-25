import os
import random
import threading
import time

from keras.models import load_model
from sklearn.cluster import KMeans
import librosa
import numpy as np
import tensorflow as tf

from tomomibot.audio import (AudioIO, slice_audio, detect_onsets,
                             is_silent, mfcc_features, get_db)
from tomomibot.const import MODELS_FOLDER
from tomomibot.train import reweight_distribution
from tomomibot.utils import plot_position


CHECK_WAV_INTERVAL = 0.1        # Check .wav queue interval (in seconds)
MAX_DENSITY_ONSETS = 10         # How many offsets for max density
PLAY_DELAY_EXP = 5              # Exponent for maximum density delay
RESET_PROPABILITY = 0.1         # Percentage of chance for resetting sequence


class Session():

    def __init__(self, ctx, voice, model, **kwargs):
        self.ctx = ctx

        self.interval = kwargs.get('interval')
        self.num_classes = kwargs.get('num_classes')
        self.samplerate = kwargs.get('samplerate')
        self.seq_len = kwargs.get('seq_len')
        self.temperature = kwargs.get('temperature')
        self.threshold_db = kwargs.get('threshold')

        try:
            self._audio = AudioIO(ctx,
                                  samplerate=self.samplerate,
                                  device_in=kwargs.get('input_device'),
                                  device_out=kwargs.get('output_device'),
                                  channel_in=kwargs.get('input_channel'),
                                  channel_out=kwargs.get('output_channel'))
        except IndexError as err:
            self.ctx.elog(err)

        self.ctx.log('Loading ..')

        # Prepare parallel tasks
        self._thread = threading.Thread(target=self.run, args=())
        self._thread.daemon = True
        self._play_thread = threading.Thread(target=self.play, args=())
        self._play_thread.daemon = True

        # Prepare playing logic
        self._sequence = []
        self._wavs = []
        self._density = 0.0

        self.is_running = False

        # Load model & make it ready for being used in another thread
        model_name = '{}.h5'.format(model)
        model_path = os.path.join(os.getcwd(), MODELS_FOLDER, model_name)

        self._model = load_model(model_path)
        self._model._make_predict_function()
        self._graph = tf.get_default_graph()

        # Prepare voice and k-means clustering
        self._voice = voice
        self._kmeans = KMeans(n_clusters=self.num_classes)
        self._kmeans.fit(self._voice.points)

        # Get the classes of the voice sound material / points
        point_classes = self._kmeans.predict(self._voice.points)
        self._point_classes = []
        for idx in range(self.num_classes):
            indices = np.where(point_classes == idx)
            self._point_classes.append(indices[0])

        self.ctx.log('Voice "{}" with {} samples'
                     .format(voice.name, len(voice.points)))

    def start(self):
        self.is_running = True

        # Start reading audio signal _input
        self._audio.start()

        # Start threads
        self._thread.start()
        self._play_thread.start()

        self.ctx.log('Ready!\n')

    def stop(self):
        self._audio.stop()
        self.is_running = False

    def run(self):
        while self.is_running:
            time.sleep(self.interval)
            if self.is_running:
                self.tick()

    def play(self):
        while self.is_running:
            time.sleep(CHECK_WAV_INTERVAL)
            if not self.is_running:
                return

            if len(self._wavs) > 1:
                # Get next wav file to play from queue
                wav = self._wavs[0]

                self.ctx.vlog(
                    '▶ play .wav sample "{}" (queue={}, density={})'.format(
                        wav,
                        len(self._wavs),
                        self._density))

                # Delay playing the sample a little bit
                rdm = random.expovariate(PLAY_DELAY_EXP) * self._density
                time.sleep(rdm)

                # Play it!
                self._audio.play(wav)

                # Remove the played sample from our queue
                self._wavs = self._wavs[1:]

    def tick(self):
        """Main routine for live sessions"""

        # Read current frame buffer from input signal
        frames = np.array(self._audio.read_frames()).flatten()

        self.ctx.vlog('Read {} frames (volume={}dB)'.format(
            len(frames), np.max(get_db(frames))))

        # Detect onsets in available data
        onsets, _ = detect_onsets(frames,
                                  self.samplerate,
                                  self.threshold_db)

        # Set a density based on amount of onsets
        self._density = min(
            MAX_DENSITY_ONSETS, len(onsets)) / MAX_DENSITY_ONSETS

        # Slice audio into parts when possible
        slices = []
        if len(onsets) == 0 and not is_silent(frames, self.threshold_db):
            slices = [[frames, 0, 0]]
        else:
            slices = slice_audio(frames, onsets)

        self.ctx.vlog('{} onsets detected & {} slices generated'.format(
            len(onsets), len(slices)))

        # Analyze and categorize slices
        for y in slices:
            # Normalize slice audio signal
            y_slice = librosa.util.normalize(y[0])

            # Calculate MFCCs
            mfcc = mfcc_features(y_slice, self.samplerate)

            # Project point into given voice PCA space
            point = self._voice.project([mfcc])[0].flatten()

            # Predict k-means class from point
            point_class = self._kmeans.predict([point])[0]

            # Add it to our sequence queue
            self._sequence.append(point_class)

        # Check for too long sequences, cut it if necessary
        penalty = self.seq_len * 2
        if len(self._sequence) > penalty:
            self._sequence = self._sequence[penalty:]

        # Check if we already have enough data to do something
        if len(self._sequence) < self.seq_len:
            self.ctx.vlog('')
            return

        with self._graph.as_default():
            max_index = len(self._sequence)
            while True:
                # Play all possible subsequences
                min_index = max_index - self.seq_len
                if min_index < 0:
                    break
                sequence_slice = self._sequence[min_index:max_index]

                # Predict next action via model
                result = self._model.predict(np.array([sequence_slice]))

                # Reweight the softmax distribution
                result_reweighted = reweight_distribution(result,
                                                          self.temperature)
                result_class = np.argmax(result_reweighted)

                # Decode to a position in PCA space
                point_index = np.random.choice(
                    self._point_classes[result_class])

                if point_index:
                    # Find closest sound to this point
                    position = self._voice.points[point_index]
                    self._wavs.append(self._voice.find_wav(position))
                    if not self.ctx.verbose:
                        self.ctx.log(plot_position(position))

                max_index -= 1

        # Remove oldest event from sequence queue
        self._sequence = self._sequence[1:]

        if random.random() < RESET_PROPABILITY:
            self._sequence = []

        self.ctx.vlog('')
