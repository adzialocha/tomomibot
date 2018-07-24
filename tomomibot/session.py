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


CHECK_WAV_INTERVAL = 0.1


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

        self.ctx.log('Loading ..\n')

        self._thread = threading.Thread(target=self.run, args=())
        self._thread.daemon = True
        self._wav_thread = threading.Thread(target=self.check_wavs, args=())
        self._wav_thread.daemon = True

        self._buffer = np.array([])
        self._sequence = []
        self._wavs = []
        self._density = 0

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
        self.ctx.log('')

    def start(self):
        # Start reading audio signal _input
        self._audio.start()

        # Start threads
        self.is_running = True
        self._thread.start()
        self._wav_thread.start()

        self.ctx.log('Ready!\n')

    def stop(self):
        self._audio.stop()
        self.is_running = False

    def run(self):
        while self.is_running:
            time.sleep(self.interval)
            if self.is_running:
                self.tick()

    def check_wavs(self):
        while self.is_running:
            time.sleep(CHECK_WAV_INTERVAL)

            if not self.is_running:
                return

            if len(self._wavs) > 1:
                # Play audio!
                wav = self._wavs[0]

                print('wav', len(self._wavs))
                print('density', self._density)

                self.ctx.vlog('▶ play .wav sample "{}"'.format(wav))

                rdm = random.expovariate(5) * self._density
                print('random', rdm)
                time.sleep(rdm)

                self._audio.play(wav)

                self._wavs = self._wavs[1:]

    def tick(self):
        """Main routine for live sessions"""

        # Read current frame buffer from input signal
        frames = np.array(self._audio.read_frames()).flatten()
        self._buffer = np.concatenate([self._buffer, frames])

        self.ctx.vlog('Read %i frames' % frames.shape)
        print('db', np.max(get_db(frames)))

        # Detect onsets in available data
        onsets, _ = detect_onsets(self._buffer,
                                  self.samplerate,
                                  self.threshold_db)

        self._density = min(10, len(onsets)) / 10
        self._audio.density = max(0.1, self._density)

        # Slice audio into parts when possible
        slices = []
        if len(onsets) == 0:
            if is_silent(self._buffer, self.threshold_db):
                self.ctx.vlog('No onsets detected ..')
            else:
                self.ctx.vlog('No onsets detected but some audio activity!')
                slices = [[self._buffer, 0, 0]]
        else:
            self.ctx.vlog('%i onsets detected' % len(onsets))
            slices = slice_audio(self._buffer, onsets)

        self.ctx.vlog('%i slices generated' % len(slices))

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

        # Reset buffer
        self._buffer = np.array([])

        # Check for too long sequences
        if len(self._sequence) > 10:
            self.ctx.vlog('Tomomibot! Too much babbeling! Lets cut it!')
            self._sequence = self._sequence[self.seq_len:]

        # Check if we already have enough data to do something
        if len(self._sequence) < self.seq_len:
            self.ctx.vlog('Sequence too short to play something ..\n')
            return

        print('Sequence: ', self._sequence)

        with self._graph.as_default():
            max_index = len(self._sequence)
            min_index = max_index - self.seq_len
            while True:
                if min_index < 0:
                    break

                sequence_slice = self._sequence[min_index:max_index]

                # Predict next action via model
                result = self._model.predict(
                    np.array([sequence_slice]))

                # Reweight the softmax distribution
                result_reweighted = reweight_distribution(result,
                                                          self.temperature)
                result_class = np.argmax(result_reweighted)

                # Decode to a position in PCA space
                point_index = np.random.choice(
                    self._point_classes[result_class])

                if point_index:
                    position = self._voice.points[point_index]
                    self.ctx.vlog(
                        'Model predicted point {} in cluster {}'.format(
                            position,
                            result_class))

                    # Find closest sound to this point
                    self._wavs.append(self._voice.find_wav(position))

                max_index -= 1
                min_index = max_index - self.seq_len

        # Remove oldest event from sequence queue
        self._sequence = self._sequence[1:]

        if random.random() < 0.1:
            print("boom!")
            self._sequence = []

        self.ctx.vlog('')
