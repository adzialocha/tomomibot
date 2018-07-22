import os
import random

import threading
import time

from keras.models import load_model
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf

from tomomibot.audio import AudioIO
from tomomibot.const import MODELS_FOLDER
from tomomibot.generate import analyze_sequence
from tomomibot.train import k_means_decode_data, reweight_distribution


def add_noise(point, factor=0.01):
    return np.array(point[0] + random.uniform(-factor, factor),
                    point[1] + random.uniform(-factor, factor))


class Session():

    def __init__(self, ctx, voice, model, **kwargs):
        self.ctx = ctx

        self.blocks_per_second = kwargs.get('blocks_per_second', 10)
        self.interval = kwargs.get('interval', 1)
        self.noise_factor = kwargs.get('noise_factor', 0.01)
        self.num_classes = kwargs.get('num_classes', 10)
        self.threshold = kwargs.get('threshold', 0.01)
        self.samplerate = kwargs.get('samplerate', 44100)
        self.seq_len = kwargs.get('seq_len', 10)
        self.temperature = kwargs.get('temperature', 1.0)

        try:
            self._audio = AudioIO(ctx,
                                  samplerate=self.samplerate,
                                  device_in=kwargs.get('input_device', 0),
                                  device_out=kwargs.get('output_device', 0),
                                  channel_in=kwargs.get('input_channel', 0),
                                  channel_out=kwargs.get('output_channel', 0))
        except IndexError as err:
            self.ctx.elog(err)

        self._thread = threading.Thread(target=self.run, args=())
        self._thread.daemon = True
        self.is_running = False

        # Load model & make it ready for being used in another thread
        model_name = '{}.h5'.format(model)
        model_path = os.path.join(os.getcwd(), MODELS_FOLDER, model_name)

        self._model = load_model(model_path)
        self._model._make_predict_function()
        self._graph = tf.get_default_graph()

        # Prepare voice and k-means clustering
        sequence = voice.project(voice.sequence)
        self._voice = voice
        self._kmeans = KMeans(n_clusters=self.num_classes)
        self._kmeans.fit(sequence)

        self.ctx.log('Voice "{}" with {} samples'
                     .format(voice.name, len(voice.points)))
        self.ctx.log('')

    def start(self):
        # Start reading audio signal _input
        self._audio.start()

        # Start thread
        self.is_running = True

        self._thread.start()

    def stop(self):
        self._audio.stop()

        self.is_running = False

    def run(self):
        while self.is_running:
            time.sleep(self.interval)
            if self.is_running:
                self.tick()

    def tick(self):
        # Read current frame buffer from input signal
        frames = np.array(self._audio.read_frames()).flatten()
        self.ctx.vlog('Read %i frames' % frames.shape)

        # Slice audio in smaller pieces and analyse MFCCs
        mfccs = analyze_sequence(frames,
                                 self.samplerate,
                                 self.blocks_per_second,
                                 threshold=self.threshold)

        # Project points into given voice PCA space
        points = self._voice.project(mfccs)

        # Encode sequence for trained model, take sample from end
        encoded = self._kmeans.predict(points)
        encoded = encoded[:self.seq_len]

        # Predict next action via model
        with self._graph.as_default():
            result = self._model.predict(np.array([encoded]))

            # Reweight the softmax distribution
            result_reweighted = reweight_distribution(result, self.temperature)
            result_class = np.argmax(result_reweighted)

            # Decode to a position in PCA space
            position = k_means_decode_data([[result_class]], self._kmeans)
            position = position.flatten()
            position = add_noise(position, self.noise_factor)
            self.ctx.vlog('Model predicted point {} in cluster {}'.format(
                position,
                result_class))

        # Find closest sound to this point
        wav = self._voice.find_wav(position)
        if wav is not None:
            self.ctx.vlog('▶ Play .wav sample "{}"'.format(wav))
            self._audio.play(wav)
            self._audio.flush()

        self.ctx.vlog('')
