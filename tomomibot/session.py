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
from tomomibot.const import MODELS_FOLDER, SILENCE_CLASS
from tomomibot.train import reweight_distribution
from tomomibot.utils import (get_num_classes,
                             encode_duration_class,
                             encode_dynamic_class,
                             encode_feature_vector,
                             decode_classes)


CHECK_WAV_INTERVAL = 0.1        # Check .wav queue interval (in seconds)
MAX_DENSITY_ONSETS = 10         # How many offsets for max density
PLAY_DELAY_EXP = 5              # Exponent for maximum density delay
RESET_PROPABILITY = 0.1         # Percentage of chance for resetting sequence


class Session():

    def __init__(self, ctx, voice, model, reference_voice=None, **kwargs):
        self.ctx = ctx

        self.num_sound_classes = kwargs.get('num_classes')
        self.use_dynamics = kwargs.get('dynamics')
        self.use_durations = kwargs.get('durations')

        self.penalty = kwargs.get('penalty')
        self.samplerate = kwargs.get('samplerate')
        self.seq_len = kwargs.get('seq_len')
        self.threshold_db = kwargs.get('threshold')

        # These parameters can be changed during performance
        self._interval = kwargs.get('interval')
        self._temperature = kwargs.get('temperature')

        # Prepare audio I/O
        try:
            self._audio = AudioIO(ctx,
                                  samplerate=self.samplerate,
                                  device_in=kwargs.get('input_device'),
                                  device_out=kwargs.get('output_device'),
                                  channel_in=kwargs.get('input_channel'),
                                  channel_out=kwargs.get('output_channel'),
                                  volume=kwargs.get('volume'))
        except RuntimeError as err:
            self.ctx.elog(err)

        self.ctx.log('Loading ..')

        # Prepare concurrent threads
        self._thread = threading.Thread(target=self.run, args=())
        self._thread.daemon = True

        self._play_thread = threading.Thread(target=self.play, args=())
        self._play_thread.daemon = True

        self._lock = threading.Lock()

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

        # Calculate number of total classes
        num_classes = get_num_classes(self.num_sound_classes,
                                      self.use_dynamics,
                                      self.use_durations)

        num_model_classes = self._model.layers[-1].output_shape[1]
        if num_model_classes != num_classes:
            self.ctx.elog('The given model was trained with a different '
                          'amount of classes: given {}, but '
                          'should be {}.'.format(num_classes,
                                                 num_model_classes))

        # Prepare voice and k-means clustering
        if reference_voice is None:
            reference_voice = voice
        else:
            voice.fit(reference_voice)

        self._voice = voice
        self._kmeans = KMeans(n_clusters=self.num_sound_classes)
        self._kmeans.fit(reference_voice.points)

        # Get the classes of the voice sound material / points
        point_classes = self._kmeans.predict(self._voice.points)
        self._point_classes = []
        for idx in range(num_classes):
            indices = np.where(point_classes == idx)
            self._point_classes.append(indices[0])

        self.ctx.log('Voice "{}" with {} samples'
                     .format(voice.name, len(voice.points)))

    @property
    def master_volume(self):
        return self._audio.volume

    @master_volume.setter
    def master_volume(self, value):
        self._audio.volume = value

    @property
    def interval(self):
        return self._interval

    @interval.setter
    def interval(self, value):
        with self._lock:
            self._interval = value

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        with self._lock:
            self._temperature = value

    def reset_sequence(self):
        with self._lock:
            self._sequence = []

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
            time.sleep(self._interval)
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
                        os.path.basename(wav),
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

        if len(frames) == 0:
            return

        self.ctx.vlog('Read {0} frames (volume={1:.2f}dB)'.format(
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
            slices = slice_audio(frames, onsets, trim=False)

        self.ctx.vlog('{} onsets detected & {} slices generated'.format(
            len(onsets), len(slices)))

        # Analyze and categorize slices
        for y in slices:
            y_slice = y[0]

            # Calculate MFCCs
            try:
                mfcc = mfcc_features(y_slice, self.samplerate)
            except RuntimeError:
                self.ctx.vlog(
                    'Not enough sample data for MFCC analysis')
            else:
                # Calculate RMS
                rms_data = librosa.feature.rms(y=y_slice) / self._voice.rms_max
                rms = np.float32(np.max(rms_data)).item()

                # Project point into given voice PCA space
                point = self._voice.project([mfcc])[0].flatten()

                # Predict k-means class from point
                class_sound = self._kmeans.predict([point])[0]

                # Get dynamic class
                class_dynamic = encode_dynamic_class(class_sound, rms)

                # Get duration class
                duration = len(y_slice) / self.samplerate * 1000
                class_duration = encode_duration_class(duration)

                # Encode it!
                feature_vector = encode_feature_vector(self.num_sound_classes,
                                                       class_sound,
                                                       class_dynamic,
                                                       class_duration,
                                                       self.use_dynamics,
                                                       self.use_durations)

                # Add it to our sequence queue
                self._sequence.append(feature_vector)

        # Check for too long sequences, cut it if necessary
        penalty = self.seq_len * self.penalty
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
                                                          self._temperature)
                result_class = np.argmax(result_reweighted)

                # Decode class back into sub classes
                class_sound, class_dynamic, class_duration = decode_classes(
                    result_class,
                    self.num_sound_classes,
                    self.use_dynamics,
                    self.use_durations)

                # Do not do anything when this is silence ..
                if class_sound != SILENCE_CLASS:
                    # Find closest sound to this point
                    wav = self._voice.find_wav(self._point_classes,
                                               class_sound,
                                               class_dynamic,
                                               class_duration)

                    smiley = '☺' if wav else '☹'
                    self.ctx.vlog('{} find sound (class={}, '
                                  'dynamic={}, duration={})'.format(
                                      smiley, class_sound, class_dynamic,
                                      class_duration))

                    if wav:
                        self._wavs.append(wav)

                max_index -= 1

        # Remove oldest event from sequence queue
        self._sequence = self._sequence[1:]

        if random.random() < RESET_PROPABILITY:
            self._sequence = []

        self.ctx.vlog('')
