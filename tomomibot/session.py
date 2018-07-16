import threading
import time

import numpy as np
import librosa

from tomomibot.audio import AudioIO, detect_onsets, slice_audio, mfcc_features


class Session():

    def __init__(self, ctx, voice, **kwargs):
        self.ctx = ctx

        self.samplerate = kwargs.get('samplerate', 44100)
        self.interval = kwargs.get('interval', 1)
        self.onset_threshold = kwargs.get('onset_threshold', 10)

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

        self._voice = voice

        self.ctx.log('Voice "{}" with {} points'
                     .format(voice.name, len(voice._pca_points)))
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

        # Detect onsets in available data
        onsets, _ = detect_onsets(frames,
                                  self.samplerate,
                                  self.onset_threshold)
        self.ctx.vlog('%i onsets detected' % len(onsets))

        # Slice audio into parts, only take long enough ones
        slices = slice_audio(frames, onsets)
        wavs = []

        for i in range(len(slices)):
            # Normalize slice audio signal
            y_slice = librosa.util.normalize(slices[i][0])

            # Calculate MFCCs
            mfcc = mfcc_features(y_slice, self.samplerate)

            # Project point into given voice PCA space
            point = self._voice.project(mfcc)

            # Find closest sound to this one
            wav = self._voice.find_wav(point)
            wavs.append(wav)

        self.ctx.vlog('%i slices generated' % len(slices))

        # @TODO Just play them for now, RNN model prediction later
        for wav in wavs:
            self.ctx.vlog('▶ Play .wav sample "{}"'.format(wav))
            self._audio.play(wav)

        self._audio.flush()
        self.ctx.vlog('')
