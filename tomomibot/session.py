import threading
import time

import numpy as np

from tomomibot.audio import AudioIO, detect_onsets


class Session():

    def __init__(self, ctx, **kwargs):
        self.ctx = ctx

        self.sample_rate = kwargs.get('sample_rate', 44100)
        self.interval = kwargs.get('interval', 3)
        self.onset_threshold = kwargs.get('onset_threshold', 10)

        try:
            self._audio = AudioIO(ctx,
                                  device_in=kwargs.get('input_device', 0),
                                  device_out=kwargs.get('output_device', 0),
                                  channel_in=kwargs.get('input_channel', 0),
                                  channel_out=kwargs.get('output_channel', 0))
        except IndexError as err:
            self.ctx.elog(err)

        self._thread = threading.Thread(target=self.run, args=())
        self._thread.daemon = True

        self.is_running = False

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
                                  self.sample_rate,
                                  self.onset_threshold)
        self.ctx.vlog('%i onsets detected' % len(onsets))
