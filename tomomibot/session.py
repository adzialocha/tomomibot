import threading
import time

import numpy as np

from tomomibot.audio import AudioInput, detect_onsets


class Session():

    def __init__(self, ctx, options):
        self.ctx = ctx

        self.sample_rate = 44100
        self.interval = 5

        self.input = AudioInput()

        self.thread = threading.Thread(target=self.run, args=())
        self.thread.daemon = True
        self.is_running = False

    def start(self):
        # Start reading audio signal input
        self.input.start()

        self.is_running = True
        self.thread.start()

    def stop(self):
        self.input.stop()
        self.is_running = False

    def run(self):
        while self.is_running:
            time.sleep(self.interval)
            if self.is_running:
                self.tick()

    def tick(self):
        # Read current frame buffer from input signal
        frames = np.array(self.input.read_frames()).flatten()
        self.ctx.vlog('Read %i frames' % frames.shape)

        # Detect onsets in available data
        onsets = detect_onsets(frames, self.sample_rate)
        self.ctx.vlog('%i onsets detected' % len(onsets))
