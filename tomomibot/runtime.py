import signal
import sys
import threading
import time

import click

from tomomibot import __version__
from tomomibot.session import Session
from tomomibot.voice import Voice


class Runtime:

    def __init__(self, ctx, voice_name, **kwargs):
        self.ctx = ctx

        self._display_welcome()

        voice = Voice(voice_name)
        self._session = Session(self.ctx, voice, **kwargs)
        self._thread = None

    def initialize(self):
        self._init_signal()
        self._session.start()

        # This is our main thread. Keep it alive!
        while self._session.is_running:
            time.sleep(1)

    def _display_welcome(self):
        self.ctx.log("""
▄▄▄▄▄      • ▌ ▄ ·.       • ▌ ▄ ·. ▪  ▄▄▄▄·      ▄▄▄▄▄
•██  ▪     ·██ ▐███▪▪     ·██ ▐███▪██ ▐█ ▀█▪▪    •██
 ▐█.▪ ▄█▀▄ ▐█ ▌▐▌▐█· ▄█▀▄ ▐█ ▌▐▌▐█·▐█·▐█▀▀█▄ ▄█▀▄ ▐█.▪
 ▐█▌·▐█▌.▐▌██ ██▌▐█▌▐█▌.▐▌██ ██▌▐█▌▐█▌██▄▪▐█▐█▌.▐▌▐█▌·
 ▀▀▀  ▀█▄▀▪▀▀  █▪▀▀▀ ▀█▄▀▪▀▀  █▪▀▀▀▀▀▀·▀▀▀▀  ▀█▄▀▪▀▀▀
        """)
        self.ctx.log('Version: %s' % __version__)
        self.ctx.log('Exit with [CTRL] + [C]\n')

    def _init_signal(self):
        if not sys.platform.startswith('win') and sys.stdin \
                and sys.stdin.isatty():
            signal.signal(signal.SIGINT, self._handle_sigint)
        signal.signal(signal.SIGTERM, self._signal_stop)

    def _handle_sigint(self, sig, frame):
        if self._thread and self._thread.isAlive():
            return
        self._thread = threading.Thread(target=self._confirm_exit)
        self._thread.daemon = True
        self._thread.start()

    def _confirm_exit(self):
        if click.confirm('Do you really want to stop this session?'):
            self._session.stop()
            self.ctx.log('Shutdown confirmed!')
            return

    def _signal_stop(self, sig, frame):
        self._session.stop()
