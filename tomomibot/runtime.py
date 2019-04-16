import random
import signal
import sys
import threading
import time

import click

from tomomibot import __version__
from tomomibot.server import Server
from tomomibot.session import Session
from tomomibot.voice import Voice


class Runtime:

    def __init__(self, ctx, voice_name, model, **kwargs):
        self.ctx = ctx

        self._display_welcome()

        self._thread = None

        # Prepare session
        voice = Voice(voice_name)
        reference_voice_name = kwargs.get('reference', None)
        if reference_voice_name is not None:
            reference_voice = Voice(reference_voice_name)
        else:
            reference_voice = None

        self._session = Session(self.ctx, voice,
                                model, reference_voice, **kwargs)

        # Prepare OSC server
        self._server = Server(ctx, **kwargs)

        # Subscribe to OSC message handler
        self._subscribe_events()

    def initialize(self):
        self._init_signal()
        self._server.start()
        self._session.start()

        # This is our main thread. Keep it alive!
        while self._session.is_running or self._server.is_running:
            time.sleep(1)

    def _display_welcome(self):
        welcome = """
▄▄▄▄▄      • ▌ ▄ ·.       • ▌ ▄ ·. ▪  ▄▄▄▄·      ▄▄▄▄▄
•██  ▪     ·██ ▐███▪▪     ·██ ▐███▪██ ▐█ ▀█▪▪    •██
 ▐█.▪ ▄█▀▄ ▐█ ▌▐▌▐█· ▄█▀▄ ▐█ ▌▐▌▐█·▐█·▐█▀▀█▄ ▄█▀▄ ▐█.▪
 ▐█▌·▐█▌.▐▌██ ██▌▐█▌▐█▌.▐▌██ ██▌▐█▌▐█▌██▄▪▐█▐█▌.▐▌▐█▌·
 ▀▀▀  ▀█▄▀▪▀▀  █▪▀▀▀ ▀█▄▀▪▀▀  █▪▀▀▀▀▀▀·▀▀▀▀  ▀█▄▀▪▀▀▀
        """
        for line in welcome.split('\n'):
            self.ctx.log(click.style(line, fg=random.choice([
                'blue',
                'cyan',
                'green',
                'magenta',
                'red',
                'yellow',
            ])))
        self.ctx.log('Version: %s' % __version__)
        self.ctx.log('Exit with [CTRL] + [C]\n')

    def _subscribe_events(self):
        @self._server.emitter.on('reset')
        def on_reset():
            self._session.reset_sequence()

        @self._server.emitter.on('param')
        def on_param(param_key, param_value):
            if param_key == 'volume':
                self._session.master_volume = param_value
            elif param_key == 'temperature':
                self._session.temperature = param_value
            elif param_key == 'interval':
                self._session.interval = param_value

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
            self._handle_exit()
            self.ctx.log('Shutdown confirmed!')
            return

    def _signal_stop(self, sig, frame):
        self._handle_exit()

    def _handle_exit(self):
        self._server.stop()
        self._session.stop()
