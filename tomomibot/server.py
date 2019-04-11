import threading

from pyee import EventEmitter
from pythonosc import dispatcher, osc_server

from tomomibot.const import OSC_ADDRESS, OSC_PORT


class Server:

    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        self.is_running = False

        # Provide an interface for event subscribers
        self.emitter = EventEmitter()

        # Prepare OSC message dispatcher and UDP server
        self.address = kwargs.get('osc_address', OSC_ADDRESS)
        self.port = kwargs.get('osc_port', OSC_PORT)

        bind = (self.address, self.port)

        disp = dispatcher.Dispatcher()
        disp.map('/tomomibot/*', self._on_param)

        self._server = osc_server.ThreadingOSCUDPServer(bind, disp)

    def start(self):
        thread = threading.Thread(target=self._start_server)
        thread.daemon = True
        thread.start()

        self.is_running = True

    def stop(self):
        self._server.shutdown()
        self.is_running = False

    def _start_server(self):
        self.ctx.log('OSC server @ {}:{}'.format(self.address,
                                                 self.port))

        self._server.serve_forever()

    def _on_param(self, address, *args):
        param = address.replace('/tomomibot/', '')

        # Commands with no arguments
        if param == 'reset':
            self.emitter.emit('reset')
            return

        # We expect one float argument from now on
        if not len(args) == 1 or type(args[0]) is not float:
            return

        if param in ['volume', 'temperature'] and 0 <= args[0] <= 1:
            self.emitter.emit('param', param, args[0])

        if param in ['interval'] and 0 <= args[0] <= 5:
            self.emitter.emit('param', param, args[0])
