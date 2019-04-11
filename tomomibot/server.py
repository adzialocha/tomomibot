import threading

from pythonosc import dispatcher, osc_server

from tomomibot.const import OSC_ADDRESS, OSC_PORT


class Server:

    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        self.is_running = False

        self.port = kwargs.get('port', OSC_PORT)
        self.address = kwargs.get('address', OSC_ADDRESS)

        # Prepare OSC message dispatcher and UDP server
        disp = dispatcher.Dispatcher()
        disp.map('/tomomibot/*', self._on_param)
        bind = (self.address, self.port)
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
            print('Le Reset!')
            return

        # We expect one float argument from now on
        if not len(args) == 1 or type(args[0]) is not float:
            return

        if param == 'volume':
            print('Volume!', args[0])
        elif param == 'temperature':
            print('temperature', args[0])
        elif param == 'interval':
            print('interval', args[0])
