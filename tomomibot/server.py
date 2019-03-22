import time


class Server:

    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        self.is_running = False

        # @TODO read arguments for network setup

    def start(self):
        self.is_running = True
        # @TODO start the http server

        # @TODO start upd server

        # @TODO start websocket server

        while self.is_running:
            time.sleep(1)

    def stop(self):
        self.is_running = False
