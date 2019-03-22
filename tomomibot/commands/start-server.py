import click

from tomomibot.cli import pass_context
from tomomibot.runtime import ServerRuntime


@click.command('start-server', short_help='Start an http/udp server with'
                                          'web-ui for live session')
@pass_context
def cli(ctx,  **kwargs):
    """Start an http/udp server with web-ui for live session."""
    runtime = ServerRuntime(ctx, **kwargs)
    runtime.initialize()
