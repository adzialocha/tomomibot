import click

from tomomibot.cli import pass_context
from tomomibot.runtime import Runtime


@click.command('start', short_help='Start a live session')
@pass_context
def cli(ctx):
    """Start a live session with tomomibot."""
    options = dict()

    runtime = Runtime(ctx, options)
    runtime.initialize()
