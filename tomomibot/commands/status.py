import click

from tomomibot.cli import pass_context


@click.command('status', short_help='Displays the current systems status')
@pass_context
def cli(ctx):
    """Display system info and audio devices."""
    ctx.log('Hello, World!')
