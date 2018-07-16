import click

from tomomibot.audio import all_inputs, all_outputs
from tomomibot.cli import pass_context
from tomomibot.utils import line


def list_audio_channels(ctx, channels):
    ctx.log(line())
    ctx.log('{0:2} {1:8} {2:30}'.format(
        ' #', 'Channels', 'Name'))
    ctx.log(line())
    for i, chn in enumerate(channels):
        ctx.log('{0:2} {1:8} {2:30}'.format(
            i,
            chn.channels,
            chn.name[:30]))


def list_audio_devices(ctx):
    inputs = all_inputs()
    outputs = all_outputs()

    ctx.log(click.style('Audio input devices', bold=True))
    list_audio_channels(ctx, inputs)
    print()
    ctx.log(click.style('Audio output devices', bold=True))
    list_audio_channels(ctx, outputs)


@click.command('status', short_help='Display system info and audio devices')
@pass_context
def cli(ctx):
    """Display system info and audio devices."""
    list_audio_devices(ctx)
