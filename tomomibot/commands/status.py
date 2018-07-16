import os

import click

from tomomibot.audio import all_inputs, all_outputs
from tomomibot.cli import pass_context
from tomomibot.const import GENERATED_FOLDER
from tomomibot.utils import line, check_valid_voice


def list_audio_channels(ctx, channels):
    ctx.log(line())
    ctx.log('{0:2} {1:8} {2:30}'.format(' #', 'Channels', 'Name'))
    ctx.log(line())
    for i, chn in enumerate(channels):
        ctx.log('{0:2} {1:8} {2:30}'.format(
            i,
            chn.channels,
            chn.name[:30]))


def list_voices(ctx):
    ctx.log(line())
    ctx.log('{0:2} {1:30} {2:10}'.format(' #', 'Name', 'Status'))
    ctx.log(line())
    voice_dir = os.path.join(os.getcwd(), GENERATED_FOLDER)
    for i, entry in enumerate(os.scandir(voice_dir)):
        if entry.is_dir():
            try:
                check_valid_voice(entry.name)
            except FileNotFoundError:
                status = click.style('✘', fg='red')
            else:
                status = click.style('✓', fg='green')
            ctx.log('{0:2} {1:30} {2:10}'.format(
                i,
                entry.name[:30],
                status))


@click.command('status', short_help='Display system info and audio devices')
@pass_context
def cli(ctx):
    """Display system info and audio devices."""
    inputs = all_inputs()
    outputs = all_outputs()

    ctx.log(click.style('Audio input devices', bold=True))
    list_audio_channels(ctx, inputs)
    ctx.log('')

    ctx.log(click.style('Audio output devices', bold=True))
    list_audio_channels(ctx, outputs)
    ctx.log('')

    ctx.log(click.style('Generated voices', bold=True))
    list_voices(ctx)
