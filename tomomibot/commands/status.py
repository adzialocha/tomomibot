import click
import pyaudio

from tomomibot.cli import pass_context
from tomomibot.utils import line


def list_audio_devices(ctx):
    ctx.log(click.style('Audio devices', bold=True))
    audio = pyaudio.PyAudio()
    ctx.log(line())
    ctx.log('{0:2} {1:6} {2:7} {3:30}'.format(
        ' #', 'Inputs', 'Outputs', 'Name'))
    ctx.log(line())
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        ctx.log('{0:2d} {1:6d} {2:7d} {3:30}'.format(
            info['index'], info['maxInputChannels'],
            info['maxOutputChannels'], info['name'][:30]))


@click.command('status', short_help='Display system info and audio devices')
@pass_context
def cli(ctx):
    """Display system info and audio devices."""
    list_audio_devices(ctx)
