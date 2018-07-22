import click

from tomomibot.cli import pass_context
from tomomibot.runtime import Runtime
from tomomibot.utils import check_valid_voice, check_valid_model


@click.command('start', short_help='Start a live session')
@click.option('--interval',
              default=1,
              help='Interval (in seconds) of analyzing incoming live signal')
@click.option('--noise_factor',
              default=0.1,
              help='Randomize predicted points to make them more diverse')
@click.option('--input_device',
              default=0,
              help='Index of audio device for incoming signal')
@click.option('--output_device',
              default=0,
              help='Index of audio device for outgoing signal')
@click.option('--input_channel',
              default=0,
              help='Index of channel for incoming signal')
@click.option('--output_channel',
              default=0,
              help='Index of channel for outgoing signal')
@click.option('--samplerate',
              default=44100,
              help='Sample rate of audio signals')
@click.option('--onset_threshold',
              default=10,
              help='Ignore audio events under this dB value')
@click.argument('voice')
@click.argument('model')
@pass_context
def cli(ctx, voice, model, **kwargs):
    """Start a live session with tomomibot."""
    try:
        check_valid_model(model)
    except FileNotFoundError as err:
        ctx.elog('Model "{}" is invalid: {}'.format(model, err))

    try:
        check_valid_voice(voice)
    except FileNotFoundError as err:
        ctx.elog('Voice "{}" is invalid: {}'.format(voice, err))
    else:
        runtime = Runtime(ctx, voice, model, **kwargs)
        runtime.initialize()
