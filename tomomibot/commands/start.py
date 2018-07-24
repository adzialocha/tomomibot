import click

from tomomibot.cli import pass_context
from tomomibot.runtime import Runtime
from tomomibot.utils import check_valid_voice, check_valid_model
from tomomibot.const import (INTERVAL_SEC, INPUT_DEVICE, OUTPUT_CHANNEL,
                             INPUT_CHANNEL, OUTPUT_DEVICE, SAMPLE_RATE,
                             THRESHOLD_DB, NUM_CLASSES, SEQ_LEN, TEMPERATURE)


@click.command('start', short_help='Start a live session')
@click.option('--interval',
              default=INTERVAL_SEC,
              help='Interval (in seconds) of analyzing incoming live signal')
@click.option('--input_device',
              default=INPUT_DEVICE,
              help='Index of audio device for incoming signal')
@click.option('--output_device',
              default=OUTPUT_DEVICE,
              help='Index of audio device for outgoing signal')
@click.option('--input_channel',
              default=INPUT_CHANNEL,
              help='Index of channel for incoming signal')
@click.option('--output_channel',
              default=OUTPUT_CHANNEL,
              help='Index of channel for outgoing signal')
@click.option('--samplerate',
              default=SAMPLE_RATE,
              help='Sample rate of audio signals')
@click.option('--threshold',
              default=THRESHOLD_DB,
              help='Ignore audio events under this db value')
@click.option('--num_classes',
              default=NUM_CLASSES,
              help='')
@click.option('--seq_len',
              default=SEQ_LEN,
              help='')
@click.option('--temperature',
              default=TEMPERATURE,
              help='')
@click.argument('voice')
@click.argument('model')
@pass_context
def cli(ctx, voice, model, **kwargs):
    """Start a live session with tomomibot."""
    try:
        check_valid_model(model)
    except FileNotFoundError as err:
        ctx.elog('Model "{}" is invalid: {}'.format(model, err))
    else:
        try:
            check_valid_voice(voice)
        except FileNotFoundError as err:
            ctx.elog('Voice "{}" is invalid: {}'.format(voice, err))
        else:
            runtime = Runtime(ctx, voice, model, **kwargs)
            runtime.initialize()
