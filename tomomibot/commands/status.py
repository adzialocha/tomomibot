import os

import click

from tomomibot.audio import all_inputs, all_outputs
from tomomibot.cli import pass_context
from tomomibot.const import GENERATED_FOLDER, MODELS_FOLDER
from tomomibot.utils import line, check_valid_voice, check_valid_model


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
    ctx.log('{0:2} {1:20} {2:10} {3:10}'.format(
        ' #', 'Name', 'Status', 'Version'))
    ctx.log(line())
    voice_dir = os.path.join(os.getcwd(), GENERATED_FOLDER)
    for i, entry in enumerate(os.scandir(voice_dir)):
        if entry.is_dir():
            try:
                version = check_valid_voice(entry.name)
            except FileNotFoundError:
                status = click.style('✘', fg='red')
            else:
                status = click.style('✓', fg='green')
            ctx.log('{0:2} {1:20} {2:10} {3:10}'.format(
                i,
                entry.name[:30],
                status,
                version or '?'))


def list_models(ctx):
    ctx.log(line())
    ctx.log('{0:2} {1:30} {2:10}'.format(' #', 'Name', 'Status'))
    ctx.log(line())
    model_dir = os.path.join(os.getcwd(), MODELS_FOLDER)
    for i, entry in enumerate(os.scandir(model_dir)):
        if entry.is_file():
            try:
                check_valid_model(entry.name.split('.')[0])
            except FileNotFoundError:
                status = click.style('✘', fg='red')
            else:
                status = click.style('✓', fg='green')
            ctx.log('{0:2} {1:30} {2:10}'.format(
                i,
                entry.name[:30],
                status))


@click.command('status', short_help='Display system info and audio devices')
@click.argument('model', required=False)
@pass_context
def cli(ctx, model):
    """Display system info and audio devices or inspect models."""
    if model:
        ctx.log(click.style('Inspect model "{}"'.format(model), bold=True))

        try:
            check_valid_model(model)
        except FileNotFoundError:
            ctx.elog('Could not find model.')
        else:
            # Load model and print a summary
            from keras.models import load_model

            model_name = '{}.h5'.format(model)
            model_path = os.path.join(os.getcwd(), MODELS_FOLDER, model_name)
            model_test = load_model(model_path)

            model_test.summary()
    else:
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
        ctx.log('')

        ctx.log(click.style('Trained models', bold=True))
        list_models(ctx)
