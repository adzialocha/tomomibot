import click

from tomomibot.cli import pass_context
from tomomibot.train import train_sequence_model
from tomomibot.utils import check_valid_voice


@click.command('train', short_help='Train a model for sequence prediction')
@click.argument('primary_voice')
@click.argument('secondary_voice')
@pass_context
def cli(ctx, primary_voice, secondary_voice, **kwargs):
    try:
        check_valid_voice(primary_voice)
        check_valid_voice(secondary_voice)
    except FileNotFoundError as err:
        ctx.elog('One of the given voices is invalid: {}'.format(err))
    else:
        train_sequence_model(ctx, primary_voice, secondary_voice)
