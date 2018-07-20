import click

from tomomibot.cli import pass_context
from tomomibot.train import train_sequence_model
from tomomibot.utils import check_valid_voice


@click.command('train', short_help='Train a model for sequence prediction')
@click.option('--grid_size',
              default=10,
              help='Reduce the possible points in the PCA space with a grid')
@click.option('--batch_size',
              default=128,
              help='How many batches to train per step')
@click.option('--data_split',
              default=0.2,
              help='Percentage of dataset for validation and testing')
@click.option('--seq_len',
              default=10,
              help='Length of a point sequence to learn')
@click.option('--epochs',
              default=10,
              help='How many training epochs')
@click.option('--num_layers',
              default=3,
              help='Number of hidden LSTM layers in neural network')
@click.option('--num_units',
              default=256,
              help='Number of units per layer in neural networks')
@click.argument('primary_voice')
@click.argument('secondary_voice')
@click.argument('name')
@pass_context
def cli(ctx, primary_voice, secondary_voice, name, **kwargs):
    try:
        check_valid_voice(primary_voice)
        check_valid_voice(secondary_voice)
    except FileNotFoundError as err:
        ctx.elog('The given voices are invalid: {}'.format(err))
    else:
        train_sequence_model(ctx,
                             primary_voice,
                             secondary_voice,
                             name,
                             **kwargs)
