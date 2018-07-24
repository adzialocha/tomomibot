import click

from tomomibot.cli import pass_context
from tomomibot.const import (NUM_UNITS, NUM_LAYERS, NUM_CLASSES, DROPOUT,
                             BATCH_SIZE, DATA_SPLIT, SEQ_LEN, EPOCHS)
from tomomibot.train import train_sequence_model
from tomomibot.utils import check_valid_voice


@click.command('train', short_help='Train a model for sequence prediction')
@click.option('--num_classes',
              default=NUM_CLASSES,
              help='Number of classes to cluster the dataset (k-means)')
@click.option('--batch_size',
              default=BATCH_SIZE,
              help='How many batches to train per step')
@click.option('--data_split',
              default=DATA_SPLIT,
              help='Percentage of dataset for validation and testing')
@click.option('--seq_len',
              default=SEQ_LEN,
              help='Length of a point sequence to learn')
@click.option('--epochs',
              default=EPOCHS,
              help='How many training epochs')
@click.option('--num_layers',
              default=NUM_LAYERS,
              help='Number of hidden LSTM layers in neural network')
@click.option('--num_units',
              default=NUM_UNITS,
              help='Number of units per layer in neural networks')
@click.option('--dropout',
              default=DROPOUT,
              help='Dropout after every layer')
@click.option('--save_sequence',
              default=False,
              help='Save sequence as .json file after generation')
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
