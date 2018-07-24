import click

from tomomibot.cli import pass_context
from tomomibot.generate import generate_voice
from tomomibot.const import THRESHOLD_DB, BLOCK_SEC


@click.command('generate', short_help='Generate voice based on .wav file')
@click.option('--onset_threshold',
              default=THRESHOLD_DB,
              help='Ignore audio events under this dB value')
@click.option('--block',
              default=BLOCK_SEC,
              help='Slice the audio file in this length (sec)')
@click.argument('file', type=click.Path(exists=True))
@click.argument('name')
@pass_context
def cli(ctx, file, name, **kwargs):
    generate_voice(ctx, file, name,
                   db_threshold=kwargs.get('onset_threshold'),
                   block=kwargs.get('block'))
