import click

from tomomibot.cli import pass_context
from tomomibot.generate import generate_voice


@click.command('generate', short_help='Generate voice based on .wav file')
@click.option('--onset_threshold',
              default=10,
              help='Ignore audio events under this dB value')
@click.argument('file', type=click.Path(exists=True))
@click.argument('name')
@pass_context
def cli(ctx, file, name, **kwargs):
    generate_voice(ctx, file, name,
                   db_threshold=kwargs.get('onset_threshold'))
