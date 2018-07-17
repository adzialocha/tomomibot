from tomomibot.generate import generate_sequence
from tomomibot.voice import Voice


def train_sequence_model(ctx, primary_voice, secondary_voice):
    sequence = generate_sequence(ctx,
                                 Voice(primary_voice),
                                 Voice(secondary_voice))
