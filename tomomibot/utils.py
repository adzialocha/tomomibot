import json
import os

import numpy as np

from tomomibot.const import (
    GENERATED_FOLDER, ONSET_FILE, SEQUENCE_FILE, MODELS_FOLDER)


def line(char='-', length=48):
    """Generates a string of characters with a certain length"""
    return ''.join([char for _ in range(length)])


def health_check():
    """Checks if needed folders exist, otherwise create them"""
    base_dir = os.path.join(os.getcwd(), GENERATED_FOLDER)
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    models_dir = os.path.join(os.getcwd(), MODELS_FOLDER)
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)


def make_wav_path(name, wav_id):
    voice_dir = os.path.join(os.getcwd(), GENERATED_FOLDER, name)
    return os.path.join(voice_dir, '{}.wav'.format(wav_id))


def check_valid_model(name):
    """Checks if a model is given"""
    model_file = os.path.join(os.getcwd(), MODELS_FOLDER, '{}.h5'.format(name))
    if not os.path.isfile(model_file):
        raise FileNotFoundError('Could not find {} in model folder'.format(
            name))


def check_valid_voice(name):
    """Checks if a voice is given and contains all needed files"""
    voice_dir = os.path.join(os.getcwd(), GENERATED_FOLDER, name)
    if not os.path.isdir(voice_dir):
        raise FileNotFoundError('Could not find voice folder')

    voice_data_file = os.path.join(voice_dir, ONSET_FILE)
    if not os.path.isfile(voice_data_file):
        raise FileNotFoundError('Could not find {} in voice folder'.format(
            ONSET_FILE))

    sequence_file = os.path.join(voice_dir, SEQUENCE_FILE)
    if not os.path.isfile(sequence_file):
        raise FileNotFoundError('Could not find {} in voice folder'.format(
            SEQUENCE_FILE))

    with open(voice_data_file) as file:
        for entry in json.load(file):
            wav_path = make_wav_path(name, entry['id'])
            if not os.path.isfile(wav_path):
                raise FileNotFoundError(
                    'Could not find wav file "{}" in voice folder'.format(
                        wav_path))


def plot_position(position, x_size=20, y_size=10):
    point_x = np.floor(((position[0] + 1.0) / 2.0) * x_size)
    point_y = np.floor(((position[1] + 1.0) / 2.0) * y_size)

    strb = ''

    for y in reversed(range(y_size + 1)):
        line = ''
        for x in range(x_size + 1):
            if point_x == x and point_y == y:
                char = '▪'
            else:
                char = '█'
            line += char
        strb += line + '\n'

    return strb
