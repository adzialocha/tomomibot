import json
import os

from tomomibot.const import GENERATED_FOLDER, DATA_FILE


def line(char='-', length=48):
    """Generates a string of characters with a certain length"""
    return ''.join([char for _ in range(48)])


def health_check():
    """Checks if needed folders exist, otherwise create them"""
    base_dir = os.path.join(os.getcwd(), GENERATED_FOLDER)
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)


def check_valid_voice(name):
    """Checks if a voice is given and contains all needed files"""
    voice_dir = os.path.join(os.getcwd(), GENERATED_FOLDER, name)
    if not os.path.isdir(voice_dir):
        raise FileNotFoundError('Could not find voice folder')

    voice_data_file = os.path.join(voice_dir, DATA_FILE)
    if not os.path.isfile(voice_data_file):
        raise FileNotFoundError('Could not find {} in voice folder'.format(
            DATA_FILE))

    with open(voice_data_file) as file:
        for entry in json.load(file):
            wav_path = os.path.join(voice_dir, '{}.wav'.format(entry['id']))
            if not os.path.isfile(wav_path):
                raise FileNotFoundError(
                    'Could not find wav file "{}" in voice folder'.format(
                        wav_path))
