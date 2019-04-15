import json
import os

import numpy as np

from tomomibot.const import (GENERATED_FOLDER, ONSET_FILE, MODELS_FOLDER,
                             DURATIONS, NUM_CLASSES_DURATIONS,
                             NUM_CLASSES_DYNAMICS, SILENCE_CLASS)


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

    with open(voice_data_file) as file:
        data = json.load(file)

        if 'version' in data:
            version = data['version']
        else:
            version = 1

        if version == 1:
            sequence = data
        elif version == 2:
            sequence = data['sequence']
        else:
            raise RuntimeError('Unknow voice version')

        for step in sequence:
            wav_path = make_wav_path(name, step['id'])
            if not os.path.isfile(wav_path):
                raise FileNotFoundError(
                    'Could not find wav file "{}" in voice folder'.format(
                        wav_path))


def get_num_classes(num_sound_classes,
                    use_dynamics, use_durations):
    """Calculate how many classes the model can predict"""
    num_classes = num_sound_classes + 1  # .. add silence class
    if use_dynamics:
        num_classes *= NUM_CLASSES_DYNAMICS
    if use_durations:
        num_classes *= NUM_CLASSES_DURATIONS

    return num_classes


def get_feature_matrix(num_sound_classes, use_dynamics, use_durations):
    """Returns the matrix of our features"""
    if not use_dynamics and not use_durations:
        feature_matrix = np.zeros((num_sound_classes + 1))
    elif use_dynamics and not use_durations:
        feature_matrix = np.zeros((num_sound_classes + 1,
                                   NUM_CLASSES_DYNAMICS))
    elif not use_dynamics and use_durations:
        feature_matrix = np.zeros((num_sound_classes + 1,
                                   NUM_CLASSES_DURATIONS))
    else:
        feature_matrix = np.zeros((num_sound_classes + 1,
                                   NUM_CLASSES_DYNAMICS,
                                   NUM_CLASSES_DURATIONS))

    return feature_matrix


def encode_duration_class(duration):
    """Translate ms into a duration class"""
    duration_class = NUM_CLASSES_DURATIONS - 1

    for duration_def_class, duration_def in enumerate(DURATIONS):
        if duration < duration_def:
            duration_class = duration_def_class
            break

    return duration_class


def encode_dynamic_class(class_sound, rms):
    """Translate RMS into a dynamic class"""
    if class_sound == SILENCE_CLASS:
        class_dynamic = 0
    else:
        class_dynamic = int(round(rms * (NUM_CLASSES_DYNAMICS - 1)))

    return class_dynamic


def encode_feature_vector(num_sound_classes, class_sound,
                          class_dynamic=None, class_duration=None,
                          use_dynamics=False, use_durations=False):
    """Translate sound, dynamic and duration classes into an indexed class"""
    # Define feature vector and matrix
    feature_matrix = get_feature_matrix(num_sound_classes,
                                        use_dynamics,
                                        use_durations)

    if not use_dynamics and not use_durations:
        feature_vector = class_sound
    else:
        feature_vector = [class_sound]

        if use_dynamics:
            feature_vector.append(class_dynamic)

        if use_durations:
            feature_vector.append(class_duration)

    # Encode sequence
    if use_dynamics or use_durations:
        feature_vector = np.ravel_multi_index(feature_vector,
                                              feature_matrix.shape)
        feature_vector = np.array(feature_vector)

    return feature_vector


def decode_classes(class_index, num_sound_classes,
                   use_dynamics, use_durations):
    """Translate indexed class back into sound, dynamic and duration classes"""
    class_dynamic = None
    class_duration = None

    if not use_dynamics and not use_durations:
        class_sound = class_index
    else:
        feature_matrix = get_feature_matrix(num_sound_classes,
                                            use_dynamics,
                                            use_durations)

        feature_vector = np.unravel_index([class_index], feature_matrix)[0]

        class_sound = feature_vector[0]

        if use_dynamics and not use_durations:
            class_dynamic = feature_vector[1]
        elif not use_dynamics and use_durations:
            class_duration = feature_vector[1]
        else:
            class_dynamic = feature_vector[1]
            class_duration = feature_vector[2]

    return class_sound, class_dynamic, class_duration
