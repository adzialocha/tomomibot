import json
import os

import click
import librosa
import numpy as np
import soundfile as sf


from tomomibot.audio import detect_onsets, slice_audio, mfcc_features
from tomomibot.const import GENERATED_FOLDER, ONSET_FILE


SILENCE_DURATION_MIN = 50
VERSION = 2
VOLUME_KERNEL_SIZE = 10


def generate_voice(ctx, file, name, db_threshold, block):
    """Generate voice based on .wav file"""

    # Extract information about sound file
    info = sf.info(file)
    sr = info.samplerate
    blocksize = sr * block
    block_num = info.frames // blocksize

    # Read file in blocks to not load the whole file into memory
    block_gen = sf.blocks(file,
                          blocksize=blocksize,
                          always_2d=True,
                          dtype='float32')

    # Create folder for voice
    voice_dir = os.path.join(os.getcwd(), GENERATED_FOLDER, name)
    if not os.path.isdir(voice_dir):
        os.mkdir(voice_dir)
    else:
        ctx.elog('Generated folder "%s" already exists.' % name)

    counter = 1
    block_no = 0
    sequence = []

    ctx.log('Analyze .wav file "%s" with %i frames and sample rate %i' % (
        click.format_filename(file, shorten=True),
        info.frames,
        sr))

    # Load audio file
    with click.progressbar(length=block_num,
                           label='Progress') as bar:
        for bl in block_gen:
            offset = blocksize * block_no

            # Downmix to mono
            y = np.mean(bl, axis=1)

            # Detect onsets
            onsets, _ = detect_onsets(y, sr=sr, db_threshold=-120)

            # Slice audio into parts, analyze mffcs and save them
            slices = slice_audio(y, onsets, offset=offset)
            for i in range(len(slices) - 1):
                y_slice = slices[i][0]

                # Calculate MFCCs
                mfcc = mfcc_features(y_slice, sr)

                # RMS Volume
                rms = librosa.feature.rms(y=y_slice)

                # Keep all information stored
                sequence.append({'id': counter,
                                 'mfcc': mfcc.tolist(),
                                 'rms': np.float32(np.max(rms)).item(),
                                 'start': np.uint32(slices[i][1]).item(),
                                 'end': np.uint32(slices[i][2]).item()})

                # Save file to generated subfolder
                path = os.path.join(voice_dir, '%i.wav' % counter)
                librosa.output.write_wav(path, y_slice, sr)
                counter += 1

            block_no += 1
            bar.update(1)

    ctx.log('Created %i slices' % (counter - 1))

    # generate and save data file
    data = {
        'version': VERSION,
        'meta': {
            'samplerate': sr,
        },
        'sequence': sequence,
    }

    data_path = os.path.join(voice_dir, ONSET_FILE)
    with open(data_path, 'w') as file:
        json.dump(data, file, indent=2, separators=(',', ': '))
    ctx.log('saved .json file with analyzed data.')


def generate_dynamics(voice):
    """Find silence and calculate average volume"""
    new_sequence = []

    last_step = voice.sequence[0]
    sr = voice.meta['samplerate']

    # Calculate average volume
    rms_data = [step['rms'] for step in voice.sequence]

    kernel = np.array(np.full((VOLUME_KERNEL_SIZE,), 1)) / VOLUME_KERNEL_SIZE
    rms_avg_data = np.convolve(rms_data, kernel, 'same')

    # Normalize rms values and store them
    rms_normalized = rms_data / np.max(rms_data)
    rms_avg_normalized = rms_avg_data / np.max(rms_avg_data)

    # Find silence in and in-between sound events
    for step_index, step in enumerate(voice.sequence):
        duration_between = ((step['start'] - last_step['end']) / sr) * 1000

        # Add silence between sound events
        if duration_between >= SILENCE_DURATION_MIN:
            silence_step = {
                'id': None,
                'rms': 0,
                'rms_avg': 0,
                'start': last_step['end'],
                'end': step['start'],
            }

            new_sequence.append(silence_step)

        converted_step = step.copy()

        # Store normalized (average) volumes
        converted_step['rms'] = rms_normalized[step_index]
        converted_step['rms_avg'] = rms_avg_normalized[step_index]

        # Add PCA data for later training
        converted_step['pca'] = voice.points[step_index]

        new_sequence.append(converted_step)

        last_step = step

    return new_sequence


def generate_sequence(ctx, voice_primary, voice_secondary,
                      save_sequence=False):
    """Generate a trainable sequence of two voices playing together"""
    sequence = []

    # Find events
    steps_primary = generate_dynamics(voice_primary)

    # Check if voices are the same
    if voice_primary.name == voice_secondary.name:
        ctx.log('Both voices are the same: generate a solo voice!')
        sequence = [[s, s] for s in steps_primary]
    else:
        steps_secondary = generate_dynamics(voice_secondary)

        # Go through all secondary sound events
        with click.progressbar(length=len(steps_secondary),
                               label='Progress') as bar:
            for i, step_secondary in enumerate(steps_secondary):
                start = steps_secondary[i]['start']
                end = steps_secondary[i]['end']

                # Find a simultaneous sound event in primary voice
                for step_index, step_primary in reversed(
                        list(enumerate(steps_primary))):
                    start_primary = steps_primary[step_index]['start']
                    end_primary = steps_primary[step_index]['end']

                    if not (end <= start_primary or start >= end_primary):
                        # Add played point to other
                        sequence.append([step_secondary, step_primary])
                        break

                bar.update(1)

    ctx.log('Sequence with {} events generated.'.format(len(sequence)))

    # ... and save sequence file
    if save_sequence:
        sequence_path = os.path.join(os.getcwd(), 'sequence-{}-{}.json'.format(
            voice_primary.name,
            voice_secondary.name))
        with open(sequence_path, 'w') as file:
            json.dump(sequence, file, indent=2, separators=(',', ': '))
        ctx.log('Saved .json file with sequence at {}.'.format(sequence_path))

    return np.array(sequence)
