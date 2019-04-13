import json
import os

import click
import librosa
import numpy as np
import soundfile as sf


from tomomibot.audio import detect_onsets, slice_audio, mfcc_features
from tomomibot.const import GENERATED_FOLDER, ONSET_FILE


SILENCE_DURATION_MIN = 50


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
    data = []

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
                data.append({'id': counter,
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
    data_path = os.path.join(voice_dir, ONSET_FILE)
    with open(data_path, 'w') as file:
        json.dump(data, file, indent=2, separators=(',', ': '))
    ctx.log('saved .json file with analyzed data.')


def find_silence(voice, sr=44100):
    """Find silence in and in-between sound events"""
    sequence = []
    last_step = voice.data[0]
    id_counter = 1

    for step_index, step in enumerate(voice.data):
        duration_between = ((step['start'] - last_step['end']) / sr) * 1000

        # Add silence between sound events
        if duration_between >= SILENCE_DURATION_MIN:
            silence_step = {
                'id': id_counter,
                'rms': 0,
                'rms_avg': 0,
                'start': last_step['end'],
                'end': step['start'],
            }

            sequence.append(silence_step)

            id_counter += 1

        converted_step = step.copy()

        # Keep soundfile id
        converted_step['sound_id'] = step['id']
        converted_step['id'] = id_counter

        # Store PCA data and average volume
        converted_step['pca'] = voice.points[step_index]
        if voice.has_rms:
            converted_step['rms_avg'] = voice.rms_avg[step_index]

        sequence.append(converted_step)

        id_counter += 1
        last_step = step

    return sequence


def generate_sequence(ctx, voice_primary, voice_secondary,
                      save_sequence=False):
    """Generate a trainable sequence of two voices playing together"""
    sequence = []
    counter = 0

    # Find events
    steps_primary = find_silence(voice_primary)

    # Check if voices are the same
    if voice_primary.name == voice_secondary.name:
        sequence = [[s, s] for s in steps_primary]
        counter = len(steps_primary)
    else:
        steps_secondary = find_silence(voice_secondary)

        # Go through all secondary sound events
        with click.progressbar(length=len(steps_secondary),
                               label='Progress') as bar:
            for i, step_secondary in enumerate(steps_secondary):
                start = steps_secondary[i]['start']
                end = steps_secondary[i]['end']

                # Find a simultaneous sound event in primary voice
                found_step = None
                for j, step_primary in reversed(
                        list(enumerate(steps_primary))):
                    start_primary = voice_primary[j]['start']
                    end_primary = voice_primary[j]['end']

                    if not (end <= start_primary or start >= end_primary):
                        found_step = j
                        counter += 1
                        break

                # Add played point to other
                if found_step:
                    sequence.append([step_primary, found_step])

                bar.update(1)

    ctx.log('Sequence with {} events and {} targets generated.'.format(
        len(sequence),
        counter))

    # ... and save sequence file
    if save_sequence:
        sequence_path = os.path.join(os.getcwd(), 'sequence-{}-{}.json'.format(
            voice_primary.name,
            voice_secondary.name))
        with open(sequence_path, 'w') as file:
            json.dump(sequence, file, indent=2, separators=(',', ': '))
        ctx.log('Saved .json file with sequence at {}.'.format(sequence_path))

    return np.array(sequence)
