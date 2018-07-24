import json
import os

import click
import librosa
import numpy as np
import soundfile as sf


from tomomibot.audio import detect_onsets, slice_audio, mfcc_features
from tomomibot.const import (GENERATED_FOLDER, SEQUENCE_FILE,
                             ONSET_FILE, SILENCE_POINT)


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
            onsets, _ = detect_onsets(y, sr=sr, db_threshold=db_threshold)

            # Slice audio into parts, analyze mffcs and save them
            slices = slice_audio(y, onsets, offset=offset)
            for i in range(len(slices) - 1):
                # Normalize slice audio signal
                y_slice = librosa.util.normalize(slices[i][0])

                # Calculate MFCCs
                mfcc = mfcc_features(y_slice, sr)

                # Keep all information stored
                data.append({'id': counter,
                             'mfcc': mfcc.tolist(),
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

    # ... and save sequence file
    data_path = os.path.join(voice_dir, SEQUENCE_FILE)
    with open(data_path, 'w') as file:
        json.dump(sequence, file, indent=2, separators=(',', ': '))
    ctx.log('saved .json file with sequence.')


def generate_sequence(ctx, voice_primary, voice_secondary,
                      save_sequence=False):
    """Generate a trainable sequence of two voices playing together"""
    sequence = []

    # Project secondary voice points into the primary voice PCA space
    points_secondary = voice_primary.project(voice_secondary.mfccs)
    counter = 0

    # Go through all secondary sound events
    with click.progressbar(length=len(points_secondary),
                           label='Progress') as bar:
        for i, point in enumerate(points_secondary):
            start = voice_secondary.positions[i][0]
            end = voice_secondary.positions[i][1]

            # Find a simultaneous sound event in primary voice
            found_point = None
            for j, point_primary in reversed(
                    list(enumerate(voice_primary.points))):
                start_primary = voice_primary.positions[j][0]
                end_primary = voice_primary.positions[j][1]
                if not (end <= start_primary or start >= end_primary):
                    found_point = point_primary.tolist()
                    counter += 1
                    break

            # Set silence marking point when nothing was played
            if found_point is None:
                found_point = SILENCE_POINT

            # Add played point to other
            sequence.append([point.tolist(), found_point])

            bar.update(1)

    ctx.log('Sequence with {} events and {} targets generated.'.format(
        len(points_secondary),
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
