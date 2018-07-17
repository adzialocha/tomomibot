import json
import os

import click
import librosa
import numpy as np
import soundfile as sf


from tomomibot.audio import detect_onsets, slice_audio, mfcc_features
from tomomibot.const import GENERATED_FOLDER, DATA_FILE, SILENCE_POINT


PRECISION = 5


def round_point(point, precision):
    """Helper to round floats in point vector"""
    return [round(x, precision) for x in point]


def generate_voice(ctx, file, name, db_threshold=10, block=120):
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

    # Generate and save data file
    data_path = os.path.join(voice_dir, DATA_FILE)
    with open(data_path, 'w') as file:
        json.dump(data, file, indent=2, separators=(',', ': '))

    ctx.log('Saved .json file with analyzed data.')


def generate_sequence(ctx, voice_primary, voice_secondary):
    """Generate a trainable sequence of two voices playing together"""
    sequence = []

    ctx.log('Generate a trainable sequence')
    ctx.log('Primary voice: "{}"'.format(voice_primary.name))
    ctx.log('Secondary voice: "{}"'.format(voice_secondary.name))

    # Project secondary voice points into the primary voice PCA space
    points_secondary = voice_primary.project_voice(voice_secondary)

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
            sequence.append([
                round_point(point.tolist(), PRECISION),
                round_point(found_point, PRECISION)])

            bar.update(1)

    ctx.log('Sequence with {} events and {} targets generated.'.format(
        len(points_secondary),
        counter))

    return sequence
