import json
import os

import click
import librosa
import numpy as np
import soundfile as sf


from tomomibot.audio import detect_onsets, slice_audio, mfcc_features
from tomomibot.const import (
    GENERATED_FOLDER, SEQUENCE_FILE, ONSET_FILE, SILENCE_POINT)


def trim_silence(audio, threshold):
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def analyze_sequence(y, sr, blocks_per_second, threshold=0):
    sequence = []
    block_size = sr // blocks_per_second
    start = 0
    end = block_size

    while end <= len(y):
        # Slice audio
        y_slice = y[start:end]

        # Remove silent passages
        trimmed = trim_silence(y_slice, threshold)
        trimmed = trimmed.reshape(-1, 1)

        if trimmed.size > 0:
            # Analyse audio
            melfc = mfcc_features(y_slice, sr)
            sequence.append(melfc.tolist())

        start += block_size
        end += block_size

    return sequence


def generate_voice(ctx, file, name,
                   db_threshold=10,
                   block=120,
                   blocks_per_second=10):
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

            # Additionally, generate a sequence for training
            sequence += analyze_sequence(y, sr, blocks_per_second)

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


def generate_training_data(ctx, voice_primary, voice_secondary):
    """Generate a trainable sequence of two voices playing together"""
    training_data = []

    # Project both sequences into the primary voice PCA space
    sequence_primary = voice_primary.project(voice_primary.sequence)
    sequence_secondary = voice_primary.project(voice_secondary.sequence)

    # Generate sequence
    sequence_len = max(len(sequence_primary), len(sequence_secondary))
    for i in range(sequence_len):
        frame = []
        if i < len(sequence_primary):
            frame.append([sequence_primary[i][0], sequence_primary[i][1]])
        else:
            frame.append(SILENCE_POINT)

        if i < len(sequence_secondary):
            frame.append([sequence_secondary[i][0], sequence_secondary[i][1]])
        else:
            frame.append(SILENCE_POINT)

        training_data.append(frame)

    ctx.log('Sequence with {} events generated.'.format(
        sequence_len))

    return np.array(training_data)
