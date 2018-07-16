import json
import os
import warnings

import click
import soundfile as sf
import numpy as np
import librosa

from tomomibot.audio import detect_onsets, slice_audio, mfcc_features
from tomomibot.const import GENERATED_FOLDER, DATA_FILE


BLOCK_SIZE = 1024 * 1024 * 2


# Ignore LAPACK warning (https://github.com/scipy/scipy/issues/5998)
warnings.filterwarnings(action='ignore',
                        module='scipy',
                        message='^internal gelsd')


def generate_voice(ctx, file, name, db_threshold=10):
    """Generate voice based on .wav file"""

    # Read file in blocks to not load the whole file into memory
    block_gen = sf.blocks(file,
                          blocksize=BLOCK_SIZE,
                          always_2d=True,
                          dtype='float32')
    sr = sf.info(file).samplerate

    voice_dir = os.path.join(os.getcwd(), GENERATED_FOLDER, name)
    if not os.path.isdir(voice_dir):
        os.mkdir(voice_dir)
    else:
        ctx.elog('Generated folder "%s" already exists.' % name)

    block_num = sf.info(file).frames // BLOCK_SIZE
    counter = 1
    data = []

    ctx.log('Analyze .wav file "%s" with sample rate %i' % (
        click.format_filename(file, shorten=True),
        sr))

    # Load audio file
    with click.progressbar(length=block_num,
                           label='Progress') as bar:
        for bl in block_gen:
            # Downmix to mono
            y = np.mean(bl, axis=1)

            # Detect onsets
            onsets, times = detect_onsets(
                y, sr=sr, db_threshold=db_threshold)

            # Slice audio into parts, analyze mffcs and save them
            slices = slice_audio(y,
                                 onsets,
                                 offset=(BLOCK_SIZE * (counter - 1)))
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

            bar.update(1)

    ctx.log('Created %i slices' % (counter - 1))

    # Generate and save data file
    data_path = os.path.join(voice_dir, DATA_FILE)
    with open(data_path, 'w') as file:
        json.dump(data, file, indent=2, separators=(',', ': '))

    ctx.log('Saved .json file with analyzed data.')
