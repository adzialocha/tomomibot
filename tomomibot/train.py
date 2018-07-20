import os
import sys

from keras import layers
from keras.models import Sequential, load_model
import click
import numpy as np

from tomomibot.const import MODELS_FOLDER
from tomomibot.generate import generate_training_data
from tomomibot.utils import line
from tomomibot.voice import Voice


def encode_point(p, grid_size):
    return [min(round(grid_size * max(f, 0)), grid_size - 1) for f in p]


def decode_point(p, grid_size):
    return [f / grid_size for f in p]


def encode_data(data, grid_size):
    """Encode 2D positions into a grid and convert this to index values"""
    new_data = []
    m = np.zeros((grid_size, grid_size))
    for column in data:
        new_column = []
        for point in column:
            ac_converted = np.array(
                encode_point(point, grid_size),
                dtype='int32')
            ac_indexed = np.ravel_multi_index(ac_converted, m.shape)
            new_column.append(ac_indexed)
        new_data.append(new_column)
    return np.array(new_data)


def decode_data(data, grid_size):
    """Decode indexed grid positions to (x, y) float values"""
    new_data = []
    m = np.zeros((grid_size, grid_size))
    for column in data:
        new_column = []
        for ac_converted in column:
            unraveled = np.unravel_index(ac_converted, m.shape)
            point = np.array(
                decode_point(unraveled, grid_size),
                dtype='float32')
            new_column.append(point)
        new_data.append(new_column)
    return np.array(new_data)


def generator(data, seq_len, min_index, max_index, batch_size):
    """Iterator to yield data for training, testing and validating"""
    i = min_index
    while 1:
        if i + batch_size >= max_index:
            i = min_index
        rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)

        samples = np.zeros((len(rows), seq_len), dtype='int32')
        targets = np.zeros((len(rows)), dtype='int32')

        for j, _ in enumerate(rows):
            indices = range(rows[j], rows[j] + seq_len)
            if indices[-1] < max_index:
                samples[j] = data[:, 0][indices]
                targets[j] = data[:, 1][indices][-1]

        yield samples, targets


def train_sequence_model(ctx, primary_voice, secondary_voice, name, **kwargs):
    """Train a LSTM neural network on sequence data from a performance"""
    model_name = '{}.h5'.format(name)
    model_path = os.path.join(os.getcwd(), MODELS_FOLDER, model_name)
    resume = False

    if os.path.isfile(model_path):
        click.confirm(
            'Found model with same name! Do you want to resume training?',
            abort=True)
        resume = True

    # Parameters and Hyperparameters
    grid_size = kwargs.get('grid_size', 10)

    batch_size = kwargs.get('batch_size', 128)
    data_split = kwargs.get('data_split', 0.2)
    seq_len = kwargs.get('seq_len', 10)

    epochs = kwargs.get('epochs', 10)
    num_layers = kwargs.get('num_layers', 3)
    num_units = kwargs.get('num_units', 256)

    ctx.log('\nParameters:')
    ctx.log(line(length=32))
    ctx.log('name:\t\t{}'.format(name))
    ctx.log('grid_size:\t{}'.format(grid_size))
    ctx.log('batch_size:\t{}'.format(batch_size))
    ctx.log('data_split:\t{}'.format(data_split))
    ctx.log('seq_len:\t{}'.format(seq_len))
    ctx.log('epochs:\t\t{}'.format(epochs))
    if not resume:
        ctx.log('num_layers:\t{}'.format(num_layers))
        ctx.log('num_units:\t{}'.format(num_units))
    ctx.log(line(length=32))
    ctx.log('')

    primary_voice = Voice(primary_voice)
    secondary_voice = Voice(secondary_voice)

    # Generate training data from voice sequences
    ctx.log(click.style('1. Generate training data from voices', bold=True))
    ctx.log('Primary voice: "{}"'.format(primary_voice.name))
    ctx.log('Secondary voice: "{}"'.format(secondary_voice.name))

    data = generate_training_data(ctx,
                                  primary_voice,
                                  secondary_voice)
    ctx.log('')

    # Encode data before training
    ctx.log(click.style('2. Encode data before training', bold=True))
    encoded_data = encode_data(data, grid_size)
    num_classes = grid_size ** 2
    ctx.log('Number of classes: {}\n'.format(num_classes))

    # Split in 3 sets for training, validation and testing
    ctx.log(click.style('3. Split data in sets', bold=True))
    validation_steps = round((data_split / 2) * len(data))

    train_max = len(data) - (validation_steps * 2)
    val_min = train_max + 1
    val_max = train_max + validation_steps + 1
    test_min = train_max + validation_steps + 2
    test_max = len(data) - 1

    training_steps = test_max - test_min

    train_gen = generator(encoded_data,
                          seq_len=seq_len,
                          batch_size=batch_size,
                          min_index=0,
                          max_index=train_max)

    val_gen = generator(encoded_data,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        min_index=val_min,
                        max_index=val_max)

    test_gen = generator(encoded_data,
                         seq_len=seq_len,
                         batch_size=batch_size,
                         min_index=test_min,
                         max_index=test_max)

    steps_per_epoch = train_max // batch_size

    ctx.log('Batch size: {}'.format(batch_size))
    ctx.log('Steps per epoch: {}'.format(steps_per_epoch))
    ctx.log('Split for validation & test @ {0:.2f}%'.format(
        data_split * 100))
    ctx.log('Training set: {}-{}'.format(0, train_max))
    ctx.log('Validation set: {}-{}'.format(val_min, val_max))
    ctx.log('Test set: {}-{}\n'.format(test_min, test_max))

    # Define model
    ctx.log(click.style('4. Define a model', bold=True))
    if resume:
        ctx.log('Load existing model to resume training ..')
        try:
            model = load_model(model_path)
        except ValueError as err:
            ctx.elog(
                'Could not load model: {}'.format(err))
            sys.exit(1)
    else:
        model = Sequential()
        model.add(layers.Embedding(input_dim=num_classes,
                                   output_dim=num_units,
                                   input_length=seq_len))
        for n in range(num_layers - 1):
            model.add(layers.LSTM(num_units, return_sequences=True))
        model.add(layers.LSTM(num_units))
        model.add(layers.Dense(num_classes, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])

        model.summary()

    ctx.log('')

    # Training!
    ctx.log(click.style('5. Training!', bold=True))
    model.fit_generator(train_gen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=val_gen,
                        validation_steps=validation_steps)
    ctx.log('Finished training.\n')

    # Evaluate training
    ctx.log(click.style('6. Evaluation', bold=True))
    scores = []
    max_dist = .25

    for i in range((training_steps // batch_size)):
        # Predict point from model
        samples, targets = next(test_gen)
        results = model.predict(samples)

        for j, result in enumerate(results):
            # Reweight the softmax distribution
            result_value = np.argmax(result)

            # Decode data
            position = decode_data([[result_value]], grid_size).flatten()
            position_target = decode_data([[targets[j]]], grid_size).flatten()

            # Calculate distance between prediction and actual test target
            dist = max_dist - min(
                max_dist,
                np.linalg.norm(position - position_target))
            scores.append(0.0 if dist == 0.0 else dist / max_dist)

    score = np.average(scores)

    ctx.log('Score: {0:.2f}%\n'.format(score * 100))

    # Save model
    ctx.log(click.style('7. Store model weights', bold=True))
    ctx.log('Stored weights at "{}"'.format(model_path))
    model.save(model_path)
    ctx.log('Done!')
