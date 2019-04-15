import os
import sys

from keras import layers
from keras.models import Sequential, load_model
from sklearn.cluster import KMeans
import click
import numpy as np

from tomomibot.const import MODELS_FOLDER, SILENCE_CLASS
from tomomibot.generate import generate_sequence
from tomomibot.utils import (line, encode_duration_class, encode_dynamic_class,
                             encode_feature_vector, get_num_classes)
from tomomibot.voice import Voice


def reweight_distribution(original_distribution, temperature):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


def encode_data(sequence, num_clusters, use_dynamics, use_durations, sr):
    dataset = []

    # KMeans clustering of primary voice PCA sequence
    pca_data = []
    for step in sequence:
        if 'pca' in step[0]:
            pca_data.append(step[0]['pca'])
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pca_data)

    # Encode both voices
    for step in sequence:
        new_step = []

        for voice in step:
            class_dynamic = None
            class_duration = None

            # Define sound class
            if voice['id'] is None:
                class_sound = SILENCE_CLASS
            else:
                # +1 for silence
                class_sound = kmeans.predict([voice['pca']])[0] + 1

            # Define dynamic class
            if use_dynamics:
                class_dynamic = encode_dynamic_class(class_sound,
                                                     voice['rms_avg'])

            # Define duration class
            if use_durations:
                duration = (voice['end'] - voice['start']) / sr * 1000
                class_duration = encode_duration_class(duration)

            # Get final feature vector
            feature_vector = encode_feature_vector(num_clusters,
                                                   class_sound,
                                                   class_dynamic,
                                                   class_duration,
                                                   use_dynamics,
                                                   use_durations)

            new_step.append(feature_vector)

        dataset.append(new_step)
    return np.array(dataset)


def generator(data, seq_len, min_index, max_index, batch_size):
    """Iterator to yield data for training, testing and validating"""
    i = min_index
    while 1:
        if i + batch_size >= max_index:
            i = min_index
        rows = np.arange(i, min(i + batch_size, max_index - 1))
        i += len(rows)

        samples = np.zeros((len(rows), seq_len), dtype='int32')
        targets = np.zeros((len(rows)), dtype='int32')

        for j, _ in enumerate(rows):
            indices = range(rows[j], rows[j] + seq_len + 1)
            if indices[-1] < max_index:
                targets[j] = data[:, 0][indices][-1]
                samples[j] = data[:, 1][indices[:seq_len]]

        yield samples, targets


def train_sequence_model(ctx, primary_voice, secondary_voice, name, **kwargs):
    """Train a LSTM neural network on sequence data from a performance"""

    # Prepare voices
    primary_voice = Voice(primary_voice)
    secondary_voice = Voice(secondary_voice)

    if primary_voice.version < 2 or secondary_voice.version < 2:
        ctx.elog('Given voices were generated with an too old version.')

    sr = primary_voice.meta['samplerate']

    if sr != secondary_voice.meta['samplerate']:
        ctx.elog('Voices need same samplerates for correct training.')

    # Prepare model
    model_name = '{}.h5'.format(name)
    model_path = os.path.join(os.getcwd(), MODELS_FOLDER, model_name)
    resume = False

    if os.path.isfile(model_path):
        click.confirm(
            'Found model with same name! Do you want to resume training?',
            abort=True)
        resume = True

    # Parameters and Hyperparameters
    use_dynamics = kwargs.get('dynamics')
    use_durations = kwargs.get('durations')

    num_sound_classes = kwargs.get('num_classes')
    batch_size = kwargs.get('batch_size')
    data_split = kwargs.get('data_split')
    seq_len = kwargs.get('seq_len')
    dropout = kwargs.get('dropout')
    epochs = kwargs.get('epochs')
    num_layers = kwargs.get('num_layers')
    num_units = kwargs.get('num_units')

    # Calculate number of total classes
    num_classes = get_num_classes(num_sound_classes,
                                  use_dynamics,
                                  use_durations)

    ctx.log('\nParameters:')
    ctx.log(line(length=32))
    ctx.log('name:\t\t{}'.format(name))
    ctx.log('num_classes:\t{}'.format(num_classes))
    ctx.log('batch_size:\t{}'.format(batch_size))
    ctx.log('data_split:\t{}'.format(data_split))
    ctx.log('seq_len:\t{}'.format(seq_len))
    ctx.log('epochs:\t\t{}'.format(epochs))
    if not resume:
        ctx.log('num_layers:\t{}'.format(num_layers))
        ctx.log('num_units:\t{}'.format(num_units))
    ctx.log(line(length=32))
    ctx.log('')

    # Generate training data from voice sequences
    ctx.log(click.style('1. Generate training data from voices', bold=True))
    ctx.log('Primary voice: "{}"'.format(primary_voice.name))
    ctx.log('Secondary voice: "{}"'.format(secondary_voice.name))

    data = generate_sequence(ctx,
                             primary_voice,
                             secondary_voice,
                             save_sequence=kwargs.get('save_sequence'))

    ctx.log('')

    # Encode data before training
    ctx.log(click.style('2. Encode data before training', bold=True))
    encoded_data = encode_data(data,
                               num_sound_classes,
                               use_dynamics,
                               use_durations,
                               sr)

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

        num_model_classes = model.layers[-1].output_shape[1]
        if num_model_classes != num_classes:
            ctx.elog('The given model was trained with a different '
                     'amount of classes: given {}, but '
                     'should be {}.'.format(num_classes,
                                            num_model_classes))
    else:
        model = Sequential()
        model.add(layers.Embedding(input_dim=num_classes,
                                   output_dim=num_units,
                                   input_length=seq_len))
        for n in range(num_layers - 1):
            model.add(layers.LSTM(num_units, return_sequences=True))
            if dropout > 0.0:
                model.add(layers.Dropout(dropout))
        model.add(layers.LSTM(num_units))
        if dropout > 0.0:
            model.add(layers.Dropout(dropout))
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
    score = 0
    total = 0

    for i in range((training_steps // batch_size)):
        # Predict point from model
        samples, targets = next(test_gen)
        results = model.predict(samples)

        for j, result in enumerate(results):
            result_class = np.argmax(result)
            target_class = targets[j]

            if result_class == target_class:
                score += 1
            total += 1

    ratio = score / total

    ctx.log('Score: {0:.2f}%\n'.format(ratio * 100))

    # Save model
    ctx.log(click.style('7. Store model weights', bold=True))
    ctx.log('Stored weights at "{}"'.format(model_path))
    model.save(model_path)
    ctx.log('Done!')
