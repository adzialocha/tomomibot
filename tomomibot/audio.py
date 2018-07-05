import threading

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import librosa
import numpy as np
import pyaudio


HOP_LENGTH = 512
MSE_FRAME_LENGTH = 2048
MIN_SAMPLE_MS = 100


def pca(features, components=2):
    pca = PCA(n_components=components)
    transformed = pca.fit(features).transform(features)
    variance = np.cumsum(
        np.round(pca.explained_variance_ratio_, decimals=3) * 100)
    scaler = MinMaxScaler()
    scaler.fit(transformed)
    return scaler.transform(transformed), variance, pca, scaler


def mfcc_features(y, sr, n_mels=128, n_mfcc=13):
    # Analyze only first second
    y = y[0:sr]

    # Calculate MFCCs (Mel-Frequency Cepstral Coefficients)
    mel_spectrum = librosa.feature.melspectrogram(y,
                                                  sr=sr,
                                                  n_mels=n_mels)
    log_spectrum = librosa.amplitude_to_db(mel_spectrum,
                                           ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_spectrum,
                                sr=sr,
                                n_mfcc=n_mfcc)

    # Standardize feature for equal variance
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    feature_vector = np.concatenate((
        np.mean(mfcc, 1),
        np.mean(delta_mfcc, 1),
        np.mean(delta2_mfcc, 1)))
    feature_vector = (
        feature_vector - np.mean(feature_vector)
    ) / np.std(feature_vector)

    return feature_vector


def slice_audio(y, onsets, sr=44100, offset=0):
    frames = []
    for i in range(len(onsets) - 1):
        start = onsets[i] * HOP_LENGTH
        end = onsets[i + 1] * HOP_LENGTH
        if end - start > (sr // 1000) * MIN_SAMPLE_MS:
            frames.append([y[start:end], start + offset, end + offset])
    return frames


def detect_onsets(y, sr=44100, db_threshold=10):
    # Get the frame->beat strength profile
    onset_envelope = librosa.onset.onset_strength(y=y,
                                                  sr=sr,
                                                  hop_length=HOP_LENGTH,
                                                  aggregate=np.median)

    # Locate note onset events
    onsets = librosa.onset.onset_detect(y=y,
                                        sr=sr,
                                        onset_envelope=onset_envelope,
                                        hop_length=HOP_LENGTH,
                                        backtrack=True)

    # Convert frames to time
    times = librosa.frames_to_time(np.arange(len(onset_envelope)),
                                   sr=sr,
                                   hop_length=HOP_LENGTH)

    # Calculate MSE energy per frame
    mse = librosa.feature.rmse(y=y,
                               frame_length=MSE_FRAME_LENGTH,
                               hop_length=HOP_LENGTH) ** 2

    # Convert power to decibels
    db = librosa.power_to_db(mse.squeeze(),
                             ref=np.median)

    # Filter out onsets which signals are too low
    onsets = [o for o in onsets if db[o] > db_threshold]

    return onsets, times[onsets]


class AudioInput():
    """Records data from an audio source.

    # Properties
        rate
        buffersize
        input_index
    """

    def __init__(self, rate=44100, buffersize=1024, index_input=0):
        self.rate = rate
        self.buffersize = buffersize
        self.index_input = index_input

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paFloat32,
                                      channels=1,
                                      rate=self.rate,
                                      input=True,
                                      input_device_index=self.index_input,
                                      frames_per_buffer=self.buffersize,
                                      stream_callback=self.stream_callback)

        self.lock = threading.Lock()
        self.is_running = False
        self.frames = []

    def stream_callback(self, data, frame_count, time_info, status):
        data = np.fromstring(data, 'float32')
        with self.lock:
            self.frames.append(data)
            if not self.is_running:
                return None, pyaudio.paComplete
        return None, pyaudio.paContinue

    def read_frames(self):
        with self.lock:
            frames = self.frames
            self.frames = []
            return frames

    def start(self):
        self.is_running = True
        self.stream.start_stream()

    def stop(self):
        with self.lock:
            self.is_running = False
        self.stream.close()
        self.audio.terminate()
