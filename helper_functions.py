import os
import sys
import warnings

import librosa
import librosa.display
# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
from tqdm import tqdm

# to play the audio files

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

Tess = os.path.join(os.path.dirname(os.path.realpath(__file__)), "TESS/")
Crema = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CREMA/")
my_audio = os.path.join(os.path.dirname(os.path.realpath(__file__)), "my_audio/")


def get_my_audio() -> pandas.DataFrame:
    my_audio_files = os.listdir(my_audio)

    file_emotion = []
    file_path = []

    for file in my_audio_files:
        # storing file paths
        file_path.append(my_audio + file)
        # storing file emotions
        if "sad" in file:
            file_emotion.append('sad')
        elif "ang" in file:
            file_emotion.append('angry')
        elif "hap" in file:
            file_emotion.append('happy')
        elif "neu" in file:
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    my_audio_df = pd.concat([emotion_df, path_df], axis=1)

    return my_audio_df


def get_crema_df() -> pandas.DataFrame:
    crema_directory_list = os.listdir(Crema)

    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        # storing file paths
        file_path.append(Crema + file)
        # storing file emotions
        part = file.split('_')

        if len(part) < 3:
            continue
        if part[2] == 'SAD':
            file_emotion.append('sad')
        elif part[2] == 'ANG':
            file_emotion.append('angry')
        elif part[2] == 'DIS':
            file_emotion.append('disgust')
        elif part[2] == 'FEA':
            file_emotion.append('fear')
        elif part[2] == 'HAP':
            file_emotion.append('happy')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    crema_df = pd.concat([emotion_df, path_df], axis=1)

    return crema_df


def get_tess_df() -> pandas.DataFrame:
    tess_directory_list = os.listdir(Tess)

    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        if "DS_St" in dir:
            continue
        directories = os.listdir(Tess + dir)
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part == 'ps':
                file_emotion.append('surprise')
            else:
                file_emotion.append(part)
            file_path.append(Tess + dir + '/' + file)

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    tess_df = pd.concat([emotion_df, path_df], axis=1)

    return tess_df


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def extract_features(data, sample_rate):
    stft = np.abs(librosa.stft(data))
    result = np.array([])

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))  # stacking vertically

    return result


def prepare_for_prediction(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    return np.array(extract_features(data, sample_rate))


emotions = ["angry", "neutral", "disgust", "sad", "fear", "happy", "surprise"]


def encode_emotion(emotion):
    return emotions.index(emotion)


def decode_emotion(index):
    return emotions[index]


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def add_features(X, Y, df):
    for path, emotion in tqdm(df):
        feature = get_features(path)
        for ele in feature:
            X.append(ele)
            # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
            Y.append(emotion)


def get_best_model(clf, x_train, y_train, x_text, y_test):
    clf.fit(x_train, y_train)

    return clf


def get_prediction(clf, x_test):
    return clf.predict(x_test)


def create_waveplot(data, sr, e, figsize=(10, 2)):
    plt.figure(figsize=figsize)
    plt.title('Waveplot for audio with {} emotion'.format(e), size=15)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def create_spectrogram(data, sr, e, figsize=(10, 2)):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=figsize)
    plt.title('Spectrogram for audio with {} emotion'.format(e), size=15)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()


def create_images(paths, emotions, figsize=(10, 5)):
    fig, axs = plt.subplots(3 * len(paths), 1, figsize=figsize)

    for i in range(len(paths)):
        data, sampling_rate = librosa.load(paths[i])
        noised_data = noise(data)
        s_p_data = pitch(stretch(data), sampling_rate)
        X = librosa.stft(data)
        Xdb = librosa.amplitude_to_db(abs(X))

        print()

        axs[0 + i * (len(paths) + 1)].set_title('Emotion {}'.format(emotions[i]))
        librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz', ax=axs[0 + i * (len(paths) + 1)])

        X = librosa.stft(noised_data)
        Xdb = librosa.amplitude_to_db(abs(X))
        axs[1 + i * (len(paths) + 1)].set_title('Emotion {} with noise'.format(emotions[i]))
        librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz', ax=axs[1 + i * (len(paths) + 1)])

        X = librosa.stft(s_p_data)
        Xdb = librosa.amplitude_to_db(abs(X))
        axs[2 + i * (len(paths) + 1)].set_title('Emotion {} with stretch and pitch'.format(emotions[i]))
        librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='hz', ax=axs[2 + i * (len(paths) + 1)])

    for ax in axs.flat:
        ax.label_outer()


def predict_for_model(clf, scaler, my_audio_df, nn=False):
    for path, emotion in zip(my_audio_df.Path, my_audio_df.Emotions):
        print(path)
        print(emotion)
        features_my = [prepare_for_prediction(path)]
        # print(features_my)
        features_my = pd.DataFrame(features_my)
        features_my['labels'] = emotion
        # print(features_my.head)

        if nn:
            my_case_x = np.expand_dims(np.asarray(scaler.transform(features_my.iloc[:, :-1].values)), axis=2)
            prediction = clf.predict(my_case_x)
            print(np.round(prediction))
        else:
            my_case_x = scaler.transform(features_my.iloc[:, :-1].values)
            prediction = clf.predict_proba(my_case_x)
            print(prediction)
        # print(decode_emotion(prediction[0]))
