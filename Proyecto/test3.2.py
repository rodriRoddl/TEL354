import numpy as np
import os
from pathlib import Path
import sounddevice as sd
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore

# Función para capturar audio
def record_audio(duration=2, sr=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("Recording complete")
    return np.squeeze(audio)

# Función para extraer características MFCC del audio
def extract_features(audio, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs.T

# Preparar datos de entrenamiento
def prepare_data(file_paths, labels, sr=22050, n_mfcc=13, max_pad_len=87):
    features = []
    for file_path in file_paths:
        audio, sr = librosa.load(file_path, sr=sr)
        mfccs = extract_features(audio, sr, n_mfcc)
        if mfccs.shape[0] > max_pad_len:
            mfccs = mfccs[:max_pad_len]
        else:
            pad_width = max_pad_len - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        features.append(mfccs)
    features = np.array(features)
    return features, np.array(labels)

# Definir el modelo de red neuronal
def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Cargar y preparar datos de audio
file_paths = []
labels = []
voice_filenames = os.listdir("audio-dataset")
for name in voice_filenames:
    dir_path = Path("audio-dataset") / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    file_paths += speaker_sample_paths
    labels += [name] * len(speaker_sample_paths)

# Asegúrate de tener suficientes datos para cada persona
max_pad_len = 87  # Ajusta según tus datos
features, labels = prepare_data(file_paths, labels, max_pad_len=max_pad_len)

# Normalizar características
scaler = StandardScaler()
features = scaler.fit_transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)

# Codificar etiquetas
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = create_model((X_train.shape[1], X_train.shape[2]), num_classes=labels_categorical.shape[1])
model.fit(X_train, y_train, epochs=15, batch_size=8, validation_data=(X_test, y_test))

# Función para predecir el hablante
def recognize_speaker(audio, sr=22050):
    features = extract_features(audio, sr).reshape(1, -1, 13)
    features = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
    features_padded = np.zeros((1, max_pad_len, 13))
    if features.shape[1] > max_pad_len:
        features_padded[0] = features[0, :max_pad_len]
    else:
        features_padded[0, :features.shape[1]] = features[0]
    prediction = model.predict(features_padded)
    label_index = np.argmax(prediction)
    label = label_encoder.inverse_transform([label_index])
    return label[0]

# Grabación y reconocimiento en tiempo real
audio = record_audio(duration=2)
recognized_speaker = recognize_speaker(audio)
print(f"Recognized speaker: {recognized_speaker}")
