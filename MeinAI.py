from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from keras.utils import plot_model

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st

train_df = pd.read_csv('data/sign_mnist_train.csv')
train_df = train_df.sample(frac=1, random_state=42)
X, y = train_df.drop('label', axis=1), train_df['label']

X = X/255.0
X = tf.reshape(X, [-1, 28, 28, 1])

label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

X_train, X_valid = X[:25000], X[25000:]
y_train, y_valid = y[:25000], y[25000:]

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(24, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

save_best_cb = keras.callbacks.ModelCheckpoint('models/initial-end-to-end', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 5)

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), callbacks=[save_best_cb, early_stopping_cb])

with open('models/intial-end-to-end-history', 'wb') as history_file:
    pickle.dump(history.history, history_file)

best_model = keras.models.load_model('models/initial-end-to-end')


test_df = pd.read_csv('data/alphabet/sign_mnist_test.csv') # Load the test data
X_test, y_test = test_df.drop('label', axis=1), test_df['label']
X_test = tf.reshape(X_test, [-1, 28, 28, 1])
y_test = label_binarizer.transform(y_test)
best_model.evaluate(X_test, y_test)

def evaluate_model(model, X_test, y_test, label_binarizer):
    X_test_reshape = tf.reshape(X_test, [-1, 28, 28, 1])
    y_test_labels = label_binarizer.transform(y_test)
    results = model.evaluate(X_test_reshape, y_test_labels)
    print(f'Loss: {results[0]:.3f} Accuracy: {results[1]:.3f}')

results = evaluate_model(best_model, test_df.drop('label', axis=1), test_df['label'], label_binarizer)

data_augmentation = keras.models.Sequential()
data_augmentation.add(keras.layers.RandomRotation(0.1, fill_mode='nearest', input_shape=(28, 28, 1)))
data_augmentation.add(keras.layers.RandomZoom((0.15, 0.2), fill_mode='nearest'))
data_augmentation.add(keras.layers.RandomTranslation(0.1, 0.1, fill_mode='nearest'))

def evaluate_model(model, X_test, y_test, label_binarizer):
    # label_binarizer: Used while preprocessing the train data
    X_test_reshape = tf.reshape(X_test, [-1, 28, 28, 1])
    y_test_labels = label_binarizer.transform(y_test)
    results = model.evaluate(X_test_reshape, y_test_labels)
    print(f'Loss: {results[0]:.3f} Accuracy: {results[1]:.3f}')

best_model = keras.models.load_model('models/experiment-dropout-0/')
evaluate_model(best_model, X_test, y_test, label_binarizer)


@st.cache(allow_output_mutation=True)
def get_best_model():
    best_model = keras.models.load_model('models/experiment-dropout-0')
    return best_model


@st.cache
def get_label_binarizer():
    train_df = pd.read_csv('data/alphabet/sign_mnist_train.csv')
    y = train_df['label']
    label_binarizer = LabelBinarizer()
    y = label_binarizer.fit_transform(y)
    return label_binarizer


best_model = get_best_model()
label_binarizer = get_label_binarizer()

def preprocess_image(image, image_file, best_model, label_binarizer):
    image = tf.reshape(image, [image.shape[0], image.shape[1], 1])
    image = image/255
    image = tf.image.resize(image, [28, 28],       preserve_aspect_ratio=True)
    preprocessed_image = np.ones((1, 28, 28, 1))
    preprocessed_image[0, :image.shape[0], :image.shape[1], :] = image
    prediction = best_model.predict(preprocessed_image)
    index_to_letter_map = {i:chr(ord('a') + i) for i in range(26)}
    letter = index_to_letter_map[label_binarizer.inverse_transform(prediction)[0]]
    return letter