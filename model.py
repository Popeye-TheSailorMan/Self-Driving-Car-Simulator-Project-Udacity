import csv
from pathlib import Path

import cv2
import matplotlib
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import PIL

matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import json
import matplotlib.image as mpimg

lines = []
with open('C:\\Users\\Dell\\Downloads\\Machine Learning\\NUERAL NETWORKS OR PERSEPTRONS\\Driving data\\driving_log.csv') as file:
    reader = csv.reader(file)
    # next(reader) -> only if the first line is unusable
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    while True:
        shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images, measurements = [], []
            # we do this to use the center camera and the side cameras
            for batch_sample in batch_samples:
                for i in range(3):
                    # path has backslashes because of Windows
                    filename = batch_sample[i].split('\\')[-1]
                    current_path = 'C:\\Users\\Dell\\Downloads\\Machine Learning\\NUERAL NETWORKS OR PERSEPTRONS\\Driving data\\IMG/' + filename
                    image = mpimg.imread(current_path)
                    image = preprocess_img(image)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    correction = 0.3
                    if i == 1:
                        # left camera with correction
                        measurement += correction
                    if i == 2:
                        # right camera with correction
                        measurement -= correction
                    measurements.append(measurement)
                    # if i == 0:
                    #     # augment data for center image
                    #     images.append(np.fliplr(image))
                    #     measurements.append(measurement * -1.0)

            x_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(x_train, y_train)


def nvidia_cnn(dropout=0.3):
    """
    :param dropout: dropout rate
    :return: Model Architecture (based on NVIDIA paper)
    """
    input_shape = (40, 160, 3)

    model = Sequential()
    model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape=input_shape, output_shape=input_shape))
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", data_format="channels_first"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    model.summary()

    return model


def preprocess_img(image):
    """
    :param image: taken full res image
    :return: image with shape of (40, 160, 3) and YUV color space
    """
    # resize to half of the original size (320x160 to 160x80)
    h, w = image.shape[:2]
    ratio = (w * 0.5) / w
    dim = (int(w * 0.5), int(h * ratio))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # crop image to 160x40 (take away 30px from top and 10px from bottom) so we only have the important parts
    image = image[30:70, 0:160]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


def load_trained_model(dropout=0.3):
    model = nvidia_cnn(dropout=dropout)
    # load weights into new model
    model.load_weights("model.h5")
    return model


def save_and_visualize(model, history_obj):
    # Save model
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)
    print('Saved model!')

    # Save model data visualization
    plt.plot(history_obj.history['loss'])
    plt.plot(history_obj.history['val_loss'])
    plt.title('Mean squared error loss')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('model.png')
    print('Saved visualization data!')


def train_model(dropout, epochs=3, batch_size=32, learning_rate=0.01):
    model = nvidia_cnn(dropout=dropout)

    if Path("model.h5").is_file():
        model = load_trained_model(dropout=dropout)

    model.compile(loss='mse', optimizer=Adam(learning_rate))

    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2),
        ModelCheckpoint(filepath='model.h5', save_best_only=True, monitor='val_loss')
    ]

    history_obj = model.fit_generator(train_generator, steps_per_epoch=int((len(train_samples) * 3) / batch_size),
                                      validation_data=validation_generator,
                                      validation_steps=int((len(validation_samples) * 3) / batch_size),
                                      epochs=epochs, callbacks=callbacks)

    save_and_visualize(model, history_obj)


BATCH_SIZE = 64
EPOCHS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001

train_model(dropout=DROPOUT, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
