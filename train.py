from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import numpy as np

MODEL = './apexile_test.npy'
LABELS = ['monkey', 'atsushi', 'onna', 'others']
NUM_LABELS = len(LABELS)
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCH = 200


def model_train(x_train, y_train):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                     input_shape=x_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_LABELS, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    callback = EarlyStopping(
        monitor='val_loss', patience=5, verbose=1, mode='auto')
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH,
              validation_split=0.1, callbacks=callback)

    return model


def model_eval(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test Loss: ', scores[0])
    print('Accuracy:  ', scores[1])
    loss = str(scores[0])[:4]
    accuracy = str(scores[1])[:4]
    model.summary()
    model.save('./model_loss{}_acc{}.h5'.format(loss, accuracy))


def main():
    x_train, x_test, y_train, y_test = np.load(MODEL, allow_pickle=True)
    x_train = x_train.astype('float') / 256
    x_test = x_test.astype('float') / 256
    y_train = np_utils.to_categorical(y_train, NUM_LABELS)
    y_test = np_utils.to_categorical(y_test, NUM_LABELS)

    model = model_train(x_train, y_train)
    model_eval(model, x_test, y_test)


if __name__ == '__main__':
    main()
