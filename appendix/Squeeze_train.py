import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.utils import np_utils
from keras_squeezenet import SqueezeNet
from keras.callbacks import EarlyStopping

MODEL = './imagefiles_224.npy'
LABELS = ['car', 'motorbike']
NUM_LABELS = len(LABELS)
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCH = 100


def model_train(x_train, y_train):
    model = SqueezeNet(weights='imagenet', include_top=False,
                       input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(NUM_LABELS, activation='softmax'))

    model = Model(inputs=model.input, outputs=top_model(model.output))

    for layer in model.layers[:15]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

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
