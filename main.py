from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import pickle

gamesList = np.array(pickle.load(open('data//gamesListSmall.p', 'rb')), dtype=object)


def getMaxM(X):
    mList = list()
    for n, entry in enumerate(np.array(X, dtype=object)):
        mList.append(entry.shape[0])
        # print("length:", entry.shape[0])
        # print(n)

    return max(mList)


def pad(X):
    newArray = list()
    maxLen = getMaxM(X)
    for entry in np.array(X, dtype=object):
        if entry.shape[0] < maxLen:
            amountToPad = getMaxM(X) - entry.shape[0]
            # for num in range(entry.shape[0], getMaxM(X)):
            entry = np.append(entry, [np.zeros((11, 11), dtype=int)] * amountToPad, axis=0)
        newArray.append(entry)
    return newArray


def preprocess(trainList):
    X = list()
    y = list()

    newTrainList = np.array(trainList, dtype=object)
    print(newTrainList.shape)
    print(np.array(newTrainList[0], dtype=object).shape)
    print(np.array(newTrainList[0][0], dtype=int).shape)

    for entry in newTrainList:
        play_X = list()
        play_y = list()

        for n, play in enumerate(entry):

            if n == 0 or n % 2 != 0:
                newPlay = np.where(play == -1, 0, play)
                play_X.append(newPlay)
                # print(newPlay)
                try:
                    if n == 0:
                        # print(entry[n+1])
                        new_yPlay = np.where(entry[n + 1] == -1, 0, entry[n + 1])
                        play_y.append(new_yPlay)
                        # print(new_yPlay)
                    else:
                        new_yPlay = np.where(entry[n + 2] == -1, 0, entry[n + 2])
                        play_y.append(new_yPlay)
                        # print("newP", new_yPlay)
                except IndexError:
                    play_y.append(np.zeros((11, 11), dtype=int))
                    # print("zeros", np.zeros((11, 11), dtype=int))
        # print(np.array(play_y, dtype=object).shape)
        X.append(np.array(play_X, dtype=object))
        y.append(np.array(play_y, dtype=object))
        # print(X[0])

    # print(len(X))
    # print(np.array(X, dtype=object).shape)
    # print(np.array(y, dtype=object).shape)
    # print(np.array(X[0], dtype=object).shape)
    # print(np.array(y[0], dtype=object).shape)
    # print(np.array(X[0][0], dtype=object).shape)
    # print(np.array(y[0][0], dtype=object).shape)
    # exit(0)

    new_X = pad(np.array(X, dtype=object))
    new_y = pad(np.array(y, dtype=object))

    return new_X, new_y


def createModel(inputShape, lr):
    # batch_size = 256
    # inputShape = (batch_size, inputShape[1], inputShape[2])
    # print(inputShape)

    inputShape = (32, 121)
    opt = Adam(learning_rate=5e-2)

    model = Sequential()
    model.add(Masking(mask_value=np.zeros(121, dtype=int), input_shape=inputShape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(121, activation='softmax'))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    """
    X, y = preprocess(gamesList)

    print(np.array(X, dtype=object).shape)
    print(np.array(X, dtype=object)[0].shape)
    for x in np.array(X, dtype=object):
        print(x.shape)

    print(y)

    pickle.dump(X, open('X.p', 'wb'))
    pickle.dump(y, open('y.p', 'wb'))
    """
    print("Start")

    X = np.array(pickle.load(open('data//gamesListSmall.p', 'rb')), dtype=object)
    y = np.array(pickle.load(open('y.p', 'rb')), dtype=object)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

    new_train_X = np.array(train_X, dtype=object)
    new_train_X = new_train_X.reshape((new_train_X.shape[0], new_train_X.shape[1], 121))
    print("new train x:", new_train_X.shape)

    new_train_y = np.array(train_y, dtype=object)
    new_train_y = new_train_y.reshape((new_train_y.shape[0], new_train_y.shape[1], 121))
    print("new train y:", new_train_y.shape)

    learning_rate = 1e-5

    print(new_train_X.shape)
    model = createModel(new_train_X.shape, learning_rate)

    hist = model.fit(new_train_X, new_train_y, epochs=10)
    # validation_data=(test_X, test_y))

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.show()
