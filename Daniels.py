# Model link: https://drive.google.com/file/d/1rEBSLEbnUjV9q6kYiuGqXGXLC_GCfD2X/view?usp=sharing

from keras.models import load_model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


class MyPlayer:
    def __init__(self, id):
        self.id = id  # player 0 is the player that makes the first move.
        # add your neural network initialization here
        self.network = load_model('Daniels.h5')

        self.boardRepresentation = [[(x, y) for y in range(11)] for x in range(11)]

    def get_move(self, board):
        # Given the current board configuration, return the move the players want to make. “board” is a (binary) numpy
        # array of the size [2, board_height, board_width]. (We fix board height and width to be 11.) Player 0’s
        # pieces are marked in board[0] and player 1’s pieces are in board[1]. You need to return x, y (the row and
        # col index) of the position where you want to put your piece. Make sure your move is legal, i.e.,
        # at location (x, y), there is no piece on the current board.

        x, y = 0, 0

        fullBoard = board[0] + board[1]
        fullBoardForPredict = np.array([fullBoard.reshape(-1, 121)] * 32, dtype=int).reshape((-1, 32, 121))

        nextBoard = self.network.predict([fullBoardForPredict])

        # print(fullBoardForPredict.shape)
        result = np.argmax(nextBoard[0][0])
        reshapedFullBoard = fullBoard.reshape(-1, 121)
        while reshapedFullBoard[0][result] != 0:
            nextBoard[0][0][result] = 0
            result = np.argmax(nextBoard[0][0])

        n = 0
        found = False
        for row in self.boardRepresentation:
            if found is True:
                break
            for tup in row:
                if n == result:
                    x, y = tup[0], tup[1]
                    found = True
                    break
                n += 1

        return x, y

    def get_model(self):
        # Return the keras neural network model you use.
        inputShape = (31, 121)
        opt = Adam(learning_rate=5e-2)

        m = Sequential()
        m.add(Masking(mask_value=np.zeros(121, dtype=int), input_shape=inputShape))
        m.add(LSTM(128, return_sequences=True))
        m.add(Dense(121, activation='relu'))

        m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return m


if __name__ == '__main__':
    print("Hello")
