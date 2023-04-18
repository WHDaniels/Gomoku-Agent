from keras.models import load_model
import numpy as np


class MyPlayer:
    def __init__(self, id):
        self.id = id  # player 0 is the player that makes the first move.
        # add your neural network initialization here
        self.network = load_model('lstm_gomoku4.h5')

        self.boardRepresentation = [[(x, y)for y in range(11)] for x in range(11)]

    def get_move(self, board):
        # Given the current board configuration, return the move the players want to make. “board” is a (binary) numpy
        # array of the size [2, board_height, board_width]. (We fix board height and width to be 11.) Player 0’s
        # pieces are marked in board[0] and player 1’s pieces are in board[1]. You need to return x, y (the row and
        # col index) of the position where you want to put your piece. Make sure your move is legal, i.e.,
        # at location (x, y), there is no piece on the current board.

        x, y = 0, 0

        print(self.boardRepresentation)

        fullBoard = board[0] + board[1]
        fullBoardForPredict = np.array([fullBoard.reshape(-1, 121)] * 32, dtype=int).reshape((-1, 32, 121))

        print("fullBoard:", fullBoard.shape)
        nextBoard = self.network.predict([fullBoardForPredict])
        print("nextBoard:", nextBoard)

        result = np.argmax(nextBoard[0][0])
        reshapedFullBoard = fullBoard.reshape(-1, 121)
        # print("FSDFDSF", reshapedFullBoard[0])
        while reshapedFullBoard[0][result] != 0:
            print("OW!")
            nextBoard[0][0][result] = 0
            result = np.argmax(nextBoard[0][0])


        n = 0
        found = False
        for row in self.boardRepresentation:
            if found is True:
                break
            for tup in row:
                print(row)
                if n == result:
                    x, y = tup[0], tup[1]
                    found = True
                    break
                n += 1


        """
        nextMove = nextBoard - fullBoard
        print("nextMove:", nextMove)
        for m, row in enumerate(nextMove):
            for n, pos in enumerate(row):
                if pos == 1:
                    x, y = m, n
        """
        return x, y

    # def get_model(self):
    # Return the keras neural network model you use.
    #    return m


if __name__ == '__main__':
    print("Hello")
