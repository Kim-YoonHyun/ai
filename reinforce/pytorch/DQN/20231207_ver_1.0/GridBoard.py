import numpy as np
import random
import sys

def rand_pair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


def add_tuple(tuple1, tuple2):
    ary_sum = np.add(tuple1, tuple2)
    result_tuple = tuple(ary_sum)
    return result_tuple
    # return tuple([sum(x) for x in zip(a,b)])


class BoardPiece:
    def __init__(self, name, code, pos):
        self.name = name #name of the piece
        self.code = code #an ASCII character to display on the board
        self.pos = pos #2-tuple e.g. (1,4)


class BoardMask:
    def __init__(self, name, mask, code):
        self.name = name
        self.mask = mask
        self.code = code

    def get_positions(self): #returns tuple of arrays
        return np.nonzero(self.mask)


def zip_positions2d(positions): #positions is tuple of two arrays
    x,y = positions
    return list(zip(x,y))


class GridBoard:
    def __init__(self, size=4):
        self.size = size #Board dimensions, e.g. 4 x 4
        self.component_dict = {} #name : board piece
        self.masks = {}


    def add_piece(self, name, code, position=(0, 0)):
        New_Piece = BoardPiece(name, code, position)
        self.component_dict[name] = New_Piece


    #basically a set of boundary elements
    def addMask(self, name, mask, code):
        #mask is a 2D-numpy array with 1s where the boundary elements are
        New_Mask = BoardMask(name, mask, code)
        self.masks[name] = New_Mask


    def move_piece(self, name, pos):
        move = True
        for _, mask in self.masks.items():
            if pos in zip_positions2d(mask.get_positions()):
                move = False
        if move:
            self.component_dict[name].pos = pos

    def delPiece(self, name):
        del self.component_dict['name']


    def render(self):
        dtype = '<U2'
        displ_board = np.zeros((self.size, self.size), dtype=dtype)
        displ_board[:] = ' '
        for name, piece in self.component_dict.items():
            displ_board[piece.pos] = piece.code
        for name, mask in self.masks.items():
            displ_board[mask.get_positions()] = mask.code
        return displ_board


    def render_np(self):
        num_pieces = len(self.component_dict) + len(self.masks)
        displ_board = np.zeros((num_pieces, self.size, self.size), dtype=np.uint8)
        layer = 0
        for name, piece in self.component_dict.items():
            pos = (layer,) + piece.pos
            displ_board[pos] = 1
            layer += 1

        for name, mask in self.masks.items():
            x,y = self.masks['boundary'].get_positions()
            z = np.repeat(layer,len(x))
            a = (z,x,y)
            displ_board[a] = 1
            layer += 1
        return displ_board
