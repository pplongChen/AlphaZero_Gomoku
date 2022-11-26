from __future__ import print_function
import pickle

import random
import numpy as np
import os
import sys
import datetime

# Loading Config
if os.path.exists('config.py'):
  from config import *
pass

from game import Board, Game
from algo_alphaZero import AlphaZero_Player as ALGO_AlphaZero
from algo_human import Algo_Human


def run():
  n = PLAY_N_IN_ROW
  width, height = PLAY_BOARD_WIDTH, PLAY_BOARD_HEIGHT
  model_file = PLAY_MODEL_PATH

  board = Board(width=width, height=height, n_in_row=n)
  game = Game(board)


  alphaZero_player = ALGO_AlphaZero(    c_puct=PLAY_C_PUCT,
                           n_playout=PLAY_ALPHAZERO_SIM_TIMES,
                           model_path=model_file
                           ) 

  # human player, input your move in the format: 2,3
  human_player = Algo_Human()

  # set start_player=0 for human first
  # set start_player=1 for alphaZero_player first
  game.start_play(human_player, alphaZero_player, start_player = PLAY_START_PLAYER, is_shown = PLAY_IS_SHOWN)

pass

if __name__ == '__main__':
  run()
pass