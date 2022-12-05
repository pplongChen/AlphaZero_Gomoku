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
from algo_pure_mcts import PureMCTSPlayer as ALGO_Pure_MCTS
from algo_alphaZero import AlphaZero_Player as ALGO_AlphaZero
from algo_human import Algo_Human


def run():
  n = PLAY_N_IN_ROW
  width, height = PLAY_BOARD_WIDTH, PLAY_BOARD_HEIGHT
  model_file = PLAY_MODEL_PATH

  board = Board(width=width, height=height, n_in_row=n)
  game = Game(board)

  # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_simulate)
  mcts_player = ALGO_Pure_MCTS(c_puct = PLAY_C_PUCT, n_simulate=PLAY_MCTS_SIM_TIMES)

  # human player, input your move in the format: 2,3
  human_player = Algo_Human()

  # set start_player=0 for human first
  # set start_player=1 for mcts_player first
  game.start_play(human_player, mcts_player, start_player = PLAY_START_PLAYER, is_shown = PLAY_IS_SHOWN)

pass

if __name__ == '__main__':
  run()
pass