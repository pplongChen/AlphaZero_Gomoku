from __future__ import print_function

import random
import numpy as np
import os
import sys
import datetime
import torch

from collections import defaultdict, deque
from tqdm import tqdm

from game import Board, Game
from algo_pure_mcts import PureMCTSPlayer as ALGO_Pure_MCTS
from algo_alphaZero import AlphaZero_Player as ALGO_AlphaZero

def show_time_now():
  return f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
pass

# Loading Config
if os.path.exists('config.py'):
  from config import *
pass

if not os.path.exists(TRAINED_MODELS_DIR):
  os.makedirs(TRAINED_MODELS_DIR)
pass

# re-factor done
class TrainProcedure():
  def __init__(self, init_model=None):

    ####################################
    # training params

    # adaptively adjust the learning rate based on KL
    self.lr_multiplier = LR_MULTIPLIER  

    # num of simulations used for the pure mcts, which is used as
    # the opponent to evaluate the trained policy
    self.pure_mcts_playout_num = DEFAULT_PURE_MCTS_PLAYOUT_NUM

    self.best_win_ratio = 0.0

    ####################################
    # AI and game setting

    self.board = Board(width    = BOARD_WIDTH,
                       height   = BOARD_HEIGHT,
                       n_in_row = N_IN_ROW)
    
    self.game = Game(self.board)

    self.data_buffer = deque(maxlen=BUFFER_SIZE)

    # ref: https://blog.csdn.net/weixin_40522801/article/details/106563354  
    if init_model:
      self.alpha_player = ALGO_AlphaZero( c_puct=C_PUCT,
                                  n_playout=N_ALPHAZERO_SIMULATE)
      self.alpha_player.link_load_model(init_model)
    else:
      self.alpha_player = ALGO_AlphaZero( c_puct=C_PUCT,
                                  n_playout=N_ALPHAZERO_SIMULATE)
    pass

    self.pure_mcts_player = ALGO_Pure_MCTS(c_puct=C_PUCT,
                               n_simulate=self.pure_mcts_playout_num)

  pass
pass

# re-factor done
def Train_expand_samples(self, play_data):
  """augment the data set by rotation and flipping
  play_data: [(state, mcts_prob, winner_z), ..., ...]
  """
  expand_data = []
  for state, mcts_porb, winner in play_data:
    for i in [1, 2, 3, 4]:
      # 90度翻轉
      expanded_equal_state = np.array([np.rot90(s, i) for s in state])
      expanded_equal_mcts_prob = np.rot90(np.flipud(
          mcts_porb.reshape(BOARD_HEIGHT, BOARD_WIDTH)), i)
      
      # 將90度翻轉的資料，存到陣列裡
      expand_data.append((expanded_equal_state,
                          np.flipud(expanded_equal_mcts_prob).flatten(),
                          winner))
      # 針對這個90度翻轉，再一次水平翻轉
      expanded_equal_state = np.array([np.fliplr(s) for s in expanded_equal_state])
      expanded_equal_mcts_prob = np.fliplr(expanded_equal_mcts_prob)

      # 將水平翻轉的資料，存到陣列裡
      expand_data.append((expanded_equal_state,
                          np.flipud(expanded_equal_mcts_prob).flatten(),
                          winner))
    pass
  pass
  return expand_data
pass
TrainProcedure.expand_samples=Train_expand_samples

# re-factor done
def Train_run_self_play(self, ai_player, in_game, is_shown=0, temp=1e-3):
  """ start a self-play game using a MCTS player, reuse the search tree,
  and store the self-play data: (state, mcts_probs, z) for training
  """
  in_game.board.init_board()
  p1, p2 = in_game.board.players
  states, mcts_probs, current_players = [], [], []

  end    = None
  winner = None
  while True:
    # 請玩家下一個決策(move)，以及決策依據(move_probs)
    # input: 
    #  1. 盤面: in_game.board
    #  2. 決策噪音(模擬現實生活): temp
    #  3. 回傳的依據(機率)總和必須是1: return_prob=1
    # output:
    #  1. 決策(move)
    #  2. 決策依據(move_probs)，即盤面上每一點的權重，我們可以把這個權重當作勝率
    move, move_probs = ai_player.get_action(in_game.board,
                                         temp=temp,
                                         is_self_play=True)
    
    # store the data: 
    # 1.盤面(in_game.board.current_state())
    # 2.決策依據(move_probs)
    # 3.目前玩家是誰(1: 1號玩家，2: 2號玩家)
    states.append(in_game.board.current_state())
    mcts_probs.append(move_probs)
    current_players.append(in_game.board.current_player)

    # perform a move
    in_game.board.do_move(move)
    if is_shown:
      in_game.graphic(in_game.board, p1, p2)
    pass
    
    # 回傳目前遊戲是否終止，如果終止，贏家是誰
    end, winner = in_game.board.game_end()

    # 遊戲終止的話，離開迴圈
    if end:
      break
    pass # if end
  pass # while end

  # winner from the perspective of the current player of each state
  winners_z = np.zeros(len(current_players))
  if winner != -1:
    winners_z[np.array(current_players) == winner] = 1.0
    winners_z[np.array(current_players) != winner] = -1.0
  pass

  # reset alphazero decision tree
  self.alpha_player.link_reset()

  if is_shown:
    if winner != -1:
        print("Game end. Winner is player:", winner)
    else:
        print("Game end. Tie")
    pass
  pass

  # 將set轉成list格式
  # zip([1,2,3], [11,22,33], [111,222,333]]
  # => [(1,11,111),(2,22,222),(3,33,333)]

  # list([(1,11,111),(2,22,222),(3,33,333)])[:]
  # => [[1,11,111],[2,22,222],[3,33,333]]
  play_record = list(zip(states, mcts_probs, winners_z))[:]

  return winner, play_record
pass
TrainProcedure.run_self_play=Train_run_self_play

# re-factor done
def Train_train_policy(self):
    
    # Step 1. 從歷史資料中，抽樣出來訓練
    mini_batch = random.sample(self.data_buffer, BATCH_SIZE)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]

    # 彙整成一個samples queue
    samples={}
    samples["state_batch"]=state_batch
    samples["mcts_probs_batch"]=mcts_probs_batch
    samples["winner_batch"]=winner_batch

    # Step 2. 開始訓練

    # 2.1. 先計算舊的模型所推估的機率分布
    old_move_probs, old_win_trend = self.alpha_player.link_predict(state_batch)

    # 2.2. 設定新的學習速度，可根據KL動態調整
    _new_learn_rate  = LEARN_RATE * self.lr_multiplier

    ####################
    # 1. KL-D / KL 是什麼
    #
    # Kullback-Leibler divergence (KL-D)，俗稱KL距離，常用來衡量兩個概率分佈的距離。
    #
    # 常用來表示目前AI推測的機率，離事實機率有多遠，也就是目前的AI最多還可以下降多少。
    # 
    # KL的單位叫做nats，如果使用nats當單位在計算上會方便許多，因為許多的分布都可以表示成以e為底的指數，例如：Normal Distribution。
    # 
    # 2. 如何計算KL-D
    # 
    # 比如有四個類別，
    # 
    # AI 得到四個類別的【推測機率】分別是 [ 0.1, 0.2, 0.3, 0.4 ] 
    # 
    # 目前抽樣得到四個類別的【事實機率】分別是 [ 0.4, 0.3, 0.2, 0.1]
    # 
    # 那麼AI的【推測機率】離【事實機率】的 KL-Distance(【推測機率】,【事實機率】)
    #                               =0.1*log(0.1/0.4)+0.2*log(0.2/0.3)+0.3*log(0.3/0.2)+0.4*log(0.4/0.1)=0.1982271233
    # 
    # 3. ref.
    #  https://www.ycc.idv.tw/deep-dl_1.html
    #  https://www.ycc.idv.tw/deep-dl_2.html
    #  https://www.ycc.idv.tw/deep-dl_3.html
    ####################
    
    # 2.3. 一個樣本訓練EPOCHS次，加快學習速度
    for i in range(EPOCHS):

      # 2.3.1. 訓練一次 (包含backward操作)
      loss, entropy = self.alpha_player.link_train(
                           samples,
                           _new_learn_rate)

      # 2.3.2. 計算新的模型所推估的機率分布
      new_move_probs, new_win_trend = self.alpha_player.link_predict(state_batch)

      # 2.3.3. 新舊模型的機率分布距離(KL距離)
      kl_between_new_old = np.mean(np.sum(old_move_probs * (
              np.log(old_move_probs + 1e-10) - np.log(new_move_probs + 1e-10)),
              axis=1)
      )

      # 2.3.4. 如果新舊模型的機率分布距離超出預期，代表學習方向歪了，趕緊跳出迴圈，不要再訓練
      if kl_between_new_old > MAX_KL_IN_ONE_LEARNING: 
          break
      pass

    pass

    # 2.4. 動態調整學習速度
    if ( kl_between_new_old > KL_TARG_UPPER_BOUND ) and ( self.lr_multiplier > LR_MULTIPLIER_LOWER_BOUND ):
      # 2.4.1. 如果新舊模型機率分布超過上限，且學習速度高過上限，則降速1.5倍
      self.lr_multiplier = self.lr_multiplier / LR_MODIFY_RATE
    elif kl_between_new_old < KL_TARG_LOWER_BOUND and self.lr_multiplier < LR_MULTIPLIER_UPPER_BOUND:
      # 2.4.2. 如果新舊模型機率分布低於下限，且學習速度低於上限，則加速1.5倍
      self.lr_multiplier = self.lr_multiplier * LR_MODIFY_RATE
    pass

    # 2.5. log紀錄

    # 舊的勝率傾向
    explained_old_win_trend = (1 -
                         np.var(np.array(winner_batch) - old_win_trend.flatten()) /
                         np.var(np.array(winner_batch)))

    # 新的勝率傾向
    explained_new_win_trend = (1 -
                         np.var(np.array(winner_batch) - new_win_trend.flatten()) /
                         np.var(np.array(winner_batch)))

    _log=("time, {}, kl_between_new_old,{:.5f},"
           "lr_multiplier,{:.3f},"
           "loss,{},"
           "entropy,{},"
           "explained_old_win_trend,{:.3f},"
           "explained_new_win_trend,{:.3f}"
           ).format(show_time_now(), kl_between_new_old,
                    self.lr_multiplier,
                    loss,
                    entropy,
                    explained_old_win_trend,
                    explained_new_win_trend)

    return loss, entropy, _log
pass
TrainProcedure.train_policy=Train_train_policy

# re-factor done
def Train_policy_evaluate(self, n_games=10):
  """
  Evaluate the trained policy by playing against the pure MCTS player
  Note: this is only for monitoring the progress of training
  """

  win_cnt = defaultdict(int)
  
  for i in range(n_games):
      winner = self.game.start_play( self.alpha_player,
                                     self.pure_mcts_player,
                                    start_player=i % 2,
                                    is_shown=0)
      win_cnt[winner] += 1
  pass
  
  win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games

  _log = f"# time, {show_time_now()}, num_playouts,{self.pure_mcts_playout_num}, win_ratio, {win_ratio}, win, {win_cnt[1]}, lose, {win_cnt[2]}, tie,{win_cnt[-1]}"

  return win_ratio, _log
pass
TrainProcedure.policy_evaluate=Train_policy_evaluate

# re-factor done
def Train_run(self):

  global IS_START_TRAINING

  bar = tqdm(range(GAME_BATCH_NUM), file=sys.stdout)
  for i in bar: # start training

    ####################
    # step 1. 自我對弈 PLAY_BATCH_SIZE 次
    for _ in range(PLAY_BATCH_SIZE):
      # 取得一局自我對弈紀錄：包含 贏家(winner) 與 棋譜(play_record)
      winner, play_record = self.run_self_play(  self.alpha_player, self.game, 
                                                  temp=TEMPERATURE)
      # 目前這一局遊戲樣本總共走了幾步
      self.episode_len = len(play_record)

      # 將現在的遊戲樣本，水平翻轉，90度翻轉，增加樣本數
      expand_play_record = self.expand_samples(play_record)

      # 將自我訓練的資料放到queue裡
      self.data_buffer.extend(expand_play_record)
    pass
    
    _msg = f"games: {i}, episode_len:{self.episode_len}"
    print( f" => {_msg}",end='')
    
    ####################
    # step 2. 開始訓練
    if len(self.data_buffer) > BATCH_SIZE:

      if not IS_START_TRAINING: 
        print("\n\n####  start training  #####\n\n")
        IS_START_TRAINING=True
      pass

      loss, entropy, _log = self.train_policy()

      _log = f"game_index,{i},{_log}"
      print(_log, file=open(LOG_FILE_TRAINING, "a"))

    pass

    ####################
    # step 3. 存檔，並檢查目前模型的能力，
    if (i) % CHECK_FREQ == 0:
      
      # Step 3.1. 先存檔
      self.alpha_player.link_save_model( f'{TRAINED_MODELS_DIR}/current_policy_at_{i}.model')

      # Step 3.2. 評估目前模型能力
      
      # Step 3.2.1. 計算勝率
      win_ratio, _log = self.policy_evaluate()
      _log = f"game_index,{i},{_log}"

      print("\n ## EVALUATION: current self-play game index: {}".format(i))
      print(f"\n{_log}")
      print(_log, file=open(LOG_FILE_EVALUATION, "a"))

      # Step 3.2.2. 如果勝率比以前高，則存檔模型，然後提高mcts的難度
      if win_ratio > self.best_win_ratio:
          
        print(f"New best policy at {i+1} vs mcts {self.pure_mcts_playout_num}")

        
        # update the best_policy
        self.best_win_ratio = win_ratio

        # 3.2.2.1. 存檔模型
        self.alpha_player.link_save_model( f'{TRAINED_MODELS_DIR}/best_policy_at_{i}_vs_mcts_{self.pure_mcts_playout_num}.model')

        # 3.2.2.2. 提高mcts的難度
        if (self.best_win_ratio >= MODEL_UPDATE_WIN_RATIO ):
          self.pure_mcts_playout_num += MODEL_UPDATE_SCALE
          self.best_win_ratio = 0.0
        pass

      pass

      # Step r3.2. 如果已經達到目標強度，停止訓練
      if (self.pure_mcts_playout_num > MODEL_UPDATE_MAX_SCALE):
        print(f"############################################")
        print(f"\n\n # finish training at game {i+1} for mcts scale {self.pure_mcts_playout_num}")
        print(f"############################################")
        break
      pass

    pass # end of step 3.

  pass # end training
pass
TrainProcedure.run=Train_run


if __name__ == '__main__':
  train_process = TrainProcedure()
  train_process.run()
pass