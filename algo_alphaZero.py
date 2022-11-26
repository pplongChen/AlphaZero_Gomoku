
import numpy as np
import copy
import os
from collections import OrderedDict

from node import TreeNode

# Loading Config
if os.path.exists('config.py'):
  from config import *
pass

def softmax(x):
  probs = np.exp(x - np.max(x))
  probs /= np.sum(probs)
  return probs
pass


class AlphaZero_MCTS(object):
  """An implementation of Monte Carlo Tree Search."""

  def __init__(self,  c_puct=5, n_playout=10000):
    """
    c_puct: a number in (0, inf) that controls how quickly exploration
        converges to the maximum-value policy. A higher value means
        relying on the prior more.
    """
    self._root = TreeNode(None, 1.0)
    self._c_puct = c_puct
    self._n_playout = n_playout

    # The main difference between principle and policy is 
    #  => that a principle is a rule that has to be followed 
    #  => while a policy is a guideline that can be adopted.
    self.decision_guideline = PolicyValueNet(board_width=BOARD_WIDTH, board_height=BOARD_HEIGHT)
  pass

  # refactor done
  def _simulate(self, state):
    """Run a single playout from the root to the leaf, getting a value at
    the leaf and propagating it back through its parents.
    State is modified in-place, so a copy must be provided.
    """
    # 1. 先走到leaf
    node = self._root
    while(1):
      if node.is_leaf():
          break
      # Greedily select next move.
      action, node = node.select(self._c_puct)
      state.do_move(action)
    pass

    # 2. 開始推估盤面贏的機會，先初始化win_trend變數，用來儲存盤面贏的機率

    win_trend = None

    # 2.1. AlphaZero 推估盤面贏的機率
    #  Input: 盤面
    #  Output: 
    #    a) 每一步贏的權重(predicted_action_weights)矩陣: [width*height]
    #    b) 目前盤面贏的機率(predicted_win_trend) :[1x1]

    # ps. pure mcts是隨機玩完一場局，來找出贏的狀況，且只有1和-1兩種可能
    #     alphaZero是根據目前盤面，推測出目前盤面贏的機率，是[-1.0~1.0]的可能
    #     alphaZero不會像pure mcts隨機玩完一場局，而是只用目前的盤面推估贏的機率
    predicted_action_weights, predicted_win_trend = self.decision_guideline._get_weight_of_available_move(state)

    # 2.2. 設定盤面贏的機率 

    end, winner = state.game_end()
    if not end:
      # 2.2.1. 如果目前的盤面還可以走，則win_trend使用AlphaZero的推估盤面贏的機率
      win_trend = predicted_win_trend

      # 另外針對這個leaf，再深度擴展一層leaf，
      # 每一個leaf(每個落子點)贏的權重使用AlphaZero的推估每一步贏的權重(predicted_action_weights)矩陣
      node.expand(predicted_action_weights)
    else:
      # 2.2.2. 如果目前的盤面已經有勝負，則捨棄AlphaZero的推估，改用實際勝負值當作win_trend
      if winner == -1:  # tie
        win_trend = 0.0
      else:
        win_trend = (
              1.0 if winner == state.get_current_player() else -1.0
          )
      pass
    pass

    # 3. 從leaf開始回頭更新每個節點的數據，一直更新到root
    node.back_propagation(-win_trend)
  pass

  # refactor done
  def _get_probs_of_available_move(self, state, temp=1e-3):
    # 將【造訪節點次數 visit】作為【權重】
    # 找出造訪次數最多的move，這個move就是最好的move
    # 跟人類一樣，思考最多次的決策，應該就是最好的決策

    # 這個函式可對應到 PURE_MCTS._get_move()

    # 將【造訪節點次數 visit】作為【權重】，並將【權重】轉成加總為1.0的機率分布
    # ref: Mastering the game of Go without human knowledge 
    # ( https://www.nature.com/articles/nature24270 )
    # 引述：
    #  MCTS may be viewed as a self-play algorithm that, 
    #  given neural network parameters θ and a root position s, 
    #  computes a vector of 【search probabilities recommending moves to play】, π = αθ(s), 
    #  proportional to 【the exponentiated visit count for each move】,
    #  πa~N(s, a)1/τ, where τ is a temperature parameter.

    # 1. 先跑self._n_playout次模擬，已更新root的children的資料
    for n in range(self._n_playout):
        # 複製盤面
        state_copy = copy.deepcopy(state)
        # 開始模擬
        self._simulate(state_copy)
    pass

    # 2. 將目前root的children的資料欄位 key=可用下子處, 以及對應的 value = node的_n_visits欄位撈出來
    act_visits = [(act, node._n_visits)
                  for act, node in self._root._children.items()]

    # zip(*)的用法
    #    zip_data = [('1', '11', '111'), ('2', '22', '222')]
    #    list( zip(*zip_data) )
    #    => [('1', '2'), ('11', '22'), ('111', '222')]
    # ps. 要用list包在外面，將指標轉成陣列，因為zip(*data)跑出來只會是指標
    acts, visits = zip(*act_visits)

    # 3. 根據visits來推估每一個可用下子位置的獲勝機率(用softmax推估)
    #  ps. 使用softmax，可以使得每一個元素的範圍都在(0.0 ~ 1.0)之間，並且所有元素的和為1
    act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

    # 回傳action，以及每一個可用下子位置的獲勝機率
    return acts, act_probs
  pass

  # refactor done
  def reuse_decision_tree(self, in_move):

    if in_move in self._root._children:
        # reuse the tree, during the training
        # drop the root, use the root.childeren[move] as the root
        self._root = self._root._children[in_move]
        self._root._parent = None
    else: 
        assert False, " no such child when reuse decision tree "
    pass
  pass

  # refactor done
  def reset_decision_tree(self):
    self._root = TreeNode(None, 1.0)
  pass

  def __str__(self):
    return "AlphaZero_MCTS"
  pass
pass


class AlphaZero_Player(object):
  """AI player based on MCTS"""

  def __init__(self, 
               c_puct=5, n_playout=2000, model_path=None):
    self.c_puct = 5
    self.n_simulate = N_ALPHAZERO_SIMULATE
    self.brain = AlphaZero_MCTS( c_puct, n_playout)

    # 捷徑
    self.link_reset   = self.brain.reset_decision_tree
    self.link_predict = self.brain.decision_guideline.predict_move_probs_and_win_trend
    self.link_train   = self.brain.decision_guideline.train_one_step
    self.link_save_model  = self.brain.decision_guideline.save_model
    self.link_load_model  = self.brain.decision_guideline.load_model

    if model_path:
      self.link_load_model(model_path)
    pass
  pass

  def set_player_ind(self, p):
      self.player = p
  pass

  # refactor done
  def get_action(self, board, temp=1e-3, is_self_play=False):

    assert len(board.availables) > 0, " no availables placement "

    # Step 1. 初始化棋盤上每個位置的獲勝機率
    move_probs = np.zeros(board.width*board.height)
    
    # Step 2. 用alphaZero，預測每個位置的獲勝機率
    #  input: 盤面(board)，環境雜訊(temp)
    #  output:
    #     acts:  可用下子處
    #     probs: 每個下子處的預測獲勝機率
    acts, probs = self.brain._get_probs_of_available_move(board, temp)

    # Step 3. 棋盤上每個位置的獲勝機率設定成alphaZero的預測獲勝機率
    move_probs[list(acts)] = probs

    if is_self_play:
      # Step 4.1 如果是自我訓練，改用dirichlet機率分布模型
      #   return: 
      #      a) move
      #      b) alphaZero預測的機率分布，供後續訓練使用

      # add Dirichlet Noise for exploration (needed for
      # self-play training)
      move = np.random.choice(
          acts,
          p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
      )

      # remove root,
      # use the children[move] as root
      # and reuse the new decision tree
      self.brain.reuse_decision_tree(move)

    else: 
      # Step 4.2 如果是對mcts或人類比賽，改用uniform機率分布模型
      #   return: 
      #      a) move

      move = np.random.choice(acts, p=probs)

      # reset the root node
      self.brain.reset_decision_tree()

    pass

    return move, move_probs
  pass

  def __str__(self):
      return "AlphaZero_MCTS {}".format(self.player)
  pass
pass

##########
##########

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_learning_rate(optimizer, lr):
  """Sets the learning rate to the given value"""
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr
  pass
pass


class PolicyValueNet(nn.Module):
  """policy-value network """
  def __init__(self, board_width, board_height):

    super(PolicyValueNet, self).__init__()

    self.board_width = board_width
    self.board_height = board_height
    self.l2_const = 1e-4  # coef of l2 penalty
    # the policy value net module

    self.ai_vars = OrderedDict()
    self.init_ai_layers()

    self.optimizer = optim.Adam(self._parameters,
                                weight_decay=self.l2_const)
  pass

  def init_ai_layers(self):
    # common layers
    self.ai_vars['conv1'] = nn.Conv2d(4, 32, kernel_size=3, padding=1).to(DEVICE)
    self.ai_vars['conv2'] = nn.Conv2d(32, 64, kernel_size=3, padding=1).to(DEVICE)
    self.ai_vars['conv3'] = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(DEVICE)

    # action policy layers
    # output: weight of each placement in a board [width x height]
    self.ai_vars['act_conv1'] = nn.Conv2d(128, 4, kernel_size=1).to(DEVICE)
    self.ai_vars['act_fc1'] = nn.Linear(4*self.board_width*self.board_height,
                               self.board_width*self.board_height).to(DEVICE)

    # state value layers
    # output: win_trend of a board state [1x1]
    self.ai_vars['val_conv1'] = nn.Conv2d(128, 2, kernel_size=1).to(DEVICE)
    self.ai_vars['val_fc1'] = nn.Linear(2*self.board_width*self.board_height, 64).to(DEVICE)
    self.ai_vars['val_fc2'] = nn.Linear(64, 1).to(DEVICE)

    # collect parameters for optimizer
    # 指定那些變數要拿去optimizer訓練
    # 這邊預設所有變數都拿去訓練
    self._parameters=[]
    for key, value in self.ai_vars.items():
      self._parameters.append({'params':value.parameters()})
    pass
  pass

  def forward(self, state_input):
    # common layers
    x = F.relu(self.ai_vars['conv1'](state_input))
    x = F.relu(self.ai_vars['conv2'](x))
    x = F.relu(self.ai_vars['conv3'](x))
    # action policy layers
    x_act = F.relu(self.ai_vars['act_conv1'](x))
    x_act = x_act.view(-1, 4*self.board_width*self.board_height)
    x_act = F.log_softmax(self.ai_vars['act_fc1'](x_act), dim=1)
    # state value layers
    x_val = F.relu(self.ai_vars['val_conv1'](x))
    x_val = x_val.view(-1, 2*self.board_width*self.board_height)
    x_val = F.relu(self.ai_vars['val_fc1'](x_val))
    x_val = torch.tanh(self.ai_vars['val_fc2'](x_val))
    return x_act, x_val
  pass

  def predict_move_probs_and_win_trend(self, state_batch):
    """
    input: a batch of states
    output: a batch of action probabilities and win trend (state values)
    """
    state_batch = Variable(torch.FloatTensor(np.array(state_batch)).to(DEVICE))
    log_act_probs, value = self.forward(state_batch)
    act_probs = np.exp(log_act_probs.data.cpu().numpy())
    return act_probs, value.data.cpu().numpy()

  pass

  def _get_weight_of_available_move(self, board):
    """
    input: board
    output: a list of (action, probability) tuples for each available
    action and the score of the board state
    """
    legal_positions = board.availables
    current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))

    log_act_probs, value = self.forward(
            Variable(torch.from_numpy(current_state)).to(DEVICE).float())
    act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())

    act_probs = zip(legal_positions, act_probs[legal_positions])
    value = value.data[0][0]
    return act_probs, value
  pass

  def train_one_step(self, samples, lr):
    """perform a training step"""
    # wrap in Variable

    state_batch = Variable(torch.FloatTensor(np.array(samples["state_batch"])).to(DEVICE))
    mcts_probs_batch = Variable(torch.FloatTensor(np.array(samples["mcts_probs_batch"])).to(DEVICE))
    winner_batch = Variable(torch.FloatTensor(np.array(samples["winner_batch"])).to(DEVICE))


    # zero the parameter gradients
    self.optimizer.zero_grad()
    # set learning rate
    set_learning_rate(self.optimizer, lr)

    # forward
    log_act_probs, value = self.forward(state_batch)
    # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
    # Note: the L2 penalty is incorporated in optimizer
    value_loss = F.mse_loss(value.view(-1), winner_batch)
    policy_loss = -torch.mean(torch.sum(mcts_probs_batch*log_act_probs, 1))
    loss = value_loss + policy_loss
    # backward and optimize
    loss.backward()
    self.optimizer.step()
    # calc policy entropy, for monitoring only
    entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
            )
    #return loss.data[0], entropy.data[0]
    #for pytorch version >= 0.5 please use the following line instead.
    return loss.item(), entropy.item()
  pass

  def save_model(self, file_path):
    torch.save( self.ai_vars , file_path)
  pass

  def load_model(self, file_path):
    self.ai_vars = torch.load(file_path, 
                     map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
  pass

pass