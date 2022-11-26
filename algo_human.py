
class Algo_Human(object):
  """
  human player
  """

  def __init__(self):
    self.player = None
  pass

  def __str__(self):
    return "Human {}".format(self.player)
  pass

pass

def Algo_Human_set_player_ind(self, p):
  self.player = p
pass
Algo_Human.set_player_ind=Algo_Human_set_player_ind

def Algo_Human_get_action(self, board):
  
  try:
    location = input("Your move y,x : ")
    if isinstance(location, str):  # for python3
      location = [int(n, 10) for n in location.split(",")]
    pass
    move = board.location_to_move(location)
  except Exception as e:
    move = -1
  pass

  if move == -1 or move not in board.availables:
      print("invalid move")
      move = self.get_action(board)
  pass

  # 回傳使用這個move的原因
  _log_of_move_probs = None

  return move, _log_of_move_probs

pass
Algo_Human.get_action=Algo_Human_get_action

