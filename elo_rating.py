# ref
# https://www.geeksforgeeks.org/elo-rating-algorithm/

'''
P_t=P_(t-1) + K * (S_a - E_a)

Where P at “t-1” is elo the score before the match, 
K is the adjustment factor, S at “a” is the result of 
the match (1 if they won, 0 if they lost, and 0.5 if 
there was a tie).

E at “a” is the expected value of the match that is 
calculated using the logistic function and takes as 
parameters the Elo rating of both players. It is 
calculated as follows.


E_a= 1 / (  1 + 10^((Elo_B - Elo_A)/400)  )

'''

# Python 3 program for Elo Rating
import math
 
# Function to calculate the Probability
def Probability(rating1, rating2): 
  return 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400))
pass

 
# Function to calculate Elo rating
# K is a constant.
# d determines whether
# Player A wins or Player B.
def EloRating(Ra, Rb, K, d):
  
 
    # To calculate the Winning
    # Probability of Player B
    Pb = Probability(Ra, Rb)
 
    # To calculate the Winning
    # Probability of Player A
    Pa = Probability(Rb, Ra)
 
    # Case -1 When Player A wins
    # Updating the Elo Ratings
    if (d == 1) :
        Ra = Ra + K * (1 - Pa)
        Rb = Rb + K * (0 - Pb)
     
 
    # Case -2 When Player B wins
    # Updating the Elo Ratings
    else :
        Ra = Ra + K * (0 - Pa)
        Rb = Rb + K * (1 - Pb)
    pass
 
    print("Updated Ratings:-")
    print("Ra =", round(Ra, 6)," Rb =", round(Rb, 6))
pass
# Driver code
 
# Ra and Rb are current ELO ratings
Ra = 1200
Rb = 1000
K = 30
d = 1 # 1: a win, 2: b win
EloRating(Ra, Rb, K, d)
 
