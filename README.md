# TD-Gammon
Implements the famous TD-Gammon algorithm for Backgammon-playing built on top of this java project: http://modelai.gettysburg.edu/2013/tdgammon/index.html

Most of the code is taken from here:
http://modelai.gettysburg.edu/2013/tdgammon/index.html
http://modelai.gettysburg.edu/2013/tdgammon/pa2.pdf
http://modelai.gettysburg.edu/2013/tdgammon/pa4.pdf

But the following classes have been added:
1. To package player - BackPropPlayer2,Utility
2. To package driver - SimulationDriver,TestStrengthDriver

BackPropPlayer2 : A NN player, that implements TD learning, takes parameters Lambda(TD-lambda), Alpha(NN learning-rate), trainingmode(true/false)
Utility: Implements primarily "BoardtoVec" to change a board to a NN representation
SimulationDriver: To run simulations
TestStrengthDriver: To test any two players against each other (the first one is always black)

Useful things to know:
1. SavedNN is a trained NN, the one provided with the code was generated after 1000000 games
of selfplay (specified in the while loop of SimulationDriver) and took 886 minutes to train on a very fast laptop.
Comparing this to another Tensorflow based (Python-simulated) implmentation: 
https://github.com/fomorians/td-gammon
https://medium.com/jim-fleming/before-alphago-there-was-td-gammon-13deff866197
The author mentions that it takes him 1 hr for 1000 games training, while for this implementation, it takes only about a minute.

2. NeuralNetworkVisualizer class can be used to visualize the Neural Nets thus generated

Important observation:
The quality of network trained depends highly on the startegy adopted to choose next moves. I trained two different networks as follows:
1. Next move is the one, where the next board-position has smallest utility for the opponent. Similar to Minimax strategy.
2. Next move is simply where there is greatest ulitilty. 

The network trained using 1st strategy has much better playing capacity than the 2nd.













————————————————————————————————————————————————————————————————————————————————————————————————————————————————
One of the most impressive applications of reinforcement learning to date is that by Gerry Tesauro to the game of backgammon (Tesauro, 1992, 1994, 1995). Tesauro's program, TD-Gammon, required little backgammon knowledge, yet learned to play extremely well, near the level of the world's strongest grandmasters. The learning algorithm in TD-Gammon was a straightforward combination of the TD($\lambda $) algorithm and nonlinear function approximation using a multilayer neural network trained by backpropagating TD errors.

Backgammon is a major game in the sense that it is played throughout the world, with numerous tournaments and regular world championship matches. It is in part a game of chance, and it is a popular vehicle for waging significant sums of money. There are probably more professional backgammon players than there are professional chess players. The game is played with 15 white and 15 black pieces on a board of 24 locations, called points. Figure  11.1 shows a typical position early in the game, seen from the perspective of the white player.


Figure 11.1: A backgammon position
 

In this figure, white has just rolled the dice and obtained a 5 and a 2. This means that he can move one of his pieces 5 steps and one (possibly the same piece) 2 steps. For example, he could move two pieces from the 12 point, one to the 17 point, and one to the 14 point. White's objective is to advance all of his pieces into the last quadrant (points 19-24) and then off the board. The first player to remove all his pieces wins. One complication is that the pieces interact as they pass each other going in different directions. For example, if it were black's move in Figure  11.1, he could use the dice roll of 2 to move a piece from the 24 point to the 22 point, "hitting" the white piece there. Pieces that have been hit are placed on the "bar" in the middle of the board (where we already see one previously hit black piece), from whence they reenter the race from the start. However, if there are two pieces on a point, then the opponent cannot move to that point; the pieces are protected from being hit. Thus, white cannot use his 5-2 dice roll to move either of his pieces on the 1 point, because their possible resulting points are occupied by groups of black pieces. Forming contiguous blocks of occupied points to block the opponent is one of the elementary strategies of the game.

Backgammon involves several further complications, but the above description gives the basic idea. With 30 pieces and 24 possible locations (26, counting the bar and off-the-board) it should be clear that the number of possible backgammon positions is enormous, far more than the number of memory elements one could have in any physically realizable computer. The number of moves possible from each position is also large. For a typical dice roll there might be 20 different ways of playing. In considering future moves, such as the response of the opponent, one must consider the possible dice rolls as well. The result is that the game tree has an effective branching factor of about 400. This is far too large to permit effective use of the conventional heuristic search methods that have proved so effective in games like chess and checkers.

On the other hand, the game is a good match to the capabilities of TD learning methods. Although the game is highly stochastic, a complete description of the game's state is available at all times. The game evolves over a sequence of moves and positions until finally ending in a win for one player or the other, ending the game. The outcome can be interpreted as a final reward to be predicted. On the other hand, the theoretical results we have described so far cannot be usefully applied to this task. The number of states is so large that a lookup table cannot be used, and the opponent is a source of uncertainty and time variation.

TD-Gammon used a nonlinear form of TD($\lambda $). The estimated value, , of any state (board position)  was meant to estimate the probability of winning starting from state . To achieve this, rewards were defined as zero for all time steps except those on which the game is won. To implement the value function, TD-Gammon used a standard multilayer neural network, much as shown in Figure  11.2. (The real network had two additional units in its final layer to estimate the probability of each player's winning in a special way called a "gammon" or "backgammon.") The network consisted of a layer of input units, a layer of hidden units, and a final output unit. The input to the network was a representation of a backgammon position, and the output was an estimate of the value of that position.


Figure 11.2: The neural network used in TD-Gammon
 

In the first version of TD-Gammon, TD-Gammon 0.0, backgammon positions were represented to the network in a relatively direct way that involved little backgammon knowledge. It did, however, involve substantial knowledge of how neural networks work and how information is best presented to them. It is instructive to note the exact representation Tesauro chose. There were a total of 198 input units to the network. For each point on the backgammon board, four units indicated the number of white pieces on the point. If there were no white pieces, then all four units took on the value zero. If there was one piece, then the first unit took on the value 1. If there were two pieces, then both the first and the second unit were 1. If there were three or more pieces on the point, then all of the first three units were 1. If there were more than three pieces, the fourth unit also came on, to a degree indicating the number of additional pieces beyond three. Letting  denote the total number of pieces on the point, if , then the fourth unit took on the value . With four units for white and four for black at each of the 24 points, that made a total of 192 units. Two additional units encoded the number of white and black pieces on the bar (each took the value , where  is the number of pieces on the bar), and two more encoded the number of black and white pieces already successfully removed from the board (these took the value , where  is the number of pieces already borne off). Finally, two units indicated in a binary fashion whether it was white's or black's turn to move. The general logic behind these choices should be clear. Basically, Tesauro tried to represent the position in a straightforward way, making little attempt to minimize the number of units. He provided one unit for each conceptually distinct possibility that seemed likely to be relevant, and he scaled them to roughly the same range, in this case between 0 and 1.

Given a representation of a backgammon position, the network computed its estimated value in the standard way. Corresponding to each connection from an input unit to a hidden unit was a real-valued weight. Signals from each input unit were multiplied by their corresponding weights and summed at the hidden unit. The output, , of hidden unit  was a nonlinear sigmoid function of the weighted sum:




where  is the value of the th input unit and  is the weight of its connection to the th hidden unit. The output of the sigmoid is always between 0 and 1, and has a natural interpretation as a probability based on a summation of evidence. The computation from hidden units to the output unit was entirely analogous. Each connection from a hidden unit to the output unit had a separate weight. The output unit formed the weighted sum and then passed it through the same sigmoid nonlinearity.
TD-Gammon used the gradient-descent form of the TD($\lambda $) algorithm described in Section 8.2, with the gradients computed by the error backpropagation algorithm (Rumelhart, Hinton, and Williams, 1986). Recall that the general update rule for this case is


 	(11.1)

where  is the vector of all modifiable parameters (in this case, the weights of the network) and  is a vector of eligibility traces, one for each component of , updated by



with . The gradient in this equation can be computed efficiently by the backpropagation procedure. For the backgammon application, in which  and the reward is always zero except upon winning, the TD error portion of the learning rule is usually just , as suggested in Figure  11.2.
