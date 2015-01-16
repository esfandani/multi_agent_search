# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from __future__ import division
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        foods = currentGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        if action == Directions.STOP:
            return -float("inf")
        for ghost in newGhostStates:
            if ghost.getPosition() == newPos:
                return -float("inf")
        dist = lambda x:manhattanDistance(x,newPos)
        return -dist(min([food for food in foods],key=dist))
    
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """
    def successors(self, gameState, agentInd):
        actions = gameState.getLegalActions(agentInd)
        return [(gameState.generateSuccessor(agentInd, action), action) for action in actions]
    
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.next = None

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def maximize(self, cur_state, depth, agent_index,max_sol,min_sol):
        ma=temp=float("-inf")
        if cur_state.isWin() or cur_state.isLose():
            return self.evaluationFunction(cur_state)
        actions = cur_state.getLegalActions(0)
        for action in actions:
            successor = cur_state.generateSuccessor(0, action)
            if successor is min_sol:
                temp = min_sol[successor]
            else:
                temp = self.minimize(successor, depth, 1,max_sol,min_sol)
                min_sol[successor] = temp
            if temp > ma:
                ma = temp
                if(depth is 1):
                    self.next = action
        return ma
    
    
    
    def minimize(self, cur_state, depth, agent_index,max_sol,min_sol):
        mi=temp= float("inf")
        num_agents = cur_state.getNumAgents()
        if cur_state.isWin() or cur_state.isLose() or depth==0:
            return self.evaluationFunction(cur_state)
        actions = cur_state.getLegalActions(agent_index)
        for action in actions:
            successor = cur_state.generateSuccessor(agent_index, action)
            if agent_index == num_agents - 1:
                if depth == self.depth:
                    temp = self.evaluationFunction(successor)
                else:
                    if successor is max_sol:
                        temp = max_sol[successor]
                    else:
                        temp = self.maximize(successor, depth+1, 0,max_sol,min_sol)
                        max_sol[successor] = temp
            else:
                if successor is min_sol:
                    temp = min_sol[successor]
                else:
                    temp = self.minimize(successor, depth, agent_index+1,max_sol,min_sol)
                    min_sol[successor] = temp
                    
            if temp < mi:
                mi = temp
        return mi
    
    def getAction(self, gameState):
        
        "*** YOUR CODE HERE ***"
        self.next = None
        self.maximize(gameState, 1, 0,{},{})
        return self.next        

class AlphaBetaAgent(MultiAgentSearchAgent):
    
    def maximizer(self, cur_state, depth, alpha, beta,max_sol,min_sol):
        if cur_state.isWin() or cur_state.isLose() or depth == 0:
            return self.evaluationFunction(cur_state)
        ma =temp=float("-inf")
        actions = cur_state.getLegalActions(0)
        for action in actions:
            successor = cur_state.generateSuccessor(0, action)
            if successor in min_sol:
                temp = min_sol[successor]
            else:
                temp = self.minimizer(successor, depth, alpha, beta,1,max_sol,min_sol)
                min_sol[successor] = temp
            if (temp > ma):
                ma = temp
                if (depth == self.depth):
                    self.next = action
            if ma > beta:
                return ma
            alpha = max(alpha, ma)
        return ma


    def minimizer(self, cur_state, depth, alpha, beta, agent,max_sol,min_sol):
        agents = cur_state.getNumAgents()
        if cur_state.isWin() or cur_state.isLose() or depth == 0:
            return self.evaluationFunction(cur_state)
        mi =temp=float("inf")
        actions = cur_state.getLegalActions(agent)
        for action in actions:
            successor = cur_state.generateSuccessor(agent, action)
            if agent == agents - 1:
                if successor in max_sol:
                    temp = max_sol[successor]
                else:
                    temp = self.maximizer(successor, depth - 1, alpha, beta,max_sol,min_sol)
                    max_sol[successor] = temp
            else:
                if successor in min_sol:
                    temp = min_sol[successor]
                else:
                    temp = self.minimizer(successor, depth, alpha, beta, agent + 1,max_sol,min_sol)
                    min_sol[successor] = temp
            mi = min(mi,temp)
            if mi < alpha:
                return mi
            beta = min(beta, mi)
        return mi
    
    def getAction(self, cur_state):
        self.next = None        
        self.maximizer(cur_state, self.depth, float("-inf"), float("inf"),{},{})
        return self.next



    """
    def getAction(self, gameState):
        """
        #   Returns the minimax action using self.depth and self.evaluationFunction
    """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
    """
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        self.next = None;
        self.maximizer(gameState)
        return self.next
        
    def maximizer(self,cur_state):
        value = float("-inf")
        actions = cur_state.getLegalActions(0)
        length = cur_state.getNumAgents()
        for action in actions:
            successor = cur_state.generateSuccessor(0,action)
            temp = self.expectimax(successor, self.depth,1,length)
            if temp > value:
                value,self.next = temp,action
        return
    
    def expectimax(self, cur_state, depth,agent_index, length):
        if  cur_state.isWin() or cur_state.isLose() or depth == 0:
            return  self.evaluationFunction(cur_state)
        value = float("-inf") if agent_index==0 else 0
        actions = cur_state.getLegalActions(agent_index)
        prob = 1 if agent_index == 0 else 1/len(actions)
        index, d = (0,(depth-1)) if agent_index == length-1 else ((agent_index+1),depth)
        for action in actions:
            successor = cur_state.generateSuccessor(agent_index,action)
            if agent_index == 0:
                temp = self.expectimax( successor, depth,1, length )
                value = max(value, temp)
            else:
                value += self.expectimax(successor, d,index, length)
        return value*prob

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    dist = lambda x:manhattanDistance(x, currentGameState.getPacmanPosition())
    ghost_sum = sum(dist(ghost.getPosition()) for ghost in currentGameState.getGhostStates())
    food_sum = sum([1/dist(food) for food in currentGameState.getFood().asList()])
    return currentGameState.getScore()+food_sum-ghost_sum

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

