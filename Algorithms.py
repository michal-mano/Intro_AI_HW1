import numpy as np
import time
from IPython.display import clear_output
from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

class Node():
    def __init__(self, parent, state_tuple, action, parent_cost, node_cost,env) -> None:
        self.env = env
        self.parent = parent
        self.state_tuple = state_tuple
        self.action_to_reach = action
        self.is_developed = False
        self.cumulative_cost = parent_cost + node_cost

class Agent():
  def __init__(self):
    self.env = None

  def animation(self, epochs: int ,state: int, action: List[int], total_cost: int) -> None:
      clear_output(wait=True)
      print(self.env.render())
      print(f"Timestep: {epochs}")
      print(f"State: {state}")
      print(f"Action: {action}")
      print(f"Total Cost: {total_cost}")
      time.sleep(1)

def get_actions(node) -> List[int]:
    path = []
    while node.parent is not None:
        path.insert(0, node.action_to_reach)
        node = node.parent
    return path

class BFSAgent():
    def __init__(self) -> None:
        self.env = None

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.env = env
        self.env.reset()
        open = []
        closed = []
        cost = 0
        total_cost = 0
        expanded = 0
        state = self.env.get_initial_state()
        rootNode = Node(None, state, None, -1, 1, self.env)
        open.append(rootNode)
        while len(open) > 0:
            node = open.pop(0)
            closed.append(node.state_tuple)
            expanded += 1
            print('expanded node:', node.state_tuple, '\n')
            row, col = self.env.to_row_col(node.state_tuple)
            if self.env.desc[row, col] != b"H":
                for action, (state, cost, terminated) in env.succ(node.state_tuple).items():
                    child_node = Node(node, state, action, node.cumulative_cost, cost, self.env)
                    child_node.state_tuple = (child_node.state_tuple[0], node.state_tuple[1], node.state_tuple[2])
                    if child_node.state_tuple[0] == env.d1[0]:
                        child_node.state_tuple = (child_node.state_tuple[0], True, child_node.state_tuple[2])
                    if child_node.state_tuple[0] == env.d2[0]:
                        child_node.state_tuple = (child_node.state_tuple[0], child_node.state_tuple[1], True)
                    if child_node.state_tuple not in closed and child_node.state_tuple not in [open_node.state_tuple for open_node in open]:
                        if env.is_final_state(child_node.state_tuple):
                            return get_actions(child_node), child_node.cumulative_cost, expanded
                        if child_node.state_tuple != node.state_tuple:
                            open.append(child_node)



class WeightedAStarAgent():
    def __init__(self) -> None:
        self.env = None
        self.h_weight = 0

    def manhattan_dist (self, state1, state2) -> int:
        (state1_row, state1_col) = self.env.to_row_col(state1)
        (state2_row, state2_col)  = self.env.to_row_col(state2)
        return abs(state1_row - state2_row) + abs(state1_col - state2_col)


    def hmsap(self, state):
        distance_to_goals = []
        distance_to_goals.append(self.manhattan_dist(state, self.env.d1))
        distance_to_goals.append(self.manhattan_dist(state, self.env.d2))
        for goal in self.env.goals:
            distance_to_goals.append(self.manhattan_dist(state, goal))
        return min(distance_to_goals)

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.env = env
        self.h_weight = h_weight
        open = heapdict.heapdict()
        close = heapdict.heapdict()
        expanded = 0
        state = self.env.get_initial_state()
        rootNode = Node(None, state, None, -1, 1, self.env)
        weight_avg = (1-h_weight) * rootNode.cumulative_cost + h_weight * self.hmsap(rootNode.state_tuple)
        open[rootNode] = (weight_avg)
        while len(open) > 0:
            (current_node, weight_avg) = open.popitem()
            close[current_node] = weight_avg
            if (env.is_final_state(current_node.state_tuple)):
                return get_actions(current_node), current_node.cumulative_cost, expanded
            expanded += 1
            print('expanded node:', current_node.state_tuple, '\n')
            row, col = self.env.to_row_col(current_node.state_tuple)
            if self.env.desc[row, col] != b"H":
                for action, (state, cost, terminated) in env.succ(current_node.state_tuple).items():
                    child_node = Node(current_node, state, action, current_node.cumulative_cost, cost, self.env)
                    child_node.state_tuple = (child_node.state_tuple[0], current_node.state_tuple[1], current_node.state_tuple[2])
                    if child_node.state_tuple[0] == env.d1[0]:
                        child_node.state_tuple = (child_node.state_tuple[0], True, child_node.state_tuple[2])
                    if child_node.state_tuple[0] == env.d2[0]:
                        child_node.state_tuple = (child_node.state_tuple[0], child_node.state_tuple[1], True)
                    new_weight = (1-h_weight) * child_node.cumulative_cost + h_weight * self.hmsap(child_node.state_tuple)
                    if child_node.state_tuple not in [node.state_tuple for node in close.keys()] and child_node.state_tuple not in [node.state_tuple for node in open.keys()]:
                        open[child_node] = new_weight
                    if child_node.state_tuple in [node.state_tuple for node in open.keys()]:
                        for key, value in open.items():
                            if key.state_tuple == child_node.state_tuple:
                                open_node = key
                                break
                        if new_weight < open[open_node]:
                            open.pop(open_node)
                            open[child_node] = new_weight

                    if child_node.state_tuple in [node.state_tuple for node in close.keys()]:
                        for key, value in close.items():
                            if key.state_tuple == child_node.state_tuple:
                                close_node = key
                                break
                        if new_weight < close[close_node]:
                            close.pop(close_node)
                            open[child_node] = new_weight



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError