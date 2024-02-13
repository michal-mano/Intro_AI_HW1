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
        path.insert(node.action_to_reach)
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
        rootNode = Node(None, state[0], None, -1, 1)
        open.append(rootNode)
        while len(open) > 0:
            node = open.pop()
            closed.append(node)
            expanded += 1
            for action, (state, cost, terminated) in env.succ(node.state_tuple).items():
                child_node = Node(node, (state,cost,terminated), action, node.cumulative_cost, cost, env)
                if child_node.state_tuple not in closed and child_node.state_tuple not in open:
                    if env.is_final_state(child_node.state_tuple):
                        return (get_actions(child_node), child_node.cumulative_cost, expanded)
                    if not terminated:
                        open.append(child_node)



class WeightedAStarAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError
        
    def ssearch(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError