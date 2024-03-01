import numpy as np
import time
from IPython.display import clear_output
from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict


class Node():
    def __init__(self, parent, state_tuple, action, parent_cost, node_cost, env) -> None:
        self.env = env
        self.parent = parent
        self.state_tuple = state_tuple
        self.action_to_reach = action
        self.is_developed = False
        self.cumulative_cost = parent_cost + node_cost


class Agent():
    def __init__(self):
        self.env = None
        self.h_weight = 0.5

    def animation(self, epochs: int, state: int, action: List[int], total_cost: int) -> None:
        clear_output(wait=True)
        print(self.env.render())
        print(f"Timestep: {epochs}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Total Cost: {total_cost}")
        time.sleep(1)

    def manhattan_dist(self, state1, state2) -> int:
        (state1_row, state1_col) = self.env.to_row_col(state1)
        (state2_row, state2_col) = self.env.to_row_col(state2)
        return abs(state1_row - state2_row) + abs(state1_col - state2_col)

    def hmsap(self, state):
        distance_to_goals = []
        if (state[1] == False):
            distance_to_goals.append(self.manhattan_dist(state, self.env.d1))
        if (state[2] == False):
            distance_to_goals.append(self.manhattan_dist(state, self.env.d2))
        for goal in self.env.goals:
            distance_to_goals.append(self.manhattan_dist(state, goal))
        return min(distance_to_goals)


    def a_star_search(self, pop_function, env : DragonBallEnv):
        self.env = env
        open = heapdict.heapdict()
        close = heapdict.heapdict()
        expanded = 0
        state = self.env.get_initial_state()
        rootNode = Node(None, state, None, -1, 1, self.env)
        weight_avg = (1 - self.h_weight) * rootNode.cumulative_cost + self.h_weight * self.hmsap(rootNode.state_tuple)
        open[rootNode] = (weight_avg)
        while len(open) > 0:
            (current_node, weight_avg) = pop_function(open)
            open.pop(current_node)
            close[current_node] = weight_avg
            if (self.env.is_final_state(current_node.state_tuple)):
                return get_actions(current_node), current_node.cumulative_cost, expanded
            expanded += 1
            print('expanded:' , current_node.state_tuple, "f-value: ", weight_avg, '\n')
            row, col = self.env.to_row_col(current_node.state_tuple)
            if self.env.desc[row, col] != b"H" and self.env.desc[row, col] != b"G":
                for action, (state, cost, terminated) in self.env.succ(current_node.state_tuple).items():
                    child_node = Node(current_node, state, action, current_node.cumulative_cost, cost, self.env)
                    child_node.state_tuple = (
                        child_node.state_tuple[0], current_node.state_tuple[1], current_node.state_tuple[2])
                    if child_node.state_tuple[0] == self.env.d1[0]:
                        child_node.state_tuple = (child_node.state_tuple[0], True, child_node.state_tuple[2])
                    if child_node.state_tuple[0] == self.env.d2[0]:
                        child_node.state_tuple = (child_node.state_tuple[0], child_node.state_tuple[1], True)
                    new_weight = (1 - self.h_weight) * child_node.cumulative_cost + self.h_weight * self.hmsap(child_node.state_tuple)
                    if child_node.state_tuple not in [node.state_tuple for node in
                                                      close.keys()] and child_node.state_tuple not in [node.state_tuple
                                                                                                       for node in
                                                                                                       open.keys()]:
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
            row, col = self.env.to_row_col(node.state_tuple)
            if self.env.desc[row, col] != b"H" and self.env.desc[row, col] != b"G":
                for action, (state, cost, terminated) in env.succ(node.state_tuple).items():
                    child_node = Node(node, state, action, node.cumulative_cost, cost, self.env)
                    child_node.state_tuple = (child_node.state_tuple[0], node.state_tuple[1], node.state_tuple[2])
                    if child_node.state_tuple[0] == env.d1[0]:
                        child_node.state_tuple = (child_node.state_tuple[0], True, child_node.state_tuple[2])
                    if child_node.state_tuple[0] == env.d2[0]:
                        child_node.state_tuple = (child_node.state_tuple[0], child_node.state_tuple[1], True)
                    if child_node.state_tuple not in closed and child_node.state_tuple not in [open_node.state_tuple for
                                                                                               open_node in open]:
                        if env.is_final_state(child_node.state_tuple):
                            return get_actions(child_node), child_node.cumulative_cost, expanded
                        if child_node.state_tuple != node.state_tuple:
                            open.append(child_node)


class WeightedAStarAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.h_weight = 0

    def manhattan_dist(self, state1, state2) -> int:
        (state1_row, state1_col) = self.env.to_row_col(state1)
        (state2_row, state2_col) = self.env.to_row_col(state2)
        return abs(state1_row - state2_row) + abs(state1_col - state2_col)

    def weighted_pop(self, open):
        node, weight = open.peekitem()
        node_options = [node]
        for key, value in open.items():
            if value == weight:
                node_options.append(key)
        lowest_state = min([node.state_tuple[0] for node in node_options])
        for key, value in open.items():
            if key.state_tuple[0] == lowest_state and value == weight:
                return key, value

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.h_weight = h_weight
        return self.a_star_search(self.weighted_pop, env)


class AStarEpsilonAgent(Agent):
    def __init__(self) -> None:
        super(AStarEpsilonAgent, self).__init__()

    def epsilon_pop(self, open):
        focal = heapdict.heapdict()
        best_node, best_val = open.peekitem()
        focal[best_node] = best_node.cumulative_cost
        for key, value in open.items():
            if value <= (1 + self.epsilon) * best_val:
                focal[key] = key.cumulative_cost

        focal_top, focal_weight = focal.peekitem()
        best_options = [focal_top]
        for key, value in focal.items():
            if value == focal_weight:
                best_options.append(key)
        lowest_state = min([node.state_tuple[0] for node in best_options])
        for key, value in focal.items():
            if key.state_tuple[0] == lowest_state and value == focal_weight:
                return key, open.get(key)

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        self.epsilon = epsilon
        return self.a_star_search(self.epsilon_pop, env)
            #choose best node and remove from open




