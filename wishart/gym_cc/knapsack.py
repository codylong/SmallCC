import gym
from gym import spaces
import os
import errno
import math
import cPickle as pickle
import numpy as np
import random


class Knapsack(gym.Env):

    ########
    # RL related methods

    def __init__(self):
        # use some standard values for initialization, overwritten by second init
        self.capacity = 12.5  # https://arxiv.org/pdf/1611.09940.pdf uses 12.5 for knpasack50 and 25 for knapsack100 and knapsack200
        self.num_items = 50
        self.all_items = self.get_items()
        self.items = random.sample(self.all_items, self.num_items)
        self.used_items = []
        self.record_value = 0
        self.current_pos = 0

        # find a good solution by ordering items by weight-to-value ratio
        sorted_items = sorted(self.items, key=lambda e: float(e[0])/float(e[1]))
        good_val, tot_weight = 0, 0
        for e in sorted_items:
            if tot_weight + e[0] > self.capacity: break
            tot_weight += e[0]
            good_val += e[1]


        print "Good solution has value {} and weight {}.".format(good_val, tot_weight)

        self.action_space = spaces.Discrete(self.num_items)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_items, 2))
        self.state = [[0, 0] for _ in range(self.num_items)]  # [[weight, value], [weight, value], ..., [weight, value]]

        self.global_t = 0
        self.process_idx = 0
        self._outputFilePath = "./output.txt"

        # for the MCTS
        self.tree = {}  # dictionary of the form {(a_1,1): [state, reward], ..., (a_1,depth): [state, reward], (a_2,1,a_2,2): [state, acuumumlated reward], ...}
        self.tree_width = 1  # note: for more tree reusability, make the tree deep but not wide
        self.tree_depth = 1

    def step(self, action):
        done = False

        # check whether move is legal, i.e. item has not been used previously
        if action in self.used_items:
            my_reward = -1
            return np.array(self.state), my_reward, done, {}

        # move is legal
        new_state, my_reward, weight = self.do_step(action, self.state)

        if weight > self.capacity:  # knapsack overpacked or full
            done = True
            if self.record_value < my_reward:
                print "New record: {} with weight {}".format(sum(v[1] for v in self.state), sum(v[0] for v in self.state))
                print "Knapsack: ", self.state
                self.record_value = my_reward

            my_reward = 0

        else:  # knapsack not full, update state
            self.state = new_state
            self.used_items.append(action)

        return np.array(self.state), my_reward, done, {}

    def do_step(self, action, start_state):
        new_state = list(start_state)
        new_state[self.current_pos] = self.items[action]
        self.current_pos += 1
        reward = self.reward()

        return new_state, reward, sum(v[0] for v in new_state)

    def reset(self):
        self.state = [[0, 0] for _ in range(self.num_items)]
        self.current_pos = 0
        self.used_items = []

        return np.array(self.state)

    def reward(self):
        return sum(v[1] for v in self.state)

    def get_items(self):
        return [[random.random(), random.random()] for _ in range(1000)]

    def init_output(self):
        if not os.path.exists(os.path.dirname(self._outputFilePath)):
            try:
                os.makedirs(os.path.dirname(self._outputFilePath))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # write header data
        hnd = open(self._outputFilePath, 'a+')
        hnd.write("head: " + str((self.eps, self.nmod, self.sigma, self.metric_index)) + "\n")
        hnd.close()

    def set_output_file_path(self, path):
        self._outputFilePath = path + "/output.txt"

    def setGlobal_t(self, global_t):
        self.set_global_t(global_t)

    def set_global_t(self, global_t):
        self.global_t = global_t

    def set_process_idx(self, idx):
        self.process_idx = idx

    def output(self, global_t, process_idx):
        return

    ###
    # MCTS Methods

    def recursive_build_tree(self, state, actions, accumulated_reward, depth, agent, always_use_predictions):
        # max recursion depth reached, update tree and return
        if depth >= self.tree_depth:
            self.tree[tuple(actions)] = [state, accumulated_reward]
            return

        # get tree_width many best actions for a given state
        if always_use_predictions:
            best_actions = self.get_best_actions(state, agent)
        else:
            if depth == 0:
                best_actions = self.get_best_actions(state, agent)
            else:
                best_actions = np.random.choice(self.action_space.n, self.tree_width)

        # get all keys corresponding to pre-computed states at this tree level
        nth_leaves = [k for k in self.tree if len(k) == depth + 1]
        # loop over all actions
        for a in best_actions:
            new_action = actions + [a]
            if tuple(new_action) in nth_leaves:
                state, reward = self.tree[tuple(new_action)]
            else:  # compute new contributions
                state, reward, _ = self.do_step(a, state)

            self.tree[tuple(new_action)] = [state, accumulated_reward + reward]
            self.recursive_build_tree(state, new_action, accumulated_reward + reward, depth + 1, agent,
                                      always_use_predictions)

    def get_best_actions(self, state, agent):
        statevar = agent.batch_states(np.array([state]), np, agent.phi)
        pout, vout = agent.model.pi_and_v(statevar)
        pout_probs = pout.all_prob.data[0]
        pout_top_action_probs = sorted(pout_probs, reverse=True)
        pout_top_action_probs = pout_top_action_probs[0:self.tree_width]
        pout_top_actions = []
        found_enough_actions = False
        for ap in pout_top_action_probs:
            if found_enough_actions:
                break
            pos = np.where(pout_probs == ap)[0]
            if len(pos) + len(pout_top_actions) > self.tree_width:
                for i in range(len(pos)):
                    pout_top_actions.append(pos[i])
                    if len(pout_top_actions) == self.tree_width:
                        found_enough_actions = True
                        break
            else:
                pout_top_actions.append(pos[0])
        return pout_top_actions

    def update_tree(self, new_tree_root):
        # reshuffle the tree by making new_tree_root the new root
        new_tree = {}
        root_reward = self.tree[(new_tree_root,)][1]
        for acts, vals in self.tree.items():
            if (acts[0] != new_tree_root) or (
                    len(acts) == 1 and acts[0] == new_tree_root):  # throw away old branches and leave that is now root
                continue

            # now shift the keys one to the left, i.e. rebuild the tree w.r.t. to new root
            new_tree[tuple([acts[i] for i in range(1, len(acts))])] = [vals[0], vals[1] - root_reward]

        self.tree = new_tree
        return

    def get_best_action_and_update_tree(self, agent):
        # return random action to keep exploring:
        if np.random.random() > 0.9:
            return np.array([np.random.randint(0, self.action_space.n)])
        # fill tree first
        self.recursive_build_tree(self.state, [], 0, 0, agent, False)

        # now find the best action, i.e. the one that accumulates the largest reward
        last_leaves = {reward[-1]: acts[0] for acts, reward in self.tree.items() if len(acts) == self.tree_depth}
        best_action = last_leaves[max(last_leaves.keys())]

        # make this selected best action the new tree root
        self.update_tree(best_action)
        return np.array([best_action])
