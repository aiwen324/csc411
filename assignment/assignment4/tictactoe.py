from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = nn.ReLU().forward(self.W1(x))
        o = self.W2.forward(h)
        # print o.size()
        output = nn.Softmax().forward(o)
        return output

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr) 
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    G_t = [] 
    for i in range(len(rewards)-1, -1, -1):
        if len(G_t) == 0:
            G_t.append(rewards[i])
        else:
            G_t = [rewards[i]+gamma*G_t[0]] + G_t
    return G_t

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 1, # TODO
            Environment.STATUS_INVALID_MOVE: -2000,
            Environment.STATUS_WIN         : 1000,
            Environment.STATUS_TIE         : 20,
            Environment.STATUS_LOSE        : -1000
    }[status]

def train(policy, env, gamma=1.0, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    highest_reward = (0, 0)

    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            if (running_reward / log_interval) > highest_reward[1]:
                highest_reward = (i_episode, running_reward / log_interval)
                print('update highest_reward: ', highest_reward)
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)


if __name__ == '__main__':
    import sys
    policy = Policy(hidden_size=64)
    env = Environment()

    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        train(policy, env, gamma=0.9)
    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        x_s = []
        empty_stack = np.empty((0, 9), dtype=float)
        # policy = Policy(hidden_size=64)
        for i in range(1, 51):
            env.reset()
            ep = i*1000
            x_s.append(ep)
            load_weights(policy, ep)
            # print("Episode: ", ep)
            # print(first_move_distr(policy, env))
            prob_dist = first_move_distr(policy, env).numpy()
            empty_stack = np.vstack((empty_stack, prob_dist))
        for i, c in zip([0,1,2,3,4],['r','g','b','c','m']):
            plt.plot(x_s, empty_stack[:, i], c, label='position: '+str(i))
        plt.legend()
        plt.savefig('report/part7_first5.png')
        plt.show()
        for i, c in zip([5,6,7,8],['r','g','b','y']):
            plt.plot(x_s, empty_stack[:, i], c, label='position: '+str(i))
        plt.legend()
        plt.savefig('report/part7_last4.png')
        plt.show()
        
            
# ====================== Part 5d ============================
ep = 50000
load_weights(policy, ep)
win_count = 0
tie_count = 0
lose_count = 0
for i in range(5):
    state = env.reset()
    while not env.done:
        act, logprob = select_action(policy, state)
        state, status, done = env.play_against_random(act)
        if i < 5:
            env.render()
        if status == env.STATUS_WIN:
            win_count += 1
            print('Agent Win!')
        elif status == env.STATUS_TIE:
            tie_count += 1
            print('Tie')
        elif status == env.STATUS_LOSE:
            lose_count += 1
            print('Agent Lose')
print(win_count)
print(lose_count)
print(tie_count)

# ====================== Part 6 ============================
x_s = []
win_rates = []
tie_rates = []
lose_rates = []
for i in range(1, 51):
    ep = i*1000
    load_weights(policy, ep)
    win_count = 0
    tie_count = 0
    lose_count = 0
    for i in range(1000):
        state = env.reset()
        while not env.done:
            act, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(act)
            if status == env.STATUS_WIN:
                win_count += 1
            elif status == env.STATUS_TIE:
                tie_count += 1
            elif status == env.STATUS_LOSE:
                lose_count += 1
    win_rate = np.true_divide(win_count, 1000)
    tie_rate = np.true_divide(tie_count, 1000)
    lose_rate = np.true_divide(lose_count, 1000)
    print('episode {}, performance is win_rate: {}, tie_rate: {}, lose_rate: {}'.format(
            ep, str(win_rate), str(tie_rate), str(lose_rate)))
    x_s.append(ep)
    win_rates.append(win_rate)
    tie_rates.append(tie_rate)
    lose_rates.append(lose_rate)

plt.plot(x_s, win_rates, 'r', label='win_rate')
plt.plot(x_s, tie_rates, 'g', label='tie_rate')
plt.plot(x_s, lose_rates, 'b', label='lose_rate')
plt.legend()
plt.savefig('report/part6.png')
plt.show()