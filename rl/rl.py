import numpy as np
import os
import copy
import time
from random import randint

os.system('cls' if os.name == 'nt' else 'clear')  # clear console


class Game:

    def __init__(self, width, height, gaps, food, gap_reward, food_reward,
                 step_reward, random_factor):
        self.width = width
        self.height = height
        self.gaps = gaps
        self.food = food
        self.gap_reward = gap_reward
        self.food_reward = food_reward
        self.step_reward = step_reward
        self.random_factor = random_factor
        self.Q = np.zeros((width * height, width * height))
        self.initial_fields = self.get_initial_fields()
        self.free_space_char = '-'
        self.food_char = 'F'
        self.agent_char = 'A'
        self.gap_char = 'G'
        self.stone_char = 'X'
        self.N = 'N'

    def create_inital_reward_matrix(self):
        size = self.width * self.height
        irm = np.full((size, size), self.N, dtype='<U21')

        def get_reward(x, y):
            if ((x, y) in self.gaps):
                return self.gap_reward
            if ((x, y) in self.food):
                return self.food_reward
            return self.step_reward

        for h in range(self.height):
            for w in range(self.width):
                from_ = self.width * h + w
                if (w - 1) >= 0:
                    to_ = self.width * h + (w - 1)
                    irm[from_, to_] = get_reward(h, w - 1)
                if (w + 1) < self.width:
                    to_ = self.width * h + (w + 1)
                    irm[from_, to_] = get_reward(h, w + 1)
                if (h + 1) < self.height:
                    to_ = self.width * (h + 1) + w
                    irm[from_, to_] = get_reward(h + 1, w)
                if (h - 1) >= 0:
                    to_ = self.width * (h - 1) + w
                    irm[from_, to_] = get_reward(h - 1, w)
        return irm

    def disable_food_field(self, reward, h, w):
        '''
        Once food has been found, agent can not return to that field.
        '''
        food_field = h * self.width + w

        if (h - 1) >= 0:
            _from = self.width * (h - 1) + w
            reward[_from, food_field] = self.N

        if (h + 1) < self.height:
            _from = self.width * (h + 1) + w
            reward[_from, food_field] = self.N

        if (w + 1) < self.width:
            _from = self.width * h + (w + 1)
            reward[_from, food_field] = self.N

        if (w - 1) >= 0:
            _from = self.width * h + (w - 1)
            reward[_from, food_field] = self.N

    def forbid_action(self, r, old_action, new_action):
        r[old_action, new_action] = self.N

    def get_initial_fields(self):
        '''
        Return fields in which agent can begin.
        '''
        initial_fields = []
        for h in range(self.height):
            for w in range(self.width):
                if not((h, w) in self.gaps or (h, w) in self.food):
                    initial_fields.append(h * self.width + w)
        return initial_fields

    def pick_next_state(self, actions, current_state, exp):
        '''
        Pick next state using best posible Q score or random
        state to increase exploration.
        '''
        next_state = None

        if (exp and randint(0, 99) < int(self.random_factor * 10)):
            next_state = actions[randint(0, len(actions) - 1)]
        else:
            best_score = -float('inf')
            for a in actions:
                if (self.Q[current_state, a] > best_score):
                    next_state = a
                    best_score = self.Q[current_state, a]
        return next_state

    def get_2d_position(self, state):
        w = state % self.width
        h = state // self.width
        return (h, w)

    def show_legend(self):
        print('LEGEND')
        print('Agent', self.agent_char)
        print('Food', self.food_char)
        print('Gap', self.gap_char)
        print('Stone', self.stone_char)
        print('Space', self.free_space_char)
        print('\n')

    def train(self, iterations, learning_rate, dicount_factor):
        for t in range(iterations):
            current_state = self.initial_fields[
                randint(0, len(self.initial_fields) - 1)]
            f = copy.copy(self.food)
            r = self.create_inital_reward_matrix()

            while (f):
                actions = np.where(r[current_state, :] != self.N)[0]

                # no where to move
                if (not len(actions)):
                    break

                next_state = self.pick_next_state(
                    actions, current_state, exp=True)

                h, w = self.get_2d_position(next_state)

                # pick best Q-value for next state
                best_score = -float('inf')
                next_actions = np.where(r[next_state, :] != self.N)[0]

                # no where to move
                if (not len(next_actions)):
                    break

                for a in next_actions:
                    if (self.Q[next_state, a] > best_score):
                        best_score = self.Q[next_state, a]

                self.Q[current_state, next_state] += learning_rate * \
                    (float(r[current_state, next_state]) +
                     (dicount_factor * best_score) - self.Q[current_state,
                                                            next_state])

                self.forbid_action(r, current_state, next_state)

                if ((h, w) in f):
                    del f[(h, w)]
                    self.disable_food_field(r, h, w)

                current_state = next_state

    def print_agent_path(self, sleep):
        '''
        Follow best way with little of exploration to avoid cycles.
        '''
        current_state = self.initial_fields[
            randint(0, len(self.initial_fields) - 1)]
        f = copy.copy(self.food)
        r = self.create_inital_reward_matrix()

        score = 0
        while (f):
            h, w = self.get_2d_position(current_state)

            print('You are at position ({},{}).'.format(h, w))
            print('Current score is {}.\n'.format(score))

            for row in range(self.height):
                s_row = []
                for col in range(self.height):
                    s = self.free_space_char
                    if (row == h and col == w):
                        s = self.agent_char
                    elif ((row, col) in f):
                        s = self.food_char
                    elif ((row, col) in self.food):
                        s = self.stone_char
                    elif ((row, col) in self.gaps):
                        s = self.gap_char
                    s_row.append(s)
                print(' '.join(s_row))
            print('\n')

            self.show_legend()

            if (sleep and sleep != 'interactive'):
                time.sleep(sleep)
                os.system('cls' if os.name == 'nt' else 'clear')
            elif (sleep and sleep == 'interactive'):
                input('Press Enter to continue...')
                os.system('cls' if os.name == 'nt' else 'clear')

            actions = np.where(r[current_state, :] != self.N)[0]
            if (not len(actions)):
                print('No where to move ... ending.')
                break

            next_state = self.pick_next_state(actions, current_state, exp=None)
            score += int(r[current_state][next_state])
            self.forbid_action(r, current_state, next_state)

            current_state = next_state
            h, w = self.get_2d_position(current_state)

            if ((h, w) in f):
                del f[(h, w)]
                self.disable_food_field(r, h, w)
                print('Found food! Earned {} points.'.format(self.food_reward))
            elif ((h, w) in self.gaps):
                print('Fell into gap! Earned {} points.'
                      .format(self.gap_reward))
            else:
                print('Basic move. Earned {} points.'.format(self.step_reward))

        print('Finished with score {}.'.format(score))


if __name__ == '__main__':
    def to_dict(items):
        return dict((i, 1) for i in items)

    # transform gaps and food to dict to access them quicker
    gaps = to_dict([(1, 2), (2, 1), (3, 2), (2, 4)])
    food = to_dict([(0, 1), (0, 4), (2, 2), (3, 4)])

    game = Game(width=5, height=5, gaps=gaps, food=food,
                gap_reward=-1000, food_reward=1000, step_reward=-100,
                random_factor=0.2)

    game.train(iterations=1000, learning_rate=0.05, dicount_factor=0.5)
    # game.print_agent_path(sleep=False)
    # game.print_agent_path(sleep=2)
    game.print_agent_path(sleep='interactive')
