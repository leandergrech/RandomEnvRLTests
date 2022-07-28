import os
import numpy as np
from collections import defaultdict
from copy import deepcopy
import gym
from random_env.envs import get_discrete_actions
from tqdm import tqdm
from random_env.envs import VREDA


class AdaptiveTile:
    count = 0
    def __init__(self, ranges:list, nb_actions, with_sub_tiles=True):
        """
        :param ranges: ((min1, max1), (min2, max2),...,(mink, maxk)) where K is the number of state dimensions
        :param with_sub_tiles: Tile includes sub_tiles. When False, Tile is a sub_tile
        """
        self.hash = AdaptiveTile.count
        AdaptiveTile.count += 1

        self.nb_actions = nb_actions

        self.values = [0.0 for _ in range(nb_actions)]

        self.left = None
        self.right = None

        self.ranges = ranges
        self.K = len(ranges)

        self.lowest_td_error = float('inf')

        self.sub_tiles = None
        if with_sub_tiles:
            self.init_sub_tiles()

    def __str__(self):
        return f'{self.ranges}'

    def init_sub_tiles(self):
        self.sub_tiles = defaultdict(list)
        for k in range(self.K):
            dim_half_width = (self.ranges[k][1] - self.ranges[k][0]) / 2.0

            ranges = deepcopy(self.ranges)
            ranges[k][1] -= dim_half_width      # Max of first tile in kth dimension is up to split
            self.sub_tiles[k].append(AdaptiveTile(ranges, self.nb_actions, False))

            ranges = deepcopy(self.ranges)
            ranges[k][0] += dim_half_width      # Min of second tile in kth dimension starts from split
            self.sub_tiles[k].append(AdaptiveTile(ranges, self.nb_actions, False))

    def in_tile(self, state):
        for s_, r_ in zip(state, self.ranges):
            if s_ < r_[0] or s_ > r_[1]:
                return False
        return True

    def update(self, alpha, delta, action_idx):
        self.lowest_td_error = min(self.lowest_td_error, abs(delta))

        self.values[action_idx] += alpha * delta

    def has_children(self):
        if self.left is None:
            return False
        else:
            return True

    def get_active_tile(self, state):
        tile = self
        while tile.has_children():
            if tile.left.in_tile(state):
                tile = tile.left
            elif tile.right.in_tile(state):
                tile = tile.right
        return tile

    def get_sub_tiles_activated(self, state):
        activated_sub_tiles = []
        all_sub_tiles = [v for vs in self.sub_tiles.values() for v in vs]
        for tile in all_sub_tiles:
            if tile.in_tile(state):
                activated_sub_tiles.append(tile)
        return activated_sub_tiles

    def get_value(self, action_idx):
        return self.values[action_idx]

    def split_tile(self, k: int):
        """
        Split along dimension k
        :param k: Which dimension along which to split
        """
        self.left = self.sub_tiles[k][0]
        self.left.init_sub_tiles()

        self.right = self.sub_tiles[k][1]
        self.right.init_sub_tiles()


class Agent:
    def __init__(self, tiling, actions):
        self.tiling = tiling
        self.actions = actions
        self.metrics = defaultdict(float)
        self.highest_metric = 0.0
        self.best_candidate_for_split = dict(tile=None, k=None)

    def get_greedy_action(self, state) -> int:
        """
        Returns index of the action within actions that maximizes estimated return
        :param state:
        :return: action index
        """
        q_values = self.tiling.get_active_tile(state).values
        action_idx = max([(v, i) for i, v in enumerate(q_values)])[1]   # argmax

        return action_idx

    # def split_criterion(self, state):
    #     active_tile = self.tiling.get_active_tile(state)
    #     for k, sub_tiles in active_tile.sub_tiles.items():
    #         for sub_tile in sub_tiles:
    #             if not sub_tile.in_tile(state):
    #                 continue
    #             advantage = max(np.subtract(sub_tile.values, active_tile.values))
    #             if advantage > 0:
    #                 self.benefits[sub_tile.hash] += advantage
    #                 cur_benefit = self.benefits[sub_tile.hash]
    #                 if cur_benefit > self.highest_benefit:
    #                     self.highest_benefit = cur_benefit
    #                     self.best_candidate_for_split['tile'] = active_tile
    #                     self.best_candidate_for_split['k'] = k
    def split_criterion(self, state):
        active_tile = self.tiling.get_active_tile(state)
        for k, sub_tiles in active_tile.sub_tiles.items():
            for sub_tile in sub_tiles:
                if not sub_tile.in_tile(state):
                    continue
                self.metrics[sub_tile.hash] += 1    # state visitation
                cur = self.metrics[sub_tile.hash]
                if cur > self.highest_metric:
                    self.highest_metric = cur
                    self.best_candidate_for_split['tile'] = active_tile
                    self.best_candidate_for_split['k'] = k


    def split_a_tile(self):
        tile_to_split = self.best_candidate_for_split['tile']
        k = self.best_candidate_for_split['k']
        # print(f'{str(tile_to_split)} at dim {k}\tDeltaV={tile_to_split.lowest_td_error}')
        tile_to_split.split_tile(k)
        self.highest_metric = 0.0


def plot_tiles(adaptive_tile, agent, env):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from itertools import product

    def ranges_to_rect(ranges):
        xy = (ranges[0][0], ranges[1][0])
        width = np.ptp(ranges[0])
        height = np.ptp(ranges[1])
        return xy, width, height


    def get_leaves(root):
        res = []
        if root.has_children():
            res += get_leaves(root.left)
            res += get_leaves(root.right)
        else:
            return [root]

        return res


    def in_order_traversal(root):
        res = []
        if root:
            temp = in_order_traversal(root.left)
            if not root.left.has_children():
                res = temp
                v = max(root.values)
                res.append([root.ranges, v])
            temp = in_order_traversal()
            res = res + in_order_traversal(root.right)
        return res


    orig_ranges = np.array(adaptive_tile.ranges)

    fig, (ax, ax_cm) = plt.subplots(2, gridspec_kw=dict(height_ratios=[10,1]))

    # heatmap
    # nb_bins = 100
    #
    # states = [item for item in product(*[np.linspace(l, h, nb_bins) for l, h in
    #                                      zip(orig_ranges.T[0], orig_ranges.T[1])])]
    # vals = []
    # for s in states:
    #     v = max(adaptive_tile.get_active_tile(s).values)
    #     vals.append(v)
    # vals = np.array(vals).reshape(nb_bins, nb_bins).T
    # extents = np.ravel(orig_ranges)
    # im = ax.matshow(vals, extent=extents, aspect='auto', origin='lower', cmap='jet')
    # plt.colorbar(im)

    # tiles
    # all_nodes = in_order_traversal(adaptive_tile)
    all_nodes = get_leaves(adaptive_tile)
    max_v = min([max(node.values) for node in all_nodes])

    ax.set_xlim(*orig_ranges[0])
    ax.set_ylim(*orig_ranges[1])
    cm = mpl.cm.cool
    for node in all_nodes:
        r = node.ranges
        v = max(node.values)
        ax.add_patch(plt.Rectangle(*ranges_to_rect(r), facecolor=cm(1 - v/max_v), edgecolor='None'))
    norm = mpl.colors.Normalize(vmin=max_v, vmax=0.0)

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm),
                 cax=ax_cm, orientation='horizontal')
    # sample trajectory
    for _ in range(5):
        o = env.reset()
        d = False
        obses = []
        while not d:
            a = agent.get_greedy_action(o)
            obses.append(o)
            otp1, r, d, _ = env.step(a)
            o = otp1
        obses = np.array(obses).T
        ax.plot(obses[0], obses[1], marker='x', zorder=10)
        ax.scatter(obses[0][0], obses[1][0], marker='o', s=20, c='c', zorder=15)

    plt.show()

def train():
    env = gym.make('MountainCar-v0')
    # env = VREDA(1, 1)
    actions = get_discrete_actions(n_act=1, act_dim=3)
    nb_actions = len(actions)

    range_scale = 1.
    ranges = [[range_scale*l, range_scale*h] for l, h in zip(env.observation_space.low, env.observation_space.high)]

    tiling = AdaptiveTile(ranges, len(actions))

    agent = Agent(tiling, actions)

    u = 0

    max_timesteps = 10000

    gamma = 0.999
    alpha = 0.1
    p = 5

    T = 0
    o = env.reset()

    pbar = tqdm(total=max_timesteps)
    while T < max_timesteps:
        if np.random.rand() < 0.1:
            action_idx = np.random.choice(nb_actions)
        else:
            action_idx = agent.get_greedy_action(o)
        a = actions[action_idx]

        otp1, r, d, _ = env.step(a)

        target = r + gamma * max(tiling.get_active_tile(otp1).values)

        active_tile = tiling.get_active_tile(o)

        delta_v =  target - active_tile.get_value(action_idx)

        active_tile.update(alpha, delta_v, action_idx)

        for sub_tile in active_tile.get_sub_tiles_activated(o):
            delta_w = target - sub_tile.get_value(action_idx)
            sub_tile.update(alpha, delta_w, action_idx)

        agent.split_criterion(o)
        abs_delta_v = abs(delta_v)
        if abs_delta_v < active_tile.lowest_td_error:
            u = 0
        else:
            u += 1

        if u > p:
            agent.split_a_tile()
            u = 0

        if d:
            o = env.reset()
        else:
            o = otp1

        T += 1
        if (T+1) % 100 == 0:
            pbar.update(100)
    pbar.close()
    plot_tiles(tiling, agent, env)


if __name__ == '__main__':
    train()
    pass








