import numpy as np
import collections

import os
import pickle

from rlcard.utils.utils import *
import time
from rlcard.games.mahjong.card import MahjongCard as Card

class MCCFRAgent():
    ''' Implement CFR algorithm
    '''
    def __init__(self, env, model_path='./mccfr_model'):
        ''' Initilize Agent

        Args:
            env (Env): Env class
        '''
        self.all_actions = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'green', 'red', 'white', 'east', 'west', 'north', 'south', 'peng', 'chi', 'gang', 'guo']

        self.use_raw = False
        self.env = env
        self.model_path = model_path

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        self.iteration = 0

        self._expl = 0.6

    def action_probs(self, obs, legal_actions, policy):
        ''' Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        if obs not in policy.keys():
            action_probs = np.array([1.0/len(legal_actions) for _ in range(len(legal_actions))])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        # action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    # def _add_regret(self, info_state_key, action_idx, amount):
    #     self._infostates[info_state_key][_REGRET_INDEX][action_idx] += amount

    def eval_step(self):
        current_player = 0
        info_state_key, legal_actions = self.get_state(current_player)

        num_legal_actions = len(legal_actions)
        action_probs = self.action_probs(info_state_key, legal_actions, self.policy)
        # print(action_probs)

        sampled_action_idx = np.random.choice(
            np.arange(num_legal_actions), p=action_probs)
        # print(legal_actions[sampled_action_idx])
        print(self.all_actions[legal_actions[sampled_action_idx]])
        self.env.step(legal_actions[sampled_action_idx])
    def show_actions(self):
        current_player = 1
        info_state_key, legal_actions = self.get_state(current_player)
        # print(legal_actions)


        sort_action = sorted(legal_actions)
        print(sort_action)
        print([self.all_actions[action_id] for action_id in sort_action])

    def do_actions(self, action):

        self.env.step(action)

    def update_policy(self):
        ''' Update policy based on the current regrets
        '''
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)

    def regret_matching(self, obs):
        ''' Apply regret matching

        Args:
            obs (string): The state_str
        '''
        regret = self.regrets[obs]
        positive_regret_sum = sum([r for r in regret if r > 0])

        action_probs = np.zeros(self.env.action_num)
        if positive_regret_sum > 0:
            for action in range(self.env.action_num):
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.action_num):
                action_probs[action] = 1.0 / self.env.action_num
        return action_probs

    def train(self):
        """Performs one iteration of outcome sampling.

        An iteration consists of one episode for each player as the update player.
        """
        # for update_player in range(self._num_players):
        #     state = self._game.new_initial_state()
        #     self._episode(
        #         state, update_player)

        for player_id in range(self.env.player_num):
            self.env.init_game()
            # probs = np.ones(self.env.player_num)
            self.traverse_tree(player_id, my_reach=1.0, opp_reach=1.0, sample_reach=1.0)

        self.update_policy()
    def traverse_tree(self, update_player, my_reach, opp_reach, sample_reach):
        """Runs an episode of outcome sampling.

        Args:
          state: the open spiel state to run from (will be modified in-place).
          update_player: the player to update regrets for (the other players update
            average strategies)
          my_reach: reach probability of the update player
          opp_reach: reach probability of all the opponents (including chance)
          sample_reach: reach probability of the sampling (behavior) policy

        Returns:
          A tuple of (util, reach_tail), where:
            - util is the utility of the update player divided by the sample reach
              of the trajectory, and
            - reach_tail is the product of all players' reach probabilities
              to the terminal state (from the state that was passed in).
        """

        #  my_reach=1.0, opp_reach=1.0, sample_reach=1.0
        # if self.env.is_over():
        #     return state.player_return(update_player) / sample_reach
        if self.env.is_over():
            return self.env.get_payoffs(), 1.0


        current_player = self.env.get_player_id()

        info_state_key, legal_actions = self.get_state(current_player)

        num_legal_actions = len(legal_actions)


        # infostate_info = self._lookup_infostate_info(info_state_key,
        #                                              num_legal_actions)

        # 获取action 并且归一化  。下面的函数没有归一化动作
        action_probs = self.action_probs(info_state_key, legal_actions, self.policy)

        if current_player == update_player:
            uniform_policy = (
                    np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions)
            # uniform_policy = remove_illegal(uniform_policy, legal_actions)
            sampling_policy = (
                    self._expl * uniform_policy + (1.0 - self._expl) * action_probs)
        else:
            sampling_policy = action_probs
        sampled_action_idx = np.random.choice(
            np.arange(num_legal_actions), p=sampling_policy)
        if current_player == update_player:
            new_my_reach = my_reach * action_probs[sampled_action_idx]
            new_opp_reach = opp_reach
        else:
            new_my_reach = my_reach
            new_opp_reach = opp_reach * action_probs[sampled_action_idx]
        new_sample_reach = sample_reach * sampling_policy[sampled_action_idx]
        self.env.step(legal_actions[sampled_action_idx])
        util, reach_tail = self.traverse_tree(update_player, new_my_reach,
                                         new_opp_reach, new_sample_reach)
        new_reach_tail = action_probs[sampled_action_idx] * reach_tail
        # The following updates are based on equations 4.9 - 4.15 (Sec 4.2) of
        # http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf
        if info_state_key not in self.regrets:
            self.regrets[info_state_key] = np.zeros(num_legal_actions)
        if current_player == update_player:
            # update regrets. Note the w here already includes the sample reach of the
            # trajectory (from root to terminal) in util due to the base case above.
            w = util[current_player] * opp_reach
            for action_idx in range(num_legal_actions):
                if action_idx == sampled_action_idx:
                    regret = w * (reach_tail - new_reach_tail)
                    # self._add_regret(info_state_key, action_idx,
                    #                  regret)
                    self.regrets[info_state_key][action_idx] += regret
                else:
                    regret = -w * new_reach_tail
                    # self._add_regret(info_state_key, action_idx, regret)
                    self.regrets[info_state_key][action_idx] += regret
        # else:
        #     # update avg strat
        #     for action_idx in range(num_legal_actions):
        #         self._add_avstrat(info_state_key, action_idx,
        #                           opp_reach * action_probs[action_idx] / sample_reach)
        return util, new_reach_tail
    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        return state['obs'].tostring(), state['legal_actions']
    def save(self, iter):
        ''' Save model
        '''

        if not os.path.exists(self.model_path + '/' + str(iter)):
            os.makedirs(self.model_path + '/' + str(iter))

        policy_file = open(os.path.join(self.model_path, str(iter), 'policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, str(iter), 'average_policy.pkl'),'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, str(iter), 'regrets.pkl'),'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, str(iter), 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self, iter):
        ''' Load model
        '''
        if not os.path.exists(self.model_path + '/' + str(iter)):
            return

        policy_file = open(os.path.join(self.model_path, str(iter), 'policy.pkl'),'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, str(iter), 'average_policy.pkl'),'rb')
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, str(iter), 'regrets.pkl'),'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, str(iter), 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()