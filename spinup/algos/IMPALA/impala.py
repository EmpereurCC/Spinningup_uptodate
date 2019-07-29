import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.ppo_pyco.core as core
from gym.spaces import Box, Discrete
from spinup.utils.logx import EpochLogger


def impala(n,num_epoch,env_fn,max_ep_len,steps_per_epoch):
    dict_continous_gym = ['CarRacing', 'LunarLander','Pong','AirRaid','Adventure', 'AirRaid', 'Alien', 'Amidar',
                          'Assault', 'Asterix', 'Asteroids', 'Atlantis',
        'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival',
        'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender', 'Demon_attack', 'DoubleDunk',
        'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
        'Hero', 'IceHockey', 'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
        'MpntezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pooyan',
        'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing',
        'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
        'Venture', 'VideoPinball', 'WizardOfWor', 'VarsRevenge', 'Zaxxon','Numberlink']
    dict_discrete_gym = []
    dict_gym = ['CarRacing', 'LunarLander','Pong','AirRaid','Adventure', 'AirRaid', 'Alien', 'Amidar',
                          'Assault', 'Asterix', 'Asteroids', 'Atlantis',
        'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival',
        'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender', 'Demon_attack', 'DoubleDunk',
        'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
        'Hero', 'IceHockey', 'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
        'MpntezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pooyan',
        'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing',
        'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
        'Venture', 'VideoPinball', 'WizardOfWor', 'VarsRevenge', 'Zaxxon','Numberlink']
    actors=[actor() for i in range(n)]


class actor:
    #logger = EpochLogger(**logger_kwargs)
    #logger.save_config(locals())

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

class ImpalaBuffer:
    """
    A buffer for storing trajectories experienced by an IMPALA actor interacting
    with the environment and maybe later the stuff for V-trace algorithm
    """

    def __init__(self, gym_or_pyco, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        # self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        if gym_or_pyco == 'pyco':
            self.obs_buf = np.zeros((size, obs_dim[0], obs_dim[1], 1), dtype=np.float32)
        else:
            self.obs_buf = np.zeros((size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]
