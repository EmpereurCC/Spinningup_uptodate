import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.ppo_pyco.core as core
import matplotlib.pyplot as plt
from gym.spaces import Box, Discrete
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from spinup.utils.logx import restore_tf_graph

def prod(iterable):
    """No built in produce for python <3.8"""
    p = 1
    for n in iterable:
        p *= n
    return p

def rgb_input_pyco(o, obs_dim):
    """Used to be a function to use pixels. Dropped that idea because it would just complicate everything."""
    A = np.zeros((obs_dim[0], obs_dim[1], 1))
    o = o.board
    return o
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if o[i, j] == 32:
                A[i, j, :] = (870, 838, 678)
            elif o[i, j] == 35:
                A[i, j, :] = (428, 135, 0)
            elif o[i, j] == 46:
                A[i, j, :] = (39, 208, 67)
            elif o[i, j] == 49 or 50 or 51 or 52 or 53:
                A[i, j, :] = (729, 394, 51)
            elif o[i, j] == 95:
                A[i, j, :] = (834, 588, 525)
            elif o[i, j] == 80:
                A[i, j, :] = (388, 400, 999)
            elif o[i, j] == 88:
                A[i, j, :] = (850, 603, 270)
    o = A
    # return o


class actor:

    def __init__(self, x_ph, a_ph, adv_ph, ret_ph):


        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.ret_buf = []
        self.val_buf = []
        self.logp_buf = []
        self.x_ph = x_ph
        self.a_ph = a_ph
        self.adv_ph = adv_ph
        self.ret_ph = ret_ph

    def get_episode(self, env, get_action_ops, gym_or_pyco, obs_dim):
        """ Need to restore the latest learner parameters of the model"""
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if gym_or_pyco == 'gym':
            o = o.reshape(1, obs_dim[0], obs_dim[1], obs_dim[2])
        else:
            o = rgb_input_pyco(o, obs_dim)
            o = o.reshape(1, obs_dim[0], obs_dim[1], 1)

        self.obs_buf = o
        self.rew_buf = r
        self.ret_buf = r
        saver = tf.train.Saver()
        self.sess = tf.Session()
        export_dir = "/home/clement/Documents/spinningup_instadeep/data/cmd_impala/cmd_impala_s0/simple_save"
        #model = restore_tf_graph(self.sess, export_dir)
        with  self.sess as sess:
            saver.restore(sess, export_dir)


            a, v_t, logp_t = self.sess.run(get_action_ops, feed_dict={self.x_ph: o})

            self.act_buf = a
            self.val_buf = v_t
            self.logp_buf = logp_t

            while d == False:
                o, r, d, _ = env.step(self.act_buf[-1])
                if gym_or_pyco == 'gym':
                    o = o.reshape(1, obs_dim[0], obs_dim[1], obs_dim[2])
                else:
                    o = rgb_input_pyco(o, obs_dim)
                    o = o.reshape(1, obs_dim[0], obs_dim[1], 1)

                self.obs_buf = np.append(self.obs_buf, o)
                self.rew_buf = np.append(self.rew_buf, r)
                if r == None:
                    self.ret_buf = np.append(self.ret_buf, self.ret_buf + 0)
                else:
                    self.ret_buf = np.append(self.ret_buf, self.ret_buf + r)


                a, v_t, logp_t = self.sess.run(get_action_ops, feed_dict={self.x_ph: o})

                self.act_buf = np.append(self.act_buf, a)
                self.val_buf = np.append(self.val_buf, v_t)
                self.logp_buf = np.append(self.logp_buf, logp_t)

    def get(self):
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.rew_buf]


def impala(gym_or_pyco, env_fn, ac_kwargs=dict(), n=4, logger_kwargs=dict(), actor_critic=core.mlp_actor_critic,num_cpu=1, epochs=1000, max_ep_len=300,
           steps_per_epoch=4000, gamma=0.99, seed=473, pi_lr = 3e-4 ,rho_bar = 1, c_bar = 1, tensorboard_path = '/home/clement/spinningup/tensorboard'):
    dict_continous_gym = ['CarRacing', 'LunarLander', 'Pong', 'AirRaid', 'Adventure', 'AirRaid', 'Alien', 'Amidar',
                          'Assault', 'Asterix', 'Asteroids', 'Atlantis',
                          'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout',
                          'Carnival',
                          'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender', 'Demon_attack', 'DoubleDunk',
                          'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
                          'Hero', 'IceHockey', 'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
                          'MpntezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pooyan',
                          'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing',
                          'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
                          'Venture', 'VideoPinball', 'WizardOfWor', 'VarsRevenge', 'Zaxxon', 'Numberlink']
    dict_discrete_gym = []
    dict_gym = ['CarRacing', 'LunarLander', 'Pong', 'AirRaid', 'Adventure', 'AirRaid', 'Alien', 'Amidar',
                'Assault', 'Asterix', 'Asteroids', 'Atlantis',
                'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival',
                'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender', 'Demon_attack', 'DoubleDunk',
                'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar',
                'Hero', 'IceHockey', 'Jamesbond', 'JourneyEscape', 'Kangaroo', 'Krull', 'KungFuMaster',
                'MpntezumaRevenge', 'MsPacman', 'NameThisGame', 'Phoenix', 'Pitfall', 'Pooyan',
                'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing',
                'Solaris', 'SpaceInvaders', 'StarGunner', 'Tennis', 'TimePilot', 'Tutankham', 'UpNDown',
                'Venture', 'VideoPinball', 'WizardOfWor', 'VarsRevenge', 'Zaxxon', 'Numberlink']

    env = env_fn()
    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    if gym_or_pyco == 'gym':
        None
    else:
        env = env()

    obs_dim = env.observation_space.shape
    if env.action_space == 4:
        act_dim = env.action_space
    try:
        act_dim = env.action_space.n
    except:
        act_dim = env.action_space.shape

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    # x_ph, a_ph = core.placeholders_from_spaces(env.observation_space, env.action_space)
    if gym_or_pyco == 'pyco':
        x_ph = tf.placeholder(tf.float32, shape=(1, obs_dim[0], obs_dim[1], 1))
    else:
        x_ph = tf.placeholder(tf.float32, shape=(1, obs_dim[0], obs_dim[1], obs_dim[2]))
    # a_ph = core.placeholders_from_spaces(env.action_space)
    if gym_or_pyco == 'gym' and isinstance(env.action_space, Discrete):
        a_ph = tf.placeholder(tf.uint8, shape=(1))

    elif gym_or_pyco == 'gym' and isinstance(env.action_space, Box):
        a_ph = tf.placeholder(tf.float32, shape=(env.action_space.shape[0]))

    else:
        a_ph = tf.placeholder(tf.uint8, shape=(1))

    if gym_or_pyco == 'gym' and isinstance(env.action_space, Discrete):
        pi, logp, logp_pi, v, logits = actor_critic(x_ph, a_ph, policy='baseline_categorical_policy',
                                                    action_space=env.action_space.n)
    elif gym_or_pyco == 'gym' and isinstance(env.action_space, Box):
        pi, logp, logp_pi, v = actor_critic(x_ph, a_ph, policy='baseline_gaussian_policy',
                                            action_space=env.action_space.shape[0])
    else:
        pi, logp, logp_pi, v, logits = actor_critic(x_ph, a_ph, policy='relational_categorical_policy',
                                                    action_space=env.action_space.n)
    adv_ph, ret_ph, pi_act_ph = core.placeholders(None, None, None)
    all_phs = [x_ph, a_ph, adv_ph, ret_ph, pi_act_ph]

    #every steps, get : action, value and logprob.
    get_action_ops = [pi, v, logp_pi]

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    #need to get rho_s from the v_trace function..
    rho_s = tf.minimum(tf.exp(logp)-pi_act_ph, rho_bar)

    #adv_ph = rew_adv + gamma * v_trace(s+1) - v ( la value de pi)
    pi_loss = -tf.reduce_mean(adv_ph*rho_s)
    #with adv_ph the advantage with v_trace. On the whole thing?..
    with tf.name_scope('pi_loss'):
        core.variable_summaries(pi_loss)

    # Optimization
    #num_env_frames = tf.train.get_global_step()
    learning_rate = tf.train.polynomial_decay(pi_lr, 30,
                                              1e9, 0)
    optimizer = tf.train.RMSPropOptimizer(pi_lr, 0.99,
                                          0., .1)
    train_op = optimizer.minimize(pi_loss)

    sess = tf.Session()
    #v_loss = tf.reduce_mean((v_trace(traj_list,rews_list,act_list,len_list,num_traj, actors, sess, c_bar, rho_bar, gamma)-v) ** 2)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(tensorboard_path + '/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(tensorboard_path + '/test')

    sess.run(tf.global_variables_initializer())
    sess.run(sync_all_params())

    #logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi':pi, 'v': v})
    saver = tf.train.Saver()
    save_path = saver.save(sess,"/home/clement/Documents/spinningup_instadeep/data/cmd_impala/cmd_impala_s0/simple_save")

    actors = [actor(x_ph,a_ph,adv_ph,ret_ph,) for i in range(n)]
    for i in range(n):
        actors[i].get_episode(env,get_action_ops,gym_or_pyco,obs_dim)
    rew_list = []
    for i in range(n):
        _, _, _, rew_act = actor[i].get()
        rew_list.append(rew_act)
    rew_adv = []
    for i in range(len(rew_list)):
        rew_adv = np.append(rew_adv, rew_list[i])




def v_trace(traj_list,rews_list,act_list,len_list,num_traj, actors, sess, c_bar, rho_bar,gamma):
    """Prend en entrée les trajectoires et les rewards associés, renvoie un dictionaire associé à des states : à un state x_s est associé un scalaire v_{x_s}
    les trajectoires seront une liste de trajectoires

    Args:
        traj_list: a list of different paths observations used for v_trace.
        rews_list: the list of the rewards lists from each of every paths used for v_trace.
        act_list: a list of the actions lists from each of every paths used for v_trace.
        len_list: a vector of ints containing the size of each path.
        num_traj: an int defined by the number of actors used.
        actors: a class of actors with the corresponding tensorflow sessions linked to them. Contains the "off" policies.
        sess: contains the up to date policy of the graph from the learner at the time of computing v_trace.
        c_bar: hyperparam of v_trace
        rho_bar: hyperparam of v_trace
    """


    v_trace_dic = {}

    #initialize with the values
    for i in range(num_traj):
        for j in range(len_list[i]):
            try:
                tmp = traj_list[i][j]
                v_trace_dic[tuple(map(tuple,tmp))]
            except:
                tmp = traj_list[i][j]
                v_trace_dic[tuple(map(tuple,tmp))] = actor[i].sess.run(traj_v, feed_dict={x_ph: tmp})

    c_param = []
    rho_param = []
    #Need to get final_adv from formula of v_trace
    for i in range(num_traj):
        c_param.append(np.zeros(len_list[i]))
        rho_param.append(np.zeros(len_list[i]))

    for i in range(num_traj):
        for j in range(len_list[i]):
            c_param[i][j] = tf.minimum(actors[i].sess.run(logits, feed_dict={x_ph: traj_list[i][j]})[act_list[i][j]]/sess.run(logits,feed_dict={x_ph: traj_list[i][j]})[act_list[i][j]],c_bar)
            rho_param[i][j] = tf.minimum(actors[i].sess.run(logits, feed_dict={x_ph: traj_list[i][j]})[act_list[i][j]]/sess.run(logits,feed_dict={x_ph: traj_list[i][j]})[act_list[i][j]],rho_bar)

    c_param_prod = []
    for i in range(num_traj):
        c_param_prod.append(np.zeros(len_list[i]))
        for j in range(len_list[i]):
            c_param_prod[i][j] = (gamma^j)*prod(c_param[i][j:])*rho_param[i][j]

    #deltas is the vector of all GAE lambda advantages
    for i in range(len_list):
        v_trace_delta = []
        v_trace_delta.append(np.zeros(len_list[i]))
        deltas = []
        for j in range(len_list[i]):
            tmp = traj_list[i][j]
            v_trace_delta[j] = v_trace_dic[tuple(map(tuple,tmp))]
        deltas = rews_list[i][:-1] + gamma * v_trace_delta[1:] - v_trace_delta[:-1]
        for j in range(len_list[i]):
            tmp = traj_list[i][j]
            v_trace_dic[tuple(map(tuple,tmp))] = v_trace_dic[tuple(map(tuple,tmp))] + sum(c_param_prod[i]*v_trace_delta)

    return rho_param, v_trace_dic






















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
