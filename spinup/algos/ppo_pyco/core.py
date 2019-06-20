import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete
from tensorflow.keras.layers import Input, Concatenate, Dense, Embedding, Conv2D, Flatten, Lambda

EPS = 1e-8

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        #x = tf.layers.dense(x, units=h, activation=activation)
        x = tf.keras.layers.Dense(units=h, activation=activation)(x)
    #return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
    return tf.keras.layers.Dense(units=hidden_sizes[-1], activation=output_activation)(x)
def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def attention_CNN(x):
    #x = tf.layers.conv2d(inputs=x, filters=12, kernel_size=[2, 2], strides=1, padding='VALID', activation=tf.nn.relu)
    #x = tf.layers.conv2d(inputs=x, filters=24, kernel_size=[2, 2], strides=1, padding='VALID', activation=tf.nn.relu)
    x = tf.keras.layers.Conv2D(filters=12, kernel_size=[2,2], strides=1, padding='same', activation=tf.nn.relu)(inputs=x)
    x = tf.keras.layers.Conv2D(filters=24, kernel_size=[2,2], strides=1, padding='same', activation=tf.nn.relu)(inputs=x)
    shape = x.get_shape()
    return x, [s.value for s in shape]

def residual_CNN(x):
    x = tf.keras.layers.Conv2D(filters=26, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu)(inputs=x)
    x = tf.keras.layers.Conv2D(filters=26, kernel_size=[3,3], strides=1, padding='same', activation=tf.nn.relu)(inputs=x)
    return x


def query_key_value(nnk, shape):
    flatten = tf.reshape(nnk, [-1, shape[1]*shape[2], shape[3]])
    #after_layer = [tf.layers.dense(inputs=flatten, units=shape[3], activation=tf.nn.relu) for i in range(3)]
    after_layer = [tf.keras.layers.Dense(units=shape[3], activation=tf.nn.relu)(inputs=flatten) for i in range(3)]
    return after_layer[0], after_layer[1], after_layer[2], flatten

def layer_normalization(x):
    feature_shape = x.get_shape()[-1:]
    mean, variance = tf.nn.moments(x, [2], keep_dims=True)
    beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(feature_shape), trainable=False)
    return gamma * (x - mean) / tf.sqrt(variance + 1e-8) + beta

def output_layer(f_theta, hidden, output_size, activation, final_activation):
    for h in hidden:
        #f_theta = tf.layers.dense(inputs=f_theta, units=h, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())
        f_theta = tf.keras.layers.Dense(units=h, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())(inputs=f_theta)
    #return tf.layers.dense(inputs=f_theta, units=output_size, activation=final_activation)
    return tf.keras.layers.Dense(units=output_size, activation=final_activation)(inputs=f_theta)

def residual(x, inp, residual_time):
    for i in range(residual_time):
        x = x + inp
        x = layer_normalization(x)
    return x

def feature_wise_max(x):
    return tf.reduce_max(x, axis=2)

def feature_wise_max_baseline(x):
    return tf.reduce_max(x, axis=[1,2])


def self_attention(query, key, value):
    key_dim_size = float(key.get_shape().as_list()[-1])
    key = tf.transpose(key, perm=[0, 2, 1])
    S = tf.matmul(query, key) / tf.sqrt(key_dim_size)
    attention_weight = tf.nn.softmax(S)
    A = tf.matmul(attention_weight, value)
    shape = A.get_shape()
    return A, attention_weight, [s.value for s in shape]

"""
Policies
"""

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    a.shape
    act_dim = action_space.n
    logits= mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi, logits


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi




def relational_categorical_policy(x,a, hidden=[256], output_size = 2, activation = tf.nn.relu, final_activation=tf.nn.softmax,act_dim=2):
    nnk, shape = attention_CNN(x)
    query, key, value, E = query_key_value(nnk, shape)
    normalized_query = layer_normalization(query)
    normalized_key = layer_normalization(key)
    normalized_value = layer_normalization(value)
    A, attention_weight, shape = self_attention(normalized_query, normalized_key, normalized_value)
    E_hat = residual(A, E, 2)
    max_E_hat = feature_wise_max(E_hat)
    logits = output_layer(max_E_hat, hidden, output_size, activation, final_activation)
    logp_all = tf.nn.log_softmax(logits)
    #pi = tf.squeeze(tf.multinomial(logits,1),axis=1)
    pi = tf.squeeze(tf.random.categorical(logits,1),axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi, logits, max_E_hat

def baseline_categorical_policy(x,a, hidden=[256],output_size = 2, activation = tf.nn.relu, final_activation=tf.nn.softmax, act_dim=2):
    x, shape = attention_CNN(x)
    new_x = residual_CNN(x)
    new_x = residual_CNN(new_x)
    new_x = residual_CNN(new_x)
    max_E_hat = feature_wise_max_baseline(new_x)
    logits = output_layer(max_E_hat, hidden, output_size, activation, final_activation)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi, logits, max_E_hat



def baseline_gaussian_policy(x, a, act_dim, output_activation, hidden_sizes=(64,64), activation = tf.nn.relu):
    x, shape = attention_CNN(x)
    new_x = residual_CNN(x)
    new_x = residual_CNN(new_x)
    new_x = residual_CNN(new_x)
    max_E_hat = feature_wise_max_baseline(new_x)
    mu = mlp(max_E_hat, list(hidden_sizes)+[act_dim], activation, output_activation)
    #log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    log_std = tf.constant(-0.5 * np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi, max_E_hat






"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy == 'relational_categorical_policy' :
        policy = relational_categorical_policy
    elif policy == 'baseline_categorical_policy' :
        policy = baseline_categorical_policy
    elif policy == 'baseline_gaussian_policy' :
        policy = baseline_gaussian_policy
    elif policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        #pi, logp, logp_pi, logits = policy(x, a, hidden_sizes, activation, output_activation, action_space)
        if policy == baseline_gaussian_policy:
            pi, logp, logp_pi, max_E_hat = policy(x, a, action_space, output_activation, hidden_sizes=(64,64), activation = tf.nn.relu)
        else:
            pi, logp, logp_pi, logits, max_E_hat = policy(x, a, output_size=action_space, act_dim=action_space )


    with tf.variable_scope('v'):
        #v = tf.squeeze(mlp(x, list(hidden_sizes) + [1], activation, None), axis=1)
        v = tf.squeeze(output_layer(max_E_hat, [256], 1, tf.nn.relu, None), axis = 1)
    if policy == baseline_gaussian_policy:
        return pi, logp, logp_pi, v
    else:
        return pi, logp, logp_pi, v, logits