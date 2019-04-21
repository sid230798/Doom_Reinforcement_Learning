import os

# disable tensorflow debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

tf.logging.set_verbosity(tf.logging.ERROR)

from settings import *

exploration_rate = EXPLORATION_RATE
def change_rate():
    if exploration_rate < 0.4:
        return exploration_rate
    else:
        return exploration_rate * EXPLORATION_DECAY_RATE

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class Model(tf.keras.Model):
    def __init(self):
        super().__init('mlp_policy')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs, dtype=tf.float32)

        action_prob = self.actor(x)
        Qvals = self.critic(x)
        return action_prob, Qvals

    def action_value(self, obs, training=True):
        global exploration_rate

        logits, value = self.predict(obs)
        if not training:
            action = np.argmax(logits, axis=1)
            return action[0], np.squeeze(value, axis=-1)

        if np.random.uniform() < exploration_rate:
            action = self.dist.predict(logits)
            exploration_rate = change_rate()
            print("\t\t\t\t\tExploring {}".format(exploration_rate), end='\r')
        else:
            action = np.argmax(logits, axis=1)

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class GymModel(Model):
    def __init__(self, num_actions):
        super().__init__()

        self.actor = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(6, activation=tf.nn.relu),
            keras.layers.Dense(num_actions, name='policy_logits'),
        ])

        self.critic = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(6, activation=tf.nn.relu),
            keras.layers.Dense(1, name='value')
        ])

        self.dist = ProbabilityDistribution()


class VizdoomModel(Model):
    def __init__(self, num_actions):
        super().__init__()
        self.actor = keras.Sequential([
            keras.layers.Conv2D(kernel_size=[6, 6], activation=tf.nn.relu, filters=32,
                                bias_initializer=tf.constant_initializer(0.1), strides=[3, 3]),
            keras.layers.Conv2D(kernel_size=[3, 3], activation=tf.nn.relu, filters=32,
                                bias_initializer=tf.constant_initializer(0.1), strides=[2, 2]),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(num_actions),
        ])

        self.critic = keras.Sequential([
            keras.layers.Conv2D(kernel_size=[6, 6], activation=tf.nn.relu, filters=32,
                                bias_initializer=tf.constant_initializer(0.1), strides=[3, 3]),
            keras.layers.Conv2D(kernel_size=[3, 3], activation=tf.nn.relu, filters=32,
                                bias_initializer=tf.constant_initializer(0.1), strides=[2, 2]),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Dense(1),
        ])

        self.dist = ProbabilityDistribution()


class A2CAgent:
    def __init__(self, model):
        self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.95}
        self.model = model
        self.model.compile(
            optimizer=tf.train.AdamOptimizer(),
            loss=[self._logits_loss, self._value_loss],
        )

    def train(self, env):
        actions = np.empty((BATCH_SIZE, ), dtype=np.int32)
        rewards, dones, values = np.empty((3, BATCH_SIZE))
        observations = np.empty((BATCH_SIZE, ) + env.observation_space_size())

        episode_reward = [0.0]
        next_obs = env.reset()
        ckpt = 0
        for update in range(UPDATES):
            print("Training {}".format(update), end='\r')
            for step in range(BATCH_SIZE):
                observations[step] = next_obs.copy()
                actions[step], values[step] = self.model.action_value(next_obs[None, :])
                next_obs, rewards[step], dones[step], _ = env.step(actions[step])

                episode_reward[-1] += rewards[step]
                if dones[step]:
                    episode_reward.append(0.0)
                    next_obs = env.reset()

            _, next_value = self.model.action_value(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            act_adv = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            losses = self.model.train_on_batch(observations, [act_adv, returns])
            self.model.save_weights(DEFAULT_MODEL_SAVEFILE.format(env.env_name, ckpt % 10))
            ckpt = (ckpt + 1) % 10
        return episode_reward

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1 - dones[t])

        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def test(self, env, render=True, times=4):
        rewardl = []
        env.set_visible(render)
        for _ in range(times):
            obs, done, rewards = env.reset(), False, 0
            while not done:
                action, _ = self.model.action_value(obs[None, :], False)
                obs, reward, done, _ = env.step(action)
                rewards += reward
                if render:
                    env.render()
            rewardl.append(rewards)
        return rewardl

    def _value_loss(self, returns, value):
        return self.params['value']*keras.losses.mean_squared_error(returns, value)

    def _logits_loss(self, act_adv, logits):
        actions, advantages = tf.split(act_adv, 2, axis=-1)
        weighted_sparse_ce = keras.losses.CategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        entropy_loss = keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        return policy_loss - self.params['entropy']*entropy_loss
