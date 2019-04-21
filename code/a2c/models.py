import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from settings import *

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

    def action_value(self, obs):
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class GymModel(Model):
    def __init__(self, num_actions):
        super().__init__()

        self.actor = keras.Sequential([
            keras.layers.Dense(6, activation=tf.nn.relu),
            keras.layers.Dense(num_actions, name='policy_logits'),
        ])

        self.critic = keras.Sequential([
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
            keras.layers.Dense(num_actions, activation=tf.nn.softmax),
        ])

        self.critic = keras.Sequential([
            keras.layers.Conv2D(kernel_size=[6, 6], activation=tf.nn.relu, filters=8,
                                bias_initializer=tf.constant_initializer(0.1), strides=[3, 3]),
            keras.layers.Conv2D(kernel_size=[3, 3], activation=tf.nn.relu, filters=32,
                                bias_initializer=tf.constant_initializer(0.1), strides=[2, 2]),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(1),
        ])
        # self.actor.build()
        # print(self.actor.summary())
        # print(self.critic.summary())

        self.dist = ProbabilityDistribution()


class A2CAgent:
    def __init__(self, model):
        self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.95}
        self.model = model
        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=[self._logits_loss, self._value_loss],
        )

    def train(self, env, batch_size=32, updates=1000):
        actions = np.empty((batch_size, ), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size, ) + env.observation_space_size())

        episode_reward = [0.0]
        next_obs = env.reset()
        for update in range(updates):
            print("Training {}".format(update), end='\r')
            for step in range(batch_size):
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
                action, _ = self.model.action_value(obs[None, :])
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
