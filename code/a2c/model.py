import tensorflow as tf
import tensorflow.keras as keras

from env import raiseNotDefined

DEFAULT_GAME = "Vizdoom"

HIDDEN_SIZE = 6

def actor_critic(env, epoch, alphaA=3e-4, alphaC=3e-4):
    num_inputs = env.observation_space_size()
    num_actions = len(env.action_space_size())

    # TODO: actor and critic model
    # actor: π(•|s), critic: Q(s, a)
    actor_model = None  
    critic_model = None

    # TODO: 1. model for Vizdoom

    if env.env_name == DEFAULT_GAME:
        raiseNotDefined()
    else:
        actor_model = keras.Sequential([
            keras.layers.Flatten(input_shape=(1, num_inputs)),
            keras.layers.Dense(HIDDEN_SIZE, activation=tf.nn.relu),
            keras.layers.Dense(1),
        ])

        critic_model = keras.Sequential([
            keras.layers.Flatten(input_shape=(1, num_inputs)),
            keras.layers.Dense(HIDDEN_SIZE, activation=tf.nn.relu),
            keras.layers.Dense(num_actions, activation=tf.nn.softmax),
        ])

    env.reset()
    # TODO: execute a2c