import tensorflow as tf
import tensorflow.keras as keras


def actor_critic(env, epoch, alphaA=3e-4, alphaC=3e-4):
    num_inputs = env.observation_space_size()
    num_actions = len(env.action_space())

    # TODO: actor and critic model
    # actor: π(•|s), critic: Q(s, a)
    actor_model = None  
    critic_model = None
    