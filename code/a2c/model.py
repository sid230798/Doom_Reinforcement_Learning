from models import GymModel, VizdoomModel, A2CAgent

from settings import *

def actor_critic(env, testEnv=None):
    num_inputs = env.observation_space_size()
    num_actions = env.action_space_size()

    # TODO: actor and critic model
    # actor: π(•|s), critic: Q(s, a)
    agent = None

    if env.env_name == DEFAULT_GAME:
        agent = A2CAgent(VizdoomModel(num_actions))
    else:
        agent = A2CAgent(GymModel(num_actions))

    reward_history = agent.train(env)
    print('Training finished...')
    if testEnv is not None:
        print("{} out of 200".format(agent.test(testEnv)))
    else:
        print("{} out of 200".format(agent.test(env)))