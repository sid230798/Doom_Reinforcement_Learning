from models import GymModel, VizdoomModel, A2CAgent

from settings import *

def actor_critic(env, ckpt="5"):
    num_inputs = env.observation_space_size()
    num_actions = env.action_space_size()

    # TODO: actor and critic model
    # actor: π(•|s), critic: Q(s, a)
    agent = None
    model = None

    if env.env_name == DEFAULT_GAME:
        model = VizdoomModel(num_actions)
    else:
        model = GymModel(num_actions)

    try:
        model.load_weights(DEFAULT_MODEL_SAVEFILE.format(env.env_name, ckpt))
        print("Model loaded from {}".format(DEFAULT_MODEL_SAVEFILE.format(env.env_name, ckpt)))
    except:
        pass

    if env.env_name == DEFAULT_GAME:
        agent = A2CAgent(model)
    else:
        agent = A2CAgent(model)

    reward_history = agent.train(env)
    print('Training finished...')
    print("{} out of 200".format(agent.test(env)))