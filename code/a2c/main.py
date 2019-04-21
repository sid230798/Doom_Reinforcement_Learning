import os
from argparse import ArgumentParser

from env import VizdoomGame
from env import TestGame
from model import actor_critic
from settings import *


if __name__ == "__main__":
    os.chdir(BASE_DIR)

    parser = ArgumentParser("ViZDoom example showing how to train a simple agent using simplified DQN.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "../../scenarios/*cfg for more scenarios.")
    
    parser.add_argument(dest="game",
                        default=DEFAULT_GAME,
                        nargs="?",
                        help="Name of the game, e.g. Vizdoom(default), CartPole-v0, any other openai gym env.")
    
    args = parser.parse_args()

    env, testEnv = None, None
    if args.game == DEFAULT_GAME:
        env = VizdoomGame(args.game, args.config)
        # testEnv = VizdoomGame(args.game, args.config, visible=True)
    else:
        env = TestGame(args.game)

    actor_critic(env, testEnv)