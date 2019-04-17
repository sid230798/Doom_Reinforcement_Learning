import os
from argparse import ArgumentParser

from env import VizdoomGame
from env import TestGame
from model import actor_critic, DEFAULT_GAME

# BASE_DIR = parent directory of code directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration file path
DEFAULT_MODEL_SAVEFILE = "tmp/a2c/model.tf"
DEFAULT_CONFIG = "health_gathering.cfg"

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

    env = None
    if args.game == DEFAULT_GAME:
        env = VizdoomGame(args.game, args.config)
    else:
        env = TestGame(args.game)

    # TODO: execute actor_critic