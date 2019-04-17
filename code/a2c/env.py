import inspect
import sys

import gym
import vizdoom

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: {} at line {} of {}".format(method, line, fileName))
    sys.exit(1)

class Env:
    """
    Env: is abstract class for vizdoom enviornment
    Vizdoom environment should implement following function to pass the object into the model
    """
    def observation_space_size(self, **kwarg):
        """
        return: state vector size
        """
        raiseNotDefined()

    def action_sapce_size(self, **kwarg):
        """
        return: action sapce vector
        """
        raiseNotDefined()

    def step(self, action):
        """
        return: new_state, reward, done, _
        """
        raiseNotDefined()

    def reset(self):
        """
        reset: resets the game enviornment
        """
        raiseNotDefined()

    def set_visible(self, visible):
        """
        set visibility of the env
        """
        raiseNotDefined()

# TODO: Define Vizdoom enviornment
class VizdoomGame(Env):
    def __init__(self, env_name, env_config):
        self.env_name = env_name
        self.env = VizdoomGame.initialize_vizdoom(env_config)
        self.is_visible = False

    def observation_space_size(self):
        pass

    def action_sapce_size(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def set_visible(self, visible=True):
        self.is_visible = visible
        self.env.set_window_visible(self.is_visible)

    # Creates and initializes ViZDoom environment.
    @staticmethod
    def initialize_vizdoom(config_file_path):
        print("Initializing doom...")
        game = vizdoom.DoomGame()
        game.load_config(config_file_path)
        game.set_window_visible(False)
        game.set_mode(vizdoom.Mode.PLAYER)
        game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
        game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        game.init()
        print("Doom initialized.")
        return game


class TestGame(Env):
    """
    TestGame provide env for openai gym games e.g. CartPole-v0
    """
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.reset()
        self.is_visible = False
    
    def observation_space_size(self):
        return self.env.observation_space.shape[0]

    def action_sapce_size(self):
        return self.env.action_sapce.n

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()

    def set_visible(self, visible=True):
        self.is_visible = visible
