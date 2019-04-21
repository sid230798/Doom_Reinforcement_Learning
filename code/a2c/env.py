import inspect
import sys
import itertools as it

import gym
import skimage
import vizdoom
import numpy as np

from settings import *

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
        reset: resets the env enviornment
        """
        raiseNotDefined()

    def render(self):
        """
        render: renders the env enviornment
        """
        raiseNotDefined()

    def set_visible(self, visible):
        """
        set visibility of the env
        """
        raiseNotDefined()


class VizdoomGame(Env):
    def __init__(self, env_name, env_config, visible=False):
        self.env_name = env_name
        self.env = VizdoomGame.initialize_vizdoom(env_config, visible)
        self.is_visible = False
        n = self.env.get_available_buttons_size()
        # self.available_actions = [list(a) for a in it.product([0, 1], repeat=n)]
        left = [1, 0, 0]
        right = [0, 1, 0]
        shoot = [0, 0, 1]
        self.available_actions = [left, right, shoot]

    def observation_space_size(self):
        return tuple(RESOLUTION + [1])

    def action_space_size(self):
        n = self.env.get_available_buttons_size()
        return 3

    def step(self, action):
        print("\t\t\t{}".format(action), end="\r")
        action = self.available_actions[action]
        state, reward = self.env.get_state(), self.env.make_action(action)
        done = self.env.is_episode_finished()
        state = VizdoomGame.preprocess(state.screen_buffer)
        state.resize(RESOLUTION + [1])
        return state, reward, done, None

    def reset(self):
        self.env.new_episode()
        s1 = VizdoomGame.preprocess(self.env.get_state().screen_buffer)
        s1.resize(RESOLUTION + [1])
        return s1

    def render(self):
        None

    def set_visible(self, visible=True):
        self.is_visible = visible
        self.env.close()
        self.env.set_window_visible(self.is_visible)
        self.env.set_mode(vizdoom.Mode.ASYNC_PLAYER)
        self.env.init()


    # Creates and initializes ViZDoom environment.
    @staticmethod
    def initialize_vizdoom(config_file_path, visible):
        print("Initializing doom...")
        env = vizdoom.DoomGame()
        env.load_config(config_file_path)
        env.set_window_visible(visible)
        env.set_mode(vizdoom.Mode.PLAYER)
        env.set_screen_format(vizdoom.ScreenFormat.GRAY8)
        env.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        env.init()
        print("Doom initialized.")
        return env

    @staticmethod
    def preprocess(img):
        img = skimage.transform.resize(img, RESOLUTION)
        img = img.astype(np.float32)
        return img


class TestGame(Env):
    """
    TestGame provide env for openai gym games e.g. CartPole-v0
    """
    def __init__(self, env_name):
        print(env_name)
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.reset()
        self.is_visible = False
    
    def observation_space_size(self):
        return self.env.observation_space.shape

    def action_space_size(self):
        return self.env.action_space.n

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def set_visible(self, visible=True):
        self.is_visible = visible
