import inspect
import sys

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

    def action_sapce(self, **kwarg):
        """
        return: action sapce vector
        """
        raiseNotDefined()

    def step(self, action):
        """
        return: new_state, reward, done, _
        """
        raiseNotDefined()


# TODO: Define test enviornment e.g. CartPole-v0 from openai.gym
# TODO: Define Vizdoom enviornment

class VizdoomGame(Env):
    # TODO
    pass

class TestGame0(Env):
    # TODO
    pass
