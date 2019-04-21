import os

RESOLUTION = [30, 45]

DEFAULT_GAME = "Vizdoom"
MAX_EPISODES = 3000
NUM_STEPS = 300
GAMMA = 0.95
HIDDEN_SIZE = 6

UPDATES = 4000
BATCH_SIZE = 512

# BASE_DIR = parent directory of code directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration file path
DEFAULT_MODEL_SAVEFILE = "tmp/a2c/model-{}-ckpt-{}.tf"
DEFAULT_CONFIG = "config/basic.cfg"
DEFAULT_CHECKPOINT = "5"

EXPLORATION_RATE=0.95
EXPLORATION_DECAY_RATE = 0.9999