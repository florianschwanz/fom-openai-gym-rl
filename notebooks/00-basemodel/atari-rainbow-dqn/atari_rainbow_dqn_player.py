import glob
import os
import random
import sys

import imageio
import torch
import torch.autograd as autograd
from tqdm import tqdm

# Make library available in path
lib_path = os.path.join(os.getcwd(), 'lib')
if not (lib_path in sys.path):
    sys.path.insert(0, lib_path)
common_lib_path = os.path.join(os.getcwd(), '..', 'common', 'lib')
if not (common_lib_path in sys.path):
    sys.path.insert(0, common_lib_path)
common_reward_shaper_path = os.path.join(os.getcwd(), '..', 'common', 'reward_shaper')
if not (common_reward_shaper_path in sys.path):
    sys.path.insert(0, common_reward_shaper_path)

# Import library classes
from action_selector import ActionSelector
from deep_q_network import RainbowCnnDQN
from environment_builder import EnvironmentBuilder
from environment_builder import EnvironmentWrapper
from model_storage import ModelStorage
from screen_plotter import ScreenPlotter

# Path to output to be loaded
RUN_TO_LOAD = os.getenv('RUN_TO_LOAD', None)
OUTPUT_DIRECTORY = os.getenv('OUTPUT_DIRECTORY', "./output/")

if RUN_TO_LOAD == None:
    raise Exception("RUN_TO_LOAD has not been specified")

# Get latest file from run
list_of_files = glob.glob(OUTPUT_DIRECTORY + RUN_TO_LOAD + "/*.model")
MODEL_TO_LOAD = max(list_of_files, key=os.path.getctime)

RUN_DIRECTORY = RUN_TO_LOAD

FINISHED_FRAMES, \
FINISHED_EPISODES, \
TOTAL_ORIGINAL_REWARDS, \
TOTAL_SHAPED_REWARDS, \
TOTAL_LOSSES, \
MODEL_STATE_DICT, \
OPTIMIZER_STATE_DICT, \
REPLAY_MEMORY, \
LOSS, \
 \
ENVIRONMENT, \
ENVIRONMENT_WRAPPERS, \
BATCH_SIZE, \
GAMMA, \
EPS_START, \
EPS_END, \
EPS_DECAY, \
NUM_ATOMS, \
VMIN, \
VMAX, \
TARGET_UPDATE_RATE, \
MODEL_SAVE_RATE, \
REPLAY_MEMORY_SIZE, \
NUM_FRAMES, \
REWARD_PONG_PLAYER_RACKET_HITS_BALL, \
REWARD_PONG_PLAYER_RACKET_COVERS_BALL, \
REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR, \
REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC, \
REWARD_PONG_OPPONENT_RACKET_HITS_BALL, \
REWARD_PONG_OPPONENT_RACKET_COVERS_BALL, \
REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR, \
REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC, \
REWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL, \
REWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL, \
REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR, \
REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC, \
REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE, \
REWARD_FREEWAY_CHICKEN_VERTICAL_POSITION, \
REWARD_POTENTIAL_BASED \
    = ModelStorage.loadModel(MODEL_TO_LOAD)

# Override epsilon values
EPS_START = 0.0
EPS_END = 0.0

# Set seed to get reproducible results
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda(True) if USE_CUDA else autograd.Variable(
    *args, **kwargs)

# Initialize environment
env = EnvironmentBuilder.make_environment_with_wrappers(ENVIRONMENT.value, [
    EnvironmentWrapper.KEEP_ORIGINAL_OBSERVATION,
    EnvironmentWrapper.NOOP_RESET,
    EnvironmentWrapper.MAX_AND_SKIP,
    EnvironmentWrapper.EPISODIC_LIFE,
    EnvironmentWrapper.FIRE_RESET,
    EnvironmentWrapper.WARP_FRAME,
    EnvironmentWrapper.IMAGE_TO_PYTORCH,
])
# Reset environment
env.reset()

# Initialize and load policy net and target net
policy_net = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, NUM_ATOMS, VMIN, VMAX, USE_CUDA).to(
    device)
policy_net.load_state_dict(MODEL_STATE_DICT)
policy_net.eval()

# Initialize total variables
total_frames = 0

# Initialize the environment and state
state = env.reset()

# Iterate over frames
progress_bar = tqdm(iterable=range(NUM_FRAMES), unit='frames', desc="Play game")
for total_frames in progress_bar:
    # Select and perform an action
    action = ActionSelector.select_action(state=state,
                                          n_actions=env.action_space.n,
                                          total_frames=total_frames,
                                          policy_net=policy_net,
                                          epsilon_end=EPS_END,
                                          epsilon_start=EPS_START,
                                          epsilon_decay=EPS_DECAY,
                                          vmin=VMIN,
                                          vmax=VMAX,
                                          num_atoms=NUM_ATOMS,
                                          device=device,
                                          USE_CUDA=USE_CUDA)

    # Perform action
    observation, _, done, info = env.step(action)

    # Retrieve current screen
    screen = env.original_observation

    # Plot screen
    ScreenPlotter.save_screen_plot(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
                                   total_frames=total_frames,
                                   env=env,
                                   title="gif-screenshot",
                                   device=device,
                                   prune=False)

    # Move to the next state
    state = observation

    if done:
        # Reset the environment and state
        state = env.reset()

        if info["ale.lives"] == 0:
            list_of_screenshots = glob.glob(OUTPUT_DIRECTORY + RUN_TO_LOAD + "/gif-screenshot*")

            # Render gif
            images = []
            progress_bar_render = tqdm(sorted(list_of_screenshots), unit='frames', desc="Render gif")
            for filename in progress_bar_render:
                images.append(imageio.imread(filename))
            imageio.mimsave(OUTPUT_DIRECTORY + RUN_TO_LOAD + '/movie.gif', images)

            # Remove screenshots
            for filename in sorted(list_of_screenshots):
                os.remove(filename)

            print("Complete")
            sys.exit()
