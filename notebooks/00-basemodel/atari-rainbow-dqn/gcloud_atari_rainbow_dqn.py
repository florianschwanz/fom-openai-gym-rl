#! /opt/conda/bin/python3

import glob
import os
import random
import sys
import time
import uuid
from datetime import datetime

import torch
import torch.optim as optim
from tqdm import tqdm

old_stdout = sys.stdout
log_file = open("./output/training.log", "w")
sys.stdout = log_file

# Make library available in path
library_paths = [
    os.path.join(os.getcwd(), 'lib'),
    os.path.join(os.getcwd(), '..', 'common', 'environment'),
    os.path.join(os.getcwd(), '..', 'common', 'log'),
    os.path.join(os.getcwd(), '..', 'common', 'persistence'),
    os.path.join(os.getcwd(), '..', 'common', 'plot'),
    os.path.join(os.getcwd(), '..', 'common', 'reward_shaper'),
]

for p in library_paths:
    if not (p in sys.path):
        sys.path.insert(0, p)

# Import library classes
from action_selector import ActionSelector
from breakout_reward_shaper import BreakoutRewardShaper
from deep_q_network import RainbowCnnDQN
from environment_builder import EnvironmentBuilder
from environment_builder import EnvironmentWrapper
from environment_enum import Environment
from freeway_reward_shaper import FreewayRewardShaper
from logger_facade import LoggerFacade
from model_optimizer import ModelOptimizer
from model_optimizer_curiosity import ModelOptimizerCuriosity
from model_storage import ModelStorage
from performance_plotter import PerformancePlotter
from pong_reward_shaper import PongRewardShaper
from potential_based_reward_shaper import PotentialBasedRewardShaper
from replay_memory import ReplayMemory
from spaceinvaders_reward_shaper import SpaceInvadersRewardShaper
from screen_animator import ScreenAnimator
from screen_plotter import ScreenPlotter
from telegram_logger import TelegramLogger

from curiosity_net import Phi
from curiosity_net import Gnet
from curiosity_net import Fnet

# Path to output to be loaded
RUN_NAME = os.getenv('RUN_NAME', str(uuid.uuid4()))
RUN_TO_LOAD = os.getenv('RUN_TO_LOAD', None)
OUTPUT_DIRECTORY = os.getenv('OUTPUT_DIRECTORY', "./output")
CONFIG_DIRECTORY = os.getenv('CONFIG_DIRECTORY', "./config")
TELEGRAM_CONFIG_FILE = os.getenv('TELEGRAM_CONFIG_FILE', None)

if RUN_TO_LOAD != None:
    RUN_DIRECTORY = RUN_TO_LOAD + "-" + RUN_NAME

    ModelStorage.unzip_model(OUTPUT_DIRECTORY, RUN_DIRECTORY)

    TARGET_NET_STATE_DICT = ModelStorage.loadNet(OUTPUT_DIRECTORY, RUN_DIRECTORY, "target")
    ENCODER_NET_STATE_DICT = ModelStorage.loadNet(OUTPUT_DIRECTORY, RUN_DIRECTORY, "encoder")
    FORWARD_MODEL_NET_STATE_DICT = ModelStorage.loadNet(OUTPUT_DIRECTORY, RUN_DIRECTORY, "forward_model")
    INVERSE_MODEL_NET_STATE_DICT = ModelStorage.loadNet(OUTPUT_DIRECTORY, RUN_DIRECTORY, "inverse_model")

    OPTIMIZER_STATE_DICT = ModelStorage.loadOptimizer(OUTPUT_DIRECTORY, RUN_DIRECTORY)
    REPLAY_MEMORY_CHUNKS = ModelStorage.loadMemoryChunks(OUTPUT_DIRECTORY, RUN_DIRECTORY)
    ENVIRONMENT, ENVIRONMENT_WRAPPERS = ModelStorage.loadEnvironment(OUTPUT_DIRECTORY, RUN_DIRECTORY)

    BATCH_SIZE, \
    LEARNING_RATE, \
    GAMMA, \
    EPS_START, \
    EPS_END, \
    EPS_DECAY, \
    NUM_ATOMS, \
    VMIN, \
    VMAX, \
 \
    ETA, \
    BETA, \
    LAMBDA1, \
 \
    NORMALIZE_SHAPED_REWARD, \
    REWARD_SHAPING_DROPOUT_RATE, \
    TARGET_UPDATE_RATE, \
    MODEL_SAVE_RATE, \
    EPISODE_LOG_RATE, \
    REPLAY_MEMORY_SIZE, \
    NUM_FRAMES = ModelStorage.loadConfig(OUTPUT_DIRECTORY, RUN_DIRECTORY)

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
    REWARD_BREAKOUT_BALL_HITTING_UPPER_BLOCK, \
    REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE, \
    REWARD_FREEWAY_DISTANCE_WALKED, \
    REWARD_FREEWAY_DISTANCE_TO_CAR, \
    REWARD_POTENTIAL_BASED = ModelStorage.loadRewards(OUTPUT_DIRECTORY, RUN_DIRECTORY)

    FINISHED_FRAMES, \
    FINISHED_EPISODES, \
    TOTAL_ORIGINAL_REWARDS, \
    TOTAL_SHAPED_REWARDS, \
    TOTAL_LOSSES = ModelStorage.loadStats(OUTPUT_DIRECTORY, RUN_DIRECTORY)
else:
    RUN_DIRECTORY = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    ENVIRONMENT_ID = os.getenv('ENVIRONMENT_ID', Environment.BREAKOUT_NO_FRAMESKIP_V4.value)
    ENVIRONMENT = Environment(ENVIRONMENT_ID)
    ENVIRONMENT_WRAPPERS = [
        EnvironmentWrapper.KEEP_ORIGINAL_OBSERVATION,
        EnvironmentWrapper.NOOP_RESET,
        EnvironmentWrapper.MAX_AND_SKIP,
        EnvironmentWrapper.EPISODIC_LIFE,
        EnvironmentWrapper.FIRE_RESET,
        EnvironmentWrapper.WARP_FRAME,
        EnvironmentWrapper.CLIP_REWARD,
        EnvironmentWrapper.FRAME_STACK,
        EnvironmentWrapper.IMAGE_TO_PYTORCH,
    ]

    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.0001))
    GAMMA = float(os.getenv('GAMMA', 0.99))
    EPS_START = float(os.getenv('EPS_START', 1.0))
    EPS_END = float(os.getenv('EPS_END', 0.01))
    EPS_DECAY = int(os.getenv('EPS_DECAY', 10_000))
    NUM_ATOMS = int(os.getenv('NUM_ATOMS', 51))
    VMIN = int(os.getenv('VMIN', -10))
    VMAX = int(os.getenv('VMAX', 10))

    ETA = float(os.getenv('eta', 0.0))
    BETA = float(os.getenv('beta', 0.0))
    LAMBDA1 = float(os.getenv('lambda', 0.0))

    NORMALIZE_SHAPED_REWARD = os.getenv('NORMALIZE_SHAPED_REWARD', False) == "True"
    REWARD_SHAPING_DROPOUT_RATE = float(os.getenv('REWARD_SHAPING_DROPOUT_RATE', 0.0))
    TARGET_UPDATE_RATE = int(os.getenv('TARGET_UPDATE_RATE', 10))
    MODEL_SAVE_RATE = int(os.getenv('MODEL_SAVE_RATE', 10))
    EPISODE_LOG_RATE = int(os.getenv('EPISODE_LOG_RATE', 10))
    REPLAY_MEMORY_SIZE = int(os.getenv('REPLAY_MEMORY', 100_000))
    NUM_FRAMES = int(os.getenv('NUM_FRAMES', 1_000_000))

    REWARD_PONG_PLAYER_RACKET_HITS_BALL = float(os.getenv('REWARD_PONG_PLAYER_RACKET_HITS_BALL', 0.0))
    REWARD_PONG_PLAYER_RACKET_COVERS_BALL = float(os.getenv('REWARD_PONG_PLAYER_RACKET_COVERS_BALL', 0.0))
    REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR = float(os.getenv('REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR', 0.0))
    REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC = float(os.getenv('REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0))
    REWARD_PONG_OPPONENT_RACKET_HITS_BALL = float(os.getenv('REWARD_PONG_OPPONENT_RACKET_HITS_BALL', 0.0))
    REWARD_PONG_OPPONENT_RACKET_COVERS_BALL = float(os.getenv('REWARD_PONG_OPPONENT_RACKET_COVERS_BALL', 0.0))
    REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR = float(os.getenv('REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR', 0.0))
    REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC = float(os.getenv('REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0))
    REWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL = float(os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL', 0.0))
    REWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL = float(os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL', 0.0))
    REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR = float(os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR', 0.0))
    REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC = float(os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0))
    REWARD_BREAKOUT_BALL_HITTING_UPPER_BLOCK = float(os.getenv('REWARD_BREAKOUT_BALL_HITTING_UPPER_BLOCK', 0.0))
    REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE = float(os.getenv('REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE', 0.0))
    REWARD_FREEWAY_DISTANCE_WALKED = float(os.getenv('REWARD_FREEWAY_DISTANCE_WALKED', 0.0))
    REWARD_FREEWAY_DISTANCE_TO_CAR = float(os.getenv('REWARD_FREEWAY_DISTANCE_TO_CAR', 0.0))
    REWARD_POTENTIAL_BASED = float(os.getenv('REWARD_POTENTIAL_BASED', 0.0))

    FINISHED_FRAMES = 0
    FINISHED_EPISODES = 0
    TOTAL_ORIGINAL_REWARDS = []
    TOTAL_SHAPED_REWARDS = []
    TOTAL_LOSSES = []

    # Log parameters
    LoggerFacade.log_parameters(run_name=RUN_NAME,
                                output_directory=OUTPUT_DIRECTORY,
                                run_directory=RUN_DIRECTORY,
                                conf_directory=CONFIG_DIRECTORY,
                                conf_file=TELEGRAM_CONFIG_FILE,
                                environment_id=ENVIRONMENT_ID,
                                batch_size=BATCH_SIZE,
                                learning_rate=LEARNING_RATE,
                                gamma=GAMMA,
                                eps_start=EPS_START,
                                eps_end=EPS_END,
                                eps_decay=EPS_DECAY,
                                num_atoms=NUM_ATOMS,
                                vmin=VMIN,
                                vmax=VMAX,
                                eta=ETA,
                                beta=BETA,
                                lambda1=LAMBDA1,
                                normalize_shaped_reward=NORMALIZE_SHAPED_REWARD,
                                reward_shaping_dropout_rate=REWARD_SHAPING_DROPOUT_RATE,
                                target_update_rate=TARGET_UPDATE_RATE,
                                model_save_rate=MODEL_SAVE_RATE,
                                episode_log_rate=EPISODE_LOG_RATE,
                                replay_memory_size=REPLAY_MEMORY_SIZE,
                                num_frames=NUM_FRAMES,
                                reward_pong_player_racket_hits_ball=REWARD_PONG_PLAYER_RACKET_HITS_BALL,
                                reward_pong_player_racket_covers_ball=REWARD_PONG_PLAYER_RACKET_COVERS_BALL,
                                reward_pong_player_racket_close_to_ball_linear=REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR,
                                reward_pong_player_racket_close_to_ball_quadratic=REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC,
                                reward_pong_opponent_racket_hits_ball=REWARD_PONG_OPPONENT_RACKET_HITS_BALL,
                                reward_pong_opponent_racket_covers_ball=REWARD_PONG_OPPONENT_RACKET_COVERS_BALL,
                                reward_pong_opponent_racket_close_to_ball_linear=REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR,
                                reward_pong_opponent_racket_close_to_ball_quadratic=REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC,
                                reward_breakout_player_racket_hits_ball=REWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL,
                                reward_breakout_player_racket_covers_ball=REWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL,
                                reward_breakout_player_racket_close_to_ball_linear=REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR,
                                reward_breakout_player_racket_close_to_ball_quadratic=REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC,
                                reward_breakout_ball_hitting_upper_block=REWARD_BREAKOUT_BALL_HITTING_UPPER_BLOCK,
                                reward_spaceinvaders_player_avoids_line_of_fire=REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE,
                                reward_freeway_distance_walked=REWARD_FREEWAY_DISTANCE_WALKED,
                                reward_freeway_distance_to_car=REWARD_FREEWAY_DISTANCE_TO_CAR,
                                reward_potential_based=REWARD_POTENTIAL_BASED)

# Assemble reward shapings
REWARD_SHAPINGS = [
    {"method": PongRewardShaper().reward_player_racket_hits_ball,
     "arguments": {"additional_reward": REWARD_PONG_PLAYER_RACKET_HITS_BALL}},
    {"method": PongRewardShaper().reward_player_racket_covers_ball,
     "arguments": {"additional_reward": REWARD_PONG_PLAYER_RACKET_COVERS_BALL}},
    {"method": PongRewardShaper().reward_player_racket_close_to_ball_linear,
     "arguments": {"additional_reward": REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR}},
    {"method": PongRewardShaper().reward_player_racket_close_to_ball_quadratic,
     "arguments": {"additional_reward": REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC}},
    {"method": PongRewardShaper().reward_opponent_racket_hits_ball,
     "arguments": {"additional_reward": REWARD_PONG_OPPONENT_RACKET_HITS_BALL}},
    {"method": PongRewardShaper().reward_opponent_racket_covers_ball,
     "arguments": {"additional_reward": REWARD_PONG_OPPONENT_RACKET_COVERS_BALL}},
    {"method": PongRewardShaper().reward_opponent_racket_close_to_ball_linear,
     "arguments": {"additional_reward": REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR}},
    {"method": PongRewardShaper().reward_opponent_racket_close_to_ball_quadratic,
     "arguments": {"additional_reward": REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC}},
    {"method": BreakoutRewardShaper().reward_player_racket_hits_ball,
     "arguments": {"additional_reward": REWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL}},
    {"method": BreakoutRewardShaper().reward_player_racket_covers_ball,
     "arguments": {"additional_reward": REWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL}},
    {"method": BreakoutRewardShaper().reward_player_racket_close_to_ball_linear,
     "arguments": {"additional_reward": REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR}},
    {"method": BreakoutRewardShaper().reward_player_racket_close_to_ball_quadratic,
     "arguments": {"additional_reward": REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC}},
    {"method": BreakoutRewardShaper().reward_ball_hitting_upper_block,
     "arguments": {"additional_reward": REWARD_BREAKOUT_BALL_HITTING_UPPER_BLOCK}},
    {"method": SpaceInvadersRewardShaper().reward_player_avoids_line_of_fire,
     "arguments": {"additional_reward": REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE}},
    {"method": FreewayRewardShaper().reward_distance_walked,
     "arguments": {"additional_reward": REWARD_FREEWAY_DISTANCE_WALKED}},
    {"method": FreewayRewardShaper().reward_distance_to_car,
     "arguments": {"additional_reward": REWARD_FREEWAY_DISTANCE_TO_CAR}},
    {"method": PotentialBasedRewardShaper().reward,
     "arguments": {"additional_reward": REWARD_POTENTIAL_BASED}},
]

# Set seed to get reproducible results
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = EnvironmentBuilder.make_environment_with_wrappers(ENVIRONMENT.value, ENVIRONMENT_WRAPPERS)
# Reset environment
env.reset()

# Only use defined parameters if there is no previous output being loaded
if RUN_TO_LOAD != None:
    # Initialize and load policy net
    policy_net = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, NUM_ATOMS, VMIN, VMAX)
    policy_net.load_state_dict(TARGET_NET_STATE_DICT)
    policy_net.to(device)
    policy_net.eval()

    # Initialize and load nets for curiosity approach
    encoder = Phi(env.observation_space.shape).to(device)
    forward_model = Fnet(env.action_space.n).to(device)
    inverse_model = Gnet().to(device)

    if ENCODER_NET_STATE_DICT != None:
        encoder.load_state_dict(ENCODER_NET_STATE_DICT)
        encoder.to(device)
        encoder.eval()
    if FORWARD_MODEL_NET_STATE_DICT != None:
        forward_model.load_state_dict(FORWARD_MODEL_NET_STATE_DICT)
        forward_model.to(device)
        forward_model.eval()
    if INVERSE_MODEL_NET_STATE_DICT != None:
        inverse_model.load_state_dict(INVERSE_MODEL_NET_STATE_DICT)
        inverse_model.to(device)
        inverse_model.eval()
else:
    # Initialize policy net
    policy_net = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, NUM_ATOMS, VMIN, VMAX)
    policy_net.to(device)

    # Initialize nets for curiosity approach
    encoder = Phi(env.observation_space.shape).to(device)
    forward_model = Fnet(env.action_space.n).to(device)
    inverse_model = Gnet().to(device)

# Copy target net from policy net
target_net = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, NUM_ATOMS, VMIN, VMAX).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Check if curiosity driven exploration should be used
if ETA == 0 and BETA == 0 and LAMBDA1 == 0:
    # Define optimizer parameters
    optimizer_parameters = policy_net.parameters()
else:
    # Define optimizer parameters
    optimizer_parameters = policy_net.parameters() + encoder.parameters() + forward_model.parameters() + inverse_model.parameters()

# Only use defined parameters if there is no previous output being loaded
if RUN_TO_LOAD != None:
    # Initialize and load optimizer
    optimizer = optim.Adam(params=optimizer_parameters, lr=LEARNING_RATE)
    optimizer.load_state_dict(OPTIMIZER_STATE_DICT)

    # Load replay memory
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    for chunk in REPLAY_MEMORY_CHUNKS:
        memory.append_storage_chunk(chunk)
else:
    # Initialize optimizer
    optimizer = optim.Adam(params=optimizer_parameters, lr=LEARNING_RATE)
    # Initialize replay memory
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)

# Initialize total variables
total_frames = 0
total_episodes = FINISHED_EPISODES
total_original_rewards = TOTAL_ORIGINAL_REWARDS
total_shaped_rewards = TOTAL_SHAPED_REWARDS
total_losses = TOTAL_LOSSES
total_start_time = time.time()

# Initialize episode variables
episode_frames = 0
episode_original_reward = 0
episode_shaped_reward = 0
episode_start_time = time.time()

# Initialize additional variables
max_episode_original_reward = None
min_episode_original_reward = None

# Initialize the environment and state
state = env.reset()

# Iterate over frames
progress_bar = tqdm(iterable=range(NUM_FRAMES), unit='frames', initial=FINISHED_FRAMES, desc="Train model")
for total_frames in progress_bar:
    total_frames += FINISHED_FRAMES

    # Select action
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
                                          device=device)

    # Perform action
    observation, original_reward, done, info = env.step(action)

    # Retrieve current screen
    screen = env.original_observation

    # Track potentially max value of shaped reward
    shaped_reward = 0
    shaped_reward_max = 0

    # Check if reward shaping should be applied
    if REWARD_SHAPING_DROPOUT_RATE == 0.0 or random.random() > REWARD_SHAPING_DROPOUT_RATE:
        # Iterate over all reward shaping mechanisms
        for reward_shaping in REWARD_SHAPINGS:
            method = reward_shaping["method"]
            additional_reward = reward_shaping["arguments"]["additional_reward"]

            if additional_reward != 0:
                shaped_reward_max += additional_reward
                shaped_reward += method(environment=ENVIRONMENT,
                                        screen=screen,
                                        reward=original_reward,
                                        done=done,
                                        info=info,
                                        **reward_shaping["arguments"],
                                        current_episode_reward=(
                                                episode_original_reward + original_reward),
                                        max_episode_reward=max_episode_original_reward,
                                        min_episode_reward=min_episode_original_reward)

    # Normalize shaped reward
    if NORMALIZE_SHAPED_REWARD and shaped_reward_max != 0:
        shaped_reward /= shaped_reward_max

    # Track episode rewards
    episode_original_reward += original_reward
    episode_shaped_reward += shaped_reward

    # Add shaped reward to original reward
    total_reward = original_reward + shaped_reward

    # Update next state
    next_state = observation

    # Store the transition in memory
    memory.push(state, action, total_reward, next_state, done)

    # Move to the next state
    state = next_state

    # Check if curiosity driven exploration should be used
    if ETA == 0 and BETA == 0 and LAMBDA1 == 0:
        # Perform one step of the optimization (on the target network)
        loss = ModelOptimizer.compute_td_loss(policy_net=policy_net,
                                              target_net=target_net,
                                              optimizer=optimizer,
                                              memory=memory,
                                              batch_size=BATCH_SIZE,
                                              gamma=GAMMA,
                                              num_atoms=NUM_ATOMS,
                                              vmin=VMIN,
                                              vmax=VMAX)
    else:
        forward_loss = torch.nn.MSELoss(reduction='none')
        inverse_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # Perform one step of the optimization (on the target network)
        loss = ModelOptimizerCuriosity.compute_td_loss(policy_net=policy_net,
                                                       target_net=target_net,
                                                       optimizer=optimizer,
                                                       memory=memory,
                                                       batch_size=BATCH_SIZE,
                                                       gamma=GAMMA,
                                                       num_atoms=NUM_ATOMS,
                                                       vmin=VMIN,
                                                       vmax=VMAX,
                                                       all_model_params=optimizer,
                                                       n_actions=env.action_space.n,
                                                       eta=ETA,
                                                       beta=BETA,
                                                       lambda1=LAMBDA1,
                                                       inverse_loss=inverse_loss,
                                                       forward_loss=forward_loss,
                                                       encoder=encoder,
                                                       forward_model=forward_model,
                                                       inverse_model=inverse_model)

    # Add loss to total loss
    total_losses.append(loss)

    if total_episodes != 0 and EPISODE_LOG_RATE != -1 \
            and (total_episodes + 1) % EPISODE_LOG_RATE == 0 \
            and total_frames % 2 == 0:

        if shaped_reward != 0:
            title = "frame " + str(total_frames) + " / s " + str(round(shaped_reward, 2))
        else:
            title = "frame " + str(total_frames)

        # Plot screen for gif
        ScreenPlotter.save_screen_plot(output_directory=OUTPUT_DIRECTORY,
                                       run_directory=RUN_DIRECTORY,
                                       total_frames=total_frames,
                                       env=env,
                                       name="gif-screenshot",
                                       title=title,
                                       device=device,
                                       prune=False)

    if done:
        # Reset the environment and state
        state = env.reset()

        if info["ale.lives"] == 0:
            # Track episode time
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            total_duration = episode_end_time - total_start_time

            # Add rewards to total reward
            total_original_rewards.append(episode_original_reward)
            total_shaped_rewards.append(episode_shaped_reward)

            # Update max and min episode rewards
            if max_episode_original_reward == None or episode_original_reward > max_episode_original_reward:
                max_episode_original_reward = episode_original_reward
            if min_episode_original_reward == None or episode_original_reward < min_episode_original_reward:
                min_episode_original_reward = episode_original_reward

            if loss is not None:
                LoggerFacade.log_episode(run_name=RUN_NAME,
                                         output_directory=OUTPUT_DIRECTORY,
                                         run_directory=RUN_DIRECTORY,
                                         conf_directory=CONFIG_DIRECTORY,
                                         conf_file=None,
                                         max_frames=NUM_FRAMES,
                                         total_episodes=total_episodes + 1,
                                         total_frames=total_frames,
                                         total_duration=total_duration,
                                         total_original_rewards=total_original_rewards,
                                         total_shaped_rewards=total_shaped_rewards,
                                         episode_frames=episode_frames + 1,
                                         episode_original_reward=episode_original_reward,
                                         episode_shaped_reward=episode_shaped_reward,
                                         episode_loss=loss.item(),
                                         episode_duration=episode_duration)

            # Update the target network, copying all weights and biases from policy net into target net
            if total_episodes != 0 and (total_episodes + 1) % TARGET_UPDATE_RATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if total_episodes != 0 \
                    and MODEL_SAVE_RATE != -1 \
                    and (total_episodes + 1) % MODEL_SAVE_RATE == 0:
                ModelStorage.saveNet(output_directory=OUTPUT_DIRECTORY,
                                     run_directory=RUN_DIRECTORY,
                                     total_frames=total_frames,
                                     net=target_net.to("cpu"),
                                     name="target")

                ModelStorage.saveNet(output_directory=OUTPUT_DIRECTORY,
                                     run_directory=RUN_DIRECTORY,
                                     total_frames=total_frames,
                                     net=encoder.to("cpu"),
                                     name="encoder")

                ModelStorage.saveNet(output_directory=OUTPUT_DIRECTORY,
                                     run_directory=RUN_DIRECTORY,
                                     total_frames=total_frames,
                                     net=forward_model.to("cpu"),
                                     name="forward_model")

                ModelStorage.saveNet(output_directory=OUTPUT_DIRECTORY,
                                     run_directory=RUN_DIRECTORY,
                                     total_frames=total_frames,
                                     net=inverse_model.to("cpu"),
                                     name="inverse_model")

                ModelStorage.saveOptimizer(output_directory=OUTPUT_DIRECTORY,
                                           run_directory=RUN_DIRECTORY,
                                           total_frames=total_frames,
                                           optimizer=optimizer)

                ModelStorage.saveMemoryChunks(output_directory=OUTPUT_DIRECTORY,
                                              run_directory=RUN_DIRECTORY,
                                              total_frames=total_frames,
                                              memory_chunks=memory.get_storage_chunks(int(REPLAY_MEMORY_SIZE / 10)))

                ModelStorage.saveEnvironment(output_directory=OUTPUT_DIRECTORY,
                                             run_directory=RUN_DIRECTORY,
                                             total_frames=total_frames,
                                             environment=ENVIRONMENT,
                                             environment_wrappers=ENVIRONMENT_WRAPPERS)

                ModelStorage.saveConfig(output_directory=OUTPUT_DIRECTORY,
                                        run_directory=RUN_DIRECTORY,
                                        total_frames=total_frames,
                                        batch_size=BATCH_SIZE,
                                        learning_rate=LEARNING_RATE,
                                        gamma=GAMMA,
                                        eps_start=EPS_START,
                                        eps_end=EPS_END,
                                        eps_decay=EPS_DECAY,
                                        num_atoms=NUM_ATOMS,
                                        vmin=VMIN,
                                        vmax=VMAX,
                                        eta=ETA,
                                        beta=BETA,
                                        lambda1=LAMBDA1,
                                        normalize_shaped_reward=NORMALIZE_SHAPED_REWARD,
                                        reward_shaping_dropout_rate=REWARD_SHAPING_DROPOUT_RATE,
                                        target_update_rate=TARGET_UPDATE_RATE,
                                        model_save_rate=MODEL_SAVE_RATE,
                                        episode_log_rate=EPISODE_LOG_RATE,
                                        replay_memory_size=REPLAY_MEMORY_SIZE,
                                        num_frames=NUM_FRAMES)

                ModelStorage.saveRewards(output_directory=OUTPUT_DIRECTORY,
                                         run_directory=RUN_DIRECTORY,
                                         total_frames=total_frames,
                                         reward_pong_player_racket_hits_ball=REWARD_PONG_PLAYER_RACKET_HITS_BALL,
                                         reward_pong_player_racket_covers_ball=REWARD_PONG_PLAYER_RACKET_COVERS_BALL,
                                         reward_pong_player_racket_close_to_ball_linear=REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR,
                                         reward_pong_player_racket_close_to_ball_quadratic=REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC,
                                         reward_pong_opponent_racket_hits_ball=REWARD_PONG_OPPONENT_RACKET_HITS_BALL,
                                         reward_pong_opponent_racket_covers_ball=REWARD_PONG_OPPONENT_RACKET_COVERS_BALL,
                                         reward_pong_opponent_racket_close_to_ball_linear=REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR,
                                         reward_pong_opponent_racket_close_to_ball_quadratic=REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC,
                                         reward_breakout_player_racket_hits_ball=REWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL,
                                         reward_breakout_player_racket_covers_ball=REWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL,
                                         reward_breakout_player_racket_close_to_ball_linear=REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR,
                                         reward_breakout_player_racket_close_to_ball_quadratic=REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC,
                                         reward_breakout_ball_hitting_upper_block=REWARD_BREAKOUT_BALL_HITTING_UPPER_BLOCK,
                                         reward_spaceinvaders_player_avoids_line_of_fire=REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE,
                                         reward_freeway_distance_walked=REWARD_FREEWAY_DISTANCE_WALKED,
                                         reward_freeway_distance_to_car=REWARD_FREEWAY_DISTANCE_TO_CAR,
                                         reward_potential_based=REWARD_POTENTIAL_BASED)

                ModelStorage.saveStats(output_directory=OUTPUT_DIRECTORY,
                                       run_directory=RUN_DIRECTORY,
                                       total_frames=total_frames,
                                       total_episodes=total_episodes,
                                       total_original_rewards=total_original_rewards,
                                       total_shaped_rewards=total_shaped_rewards,
                                       total_losses=total_losses)

                ModelStorage.zip_model(output_directory=OUTPUT_DIRECTORY,
                                       run_directory=RUN_DIRECTORY,
                                       total_frames=total_frames)

            # Move back target net to device
            target_net.to(device)

            if total_episodes != 0 and EPISODE_LOG_RATE != -1 and (total_episodes + 1) % EPISODE_LOG_RATE == 0:
                PerformancePlotter.save_values_plot(output_directory=OUTPUT_DIRECTORY,
                                                    run_directory=RUN_DIRECTORY,
                                                    total_frames=total_frames,
                                                    values=total_original_rewards,
                                                    title="original rewards",
                                                    xlabel="episode",
                                                    ylabel="reward")

                PerformancePlotter.save_values_plot(output_directory=OUTPUT_DIRECTORY,
                                                    run_directory=RUN_DIRECTORY,
                                                    total_frames=total_frames,
                                                    values=total_shaped_rewards,
                                                    title="shaped rewards",
                                                    xlabel="episode",
                                                    ylabel="reward")

                PerformancePlotter.save_values_plot(output_directory=OUTPUT_DIRECTORY,
                                                    run_directory=RUN_DIRECTORY,
                                                    total_frames=total_frames,
                                                    values=total_losses,
                                                    title="losses",
                                                    xlabel="frame",
                                                    ylabel="loss")

                ScreenAnimator.save_screen_animation(output_directory=OUTPUT_DIRECTORY,
                                                     run_directory=RUN_DIRECTORY,
                                                     total_episodes=total_episodes,
                                                     title="gif-screenshot")

                TelegramLogger.log_episode(run_name=RUN_NAME,
                                         output_directory=OUTPUT_DIRECTORY,
                                         run_directory=RUN_DIRECTORY,
                                         conf_directory=CONFIG_DIRECTORY,
                                         conf_file=TELEGRAM_CONFIG_FILE,
                                         max_frames=NUM_FRAMES,
                                         total_episodes=total_episodes + 1,
                                         total_frames=total_frames,
                                         total_duration=total_duration,
                                         total_original_rewards=total_original_rewards,
                                         total_shaped_rewards=total_shaped_rewards,
                                         episode_frames=episode_frames + 1,
                                         episode_original_reward=episode_original_reward,
                                         episode_shaped_reward=episode_shaped_reward,
                                         episode_loss=loss.item(),
                                         episode_duration=episode_duration)

            # Reset reward shapers
            BreakoutRewardShaper().reset_reward_shaper()

            # Reset episode variables
            episode_frames = 0
            episode_original_reward = 0
            episode_shaped_reward = 0
            episode_start_time = time.time()

            # Make sure to break iteration even when restarted
            if total_episodes >= NUM_FRAMES:
                break

            # Increment counter
            total_episodes += 1

    # Increment counter
    episode_frames += 1

LoggerFacade.log_results(run_name=RUN_NAME,
                         output_directory=OUTPUT_DIRECTORY,
                         run_directory=RUN_DIRECTORY,
                         conf_directory=CONFIG_DIRECTORY,
                         conf_file=TELEGRAM_CONFIG_FILE)

print('Complete')
