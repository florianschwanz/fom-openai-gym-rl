import glob
import os
import sys
import time
from datetime import datetime

import torch
import torch.autograd as autograd
import torch.optim as optim
from tqdm import tqdm

# Make library available in path
lib_path = os.path.join(os.getcwd(), 'lib')
if not (lib_path in sys.path):
    sys.path.insert(0, lib_path)
common_lib_path = os.path.join(os.getcwd(), '..', 'common', 'lib')
if not (common_lib_path in sys.path):
    sys.path.insert(0, common_lib_path)

# Import library classes
from breakout_reward_shaper import BreakoutRewardShaper
from deep_q_network import RainbowCnnDQN
from environment_builder import EnvironmentBuilder
from environment_builder import EnvironmentWrapper
from environment_enum import Environment
from input_extractor import InputExtractor
from model_optimizer import ModelOptimizer
from model_storage import ModelStorage
from performance_logger import PerformanceLogger
from pong_reward_shaper import PongRewardShaper
from replay_buffer import ReplayBuffer
from spaceinvaders_reward_shaper import SpaceInvadersRewardShaper

# Path to output to be loaded
RUN_TO_LOAD = os.getenv('RUN_TO_LOAD', None)
OUTPUT_DIRECTORY = os.getenv('OUTPUT_DIRECTORY', "./output/")

if RUN_TO_LOAD != None:
    # Get latest file from run
    list_of_files = glob.glob(OUTPUT_DIRECTORY + RUN_TO_LOAD + "/*")
    MODEL_TO_LOAD = max(list_of_files, key=os.path.getctime)

    RUN_DIRECTORY = RUN_TO_LOAD

    FINISHED_FRAMES, \
    FINISHED_EPISODES, \
    MODEL_STATE_DICT, \
    OPTIMIZER_STATE_DICT, \
    REPLAY_MEMORY, \
    LOSS, \
 \
    ENVIRONMENT, \
    ENVIRONMENT_WRAPPERS, \
    BATCH_SIZE, \
    GAMMA, \
    NUM_ATOMS, \
    VMIN, \
    VMAX, \
    TARGET_UPDATE, \
    REPLAY_MEMORY_SIZE, \
    NUM_FRAMES, \
    REWARD_SHAPINGS, \
        = ModelStorage.loadModel(MODEL_TO_LOAD)
else:
    RUN_DIRECTORY = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # Only use defined parameters if there is no previous output being loaded
    FINISHED_FRAMES = 0
    FINISHED_EPISODES = 0

    # Define setup
    ENVIRONMENT_ID = os.getenv('ENVIRONMENT_ID', Environment.SPACE_INVADERS_V0.value)
    ENVIRONMENT = Environment(ENVIRONMENT_ID)
    ENVIRONMENT_WRAPPERS = [
        EnvironmentWrapper.KEEP_ORIGINAL_OBSERVATION,
        EnvironmentWrapper.NOOP_RESET,
        EnvironmentWrapper.MAX_AND_SKIP,
        EnvironmentWrapper.EPISODIC_LIFE,
        EnvironmentWrapper.FIRE_RESET,
        EnvironmentWrapper.WARP_FRAME,
        EnvironmentWrapper.IMAGE_TO_PYTORCH,
    ]
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    GAMMA = float(os.getenv('GAMMA', 0.99))
    NUM_ATOMS = int(os.getenv('NUM_ATOMS', 51))
    VMIN = int(os.getenv('VMIN', -10))
    VMAX = int(os.getenv('VMAX', 10))
    TARGET_UPDATE = int(os.getenv('TARGET_UPDATE', 1_000))
    REPLAY_MEMORY_SIZE = int(os.getenv('REPLAY_MEMORY', 100_000))
    NUM_FRAMES = int(os.getenv('NUM_FRAMES', 1_000_000))
    REWARD_SHAPINGS = [
        {"method": PongRewardShaper().reward_player_racket_hits_ball,
         "arguments": {"additional_reward": float(os.getenv('REWARD_PONG_PLAYER_RACKET_HITS_BALL', 0.0))}},
        {"method": PongRewardShaper().reward_player_racket_covers_ball,
         "arguments": {"additional_reward": float(os.getenv('REWARD_PONG_PLAYER_RACKET_COVERS_BALL', 0.0))}},
        {"method": PongRewardShaper().reward_player_racket_close_to_ball_linear,
         "arguments": {"additional_reward": float(os.getenv('REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR', 0.0))}},
        {"method": PongRewardShaper().reward_player_racket_close_to_ball_quadratic,
         "arguments": {"additional_reward": float(os.getenv('REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0))}},
        {"method": PongRewardShaper().reward_opponent_racket_hits_ball,
         "arguments": {"additional_reward": float(os.getenv('REWARD_PONG_OPPONENT_RACKET_HITS_BALL', 0.0))}},
        {"method": PongRewardShaper().reward_opponent_racket_covers_ball,
         "arguments": {"additional_reward": float(os.getenv('REWARD_PONG_OPPONENT_RACKET_COVERS_BALL', 0.0))}},
        {"method": PongRewardShaper().reward_opponent_racket_close_to_ball_linear,
         "arguments": {"additional_reward": float(os.getenv('REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR', 0.0))}},
        {"method": PongRewardShaper().reward_opponent_racket_close_to_ball_quadratic,
         "arguments": {"additional_reward": float(os.getenv('REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0))}},

        {"method": BreakoutRewardShaper().reward_player_racket_hits_ball,
         "arguments": {"additional_reward": float(os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL', 0.0))}},
        {"method": BreakoutRewardShaper().reward_player_racket_covers_ball,
         "arguments": {"additional_reward": float(os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL', 0.0))}},
        {"method": BreakoutRewardShaper().reward_player_racket_close_to_ball_linear,
         "arguments": {"additional_reward": float(os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR', 0.0))}},
        {"method": BreakoutRewardShaper().reward_player_racket_close_to_ball_quadratic,
         "arguments": {"additional_reward": float(os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0))}},

        {"method": SpaceInvadersRewardShaper().reward_player_avoids_line_of_fire,
         "arguments": {"additional_reward": float(os.getenv('REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE',
                                                            0.025))}},
    ]

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda(True) if USE_CUDA else autograd.Variable(
    *args, **kwargs)

# Initialize environment
env = EnvironmentBuilder.make_environment_with_wrappers(ENVIRONMENT.value, ENVIRONMENT_WRAPPERS)
# Reset environment
env.reset()

# Initialize policy net and target net
policy_net = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, NUM_ATOMS, VMIN, VMAX, USE_CUDA).to(device)
target_net = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, NUM_ATOMS, VMIN, VMAX, USE_CUDA).to(device)

# Initialize optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
# Initialize replay memory
memory = ReplayBuffer(REPLAY_MEMORY_SIZE)

# Initialize total variables
total_frames = 0
total_episodes = FINISHED_EPISODES
total_original_rewards = []
total_shaped_rewards = []
total_start_time = time.time()

# Initialize episode variables
episode_frames = 0
episode_original_reward = 0
episode_shaped_reward = 0
episode_start_time = time.time()

# Initialize the environment and state
state = env.reset()

# Iterate over frames
progress_bar = tqdm(range(NUM_FRAMES), unit='frames')
for total_frames in progress_bar:
    # Select and perform an action
    action = policy_net.act(state)

    # Perform action
    observation, reward, done, info = env.step(action)

    # Shape reward
    original_reward = reward
    shaped_reward = reward

    # Retrieve current screen
    screen = env.original_observation

    # Iterate overall reward shaping mechanisms
    for reward_shaping in REWARD_SHAPINGS:
        if reward_shaping["arguments"]["additional_reward"] != 0:
            shaped_reward += reward_shaping["method"](environment=ENVIRONMENT,
                                                      screen=screen,
                                                      reward=reward,
                                                      done=done,
                                                      info=info,
                                                      **reward_shaping["arguments"])

    # Plot intermediate screen
    # if total_frames % 50 == 0:
    #     InputExtractor.plot_screen(InputExtractor.get_sharp_screen(env=env, device=device), "Frame " + str(
    #         total_frames) + " / shaped reward " + str(round(shaped_reward, 4)))

    # Use shaped reward for further processing
    reward = shaped_reward

    # Add reward to episode reward
    episode_original_reward += original_reward
    episode_shaped_reward += shaped_reward

    # Update next state
    next_state = observation

    # Store the transition in memory
    memory.push(state, action, reward, next_state, done)

    # Move to the next state
    state = next_state

    # Perform one step of the optimization (on the target network)
    loss = ModelOptimizer.compute_td_loss(policy_net=policy_net,
                                          target_net=target_net,
                                          optimizer=optimizer,
                                          memory=memory,
                                          batch_size=BATCH_SIZE,
                                          num_atoms=NUM_ATOMS,
                                          vmin=VMIN,
                                          vmax=VMAX,
                                          USE_CUDA=USE_CUDA)

    if done:
        # Track episode time
        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time
        total_duration = episode_end_time - total_start_time

        # Add rewards to total reward
        total_original_rewards.append(episode_original_reward)
        total_shaped_rewards.append(episode_shaped_reward)

        if loss is not None:
            PerformanceLogger.log_episode(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
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
        if total_episodes % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

            # Save output
            ModelStorage.saveModel(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
                                   total_frames=total_frames,
                                   total_episodes=total_episodes,
                                   net=target_net,
                                   optimizer=optimizer,
                                   memory=memory,
                                   loss=loss,
                                   environment=ENVIRONMENT,
                                   environment_wrappers=ENVIRONMENT_WRAPPERS,
                                   batch_size=BATCH_SIZE,
                                   gamma=GAMMA,
                                   num_atoms=NUM_ATOMS,
                                   vmin=VMIN,
                                   vmax=VMAX,
                                   target_update=TARGET_UPDATE,
                                   replay_memory_size=REPLAY_MEMORY_SIZE,
                                   num_frames=NUM_FRAMES,
                                   reward_shapings=REWARD_SHAPINGS
                                   )

        # Reset episode variables
        episode_frames = 0
        episode_original_reward = 0
        episode_shaped_reward = 0
        episode_start_time = time.time()

        # Reset the environment and state
        state = env.reset()

        # Increment counter
        total_episodes += 1

    # Increment counter
    episode_frames += 1

print('Complete')
# env.render()
# env.close()
# plt.ioff()
# plt.show()
