import glob
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import torch
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
from action_selector import ActionSelector
from deep_q_network import DeepQNetwork
from environment_builder import EnvironmentBuilder
from environment_builder import EnvironmentWrapper
from environment_enum import Environment
from input_extractor import InputExtractor
from model_optimizer import ModelOptimizer
from model_storage import ModelStorage
from performance_logger import PerformanceLogger
from pong_reward_shaper import PongRewardShaper
from replay_memory import ReplayMemory

# Path to model to be loaded
RUN_TO_LOAD = os.getenv('RUN_TO_LOAD', None)
OUTPUT_DIRECTORY = os.getenv('OUTPUT_DIRECTORY', "./model/")

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
    ENVIRONMENT_NAME, \
    ENVIRONMENT_WRAPPERS, \
    BATCH_SIZE, \
    GAMMA, \
    EPS_START, \
    EPS_END, \
    EPS_DECAY, \
    TARGET_UPDATE, \
    REPLAY_MEMORY_SIZE, \
    NUM_FRAMES, \
    REWARD_SHAPINGS, \
        = ModelStorage.loadModel(MODEL_TO_LOAD)
else:
    RUN_DIRECTORY = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # Only use defined parameters if there is no previous model being loaded
    FINISHED_FRAMES = 0
    FINISHED_EPISODES = 0

    # Define setup
    ENVIRONMENT_NAME = os.getenv('ENVIRONMENT_NAME', Environment.PONG_NO_FRAMESKIP_v4)
    ENVIRONMENT_WRAPPERS = [
        EnvironmentWrapper.KEEP_ORIGINAL_OBSERVATION,
        EnvironmentWrapper.NOOP_RESET,
        EnvironmentWrapper.MAX_AND_SKIP,
        EnvironmentWrapper.EPISODIC_LIFE,
        EnvironmentWrapper.FIRE_RESET,
        EnvironmentWrapper.WARP_FRAME,
        EnvironmentWrapper.IMAGE_TO_PYTORCH,
    ]
    BATCH_SIZE = os.getenv('BATCH_SIZE', 32)
    GAMMA = os.getenv('GAMMA', 0.99)
    EPS_START = os.getenv('EPS_START', 1.0)
    EPS_END = os.getenv('EPS_END', 0.01)
    EPS_DECAY = os.getenv('EPS_DECAY', 500)
    TARGET_UPDATE = os.getenv('TARGET_UPDATE', 1_000)
    REPLAY_MEMORY_SIZE = os.getenv('REPLAY_MEMORY', 100_000)
    NUM_FRAMES = int(os.getenv('NUM_FRAMES', 1_000_000))
    REWARD_SHAPINGS = [
        {"method": PongRewardShaper().reward_player_racket_hits_ball,
         "arguments": {"additional_reward": os.getenv('REWARD_PONG_PLAYER_RACKET_HITS_BALL', 0.025)}},
        {"method": PongRewardShaper().reward_player_racket_covers_ball,
         "arguments": {"additional_reward": os.getenv('REWARD_PONG_PLAYER_RACKET_COVERS_BALL', 0.0)}},
        {"method": PongRewardShaper().reward_player_racket_close_to_ball_linear,
         "arguments": {"additional_reward": os.getenv('REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR', 0.05)}},
        {"method": PongRewardShaper().reward_player_racket_close_to_ball_quadratic,
         "arguments": {"additional_reward": os.getenv('REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0)}},
        {"method": PongRewardShaper().reward_opponent_racket_hits_ball,
         "arguments": {"additional_reward": os.getenv('REWARD_PONG_OPPONENT_RACKET_HITS_BALL', -0.025)}},
        {"method": PongRewardShaper().reward_opponent_racket_covers_ball,
         "arguments": {"additional_reward": os.getenv('REWARD_PONG_OPPONENT_RACKET_COVERS_BALL', 0.0)}},
        {"method": PongRewardShaper().reward_opponent_racket_close_to_ball_linear,
         "arguments": {"additional_reward": os.getenv('REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR', -0.05)}},
        {"method": PongRewardShaper().reward_opponent_racket_close_to_ball_quadratic,
         "arguments": {"additional_reward": os.getenv('REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0)}},
    ]

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable interactive mode of matplotlib
plt.ion()

# Initialize environment
env = EnvironmentBuilder.make_environment_with_wrappers(ENVIRONMENT_NAME.value, ENVIRONMENT_WRAPPERS)
# Reset environment
env.reset()

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = InputExtractor.get_screen(env=env, device=device)
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

# Only use defined parameters if there is no previous model being loaded
if RUN_TO_LOAD != None:
    # Initialize and load policy net and target net
    policy_net = DeepQNetwork(screen_height, screen_width, n_actions).to(device)
    policy_net.load_state_dict(MODEL_STATE_DICT)

    target_net = DeepQNetwork(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(MODEL_STATE_DICT)
else:
    # Initialize policy net and target net
    policy_net = DeepQNetwork(screen_height, screen_width, n_actions).to(device)

    target_net = DeepQNetwork(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

# Only use defined parameters if there is no previous model being loaded
if RUN_TO_LOAD != None:
    # Initialize and load optimizer
    optimizer = optim.RMSprop(policy_net.parameters())
    optimizer.load_state_dict(OPTIMIZER_STATE_DICT)

    # Load memory
    memory = REPLAY_MEMORY
else:
    # Initialize optimizer
    optimizer = optim.RMSprop(policy_net.parameters())
    # Initialize replay memory
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)

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
env.reset()
last_screen = InputExtractor.get_screen(env=env, device=device)
current_screen = InputExtractor.get_screen(env=env, device=device)
state = current_screen - last_screen

# Iterate over frames
progress_bar = tqdm(range(NUM_FRAMES), unit='frames')
progress_bar.update(FINISHED_FRAMES)
for total_frames in progress_bar:

    # Select action
    action = ActionSelector.select_action(state=state,
                                          n_actions=n_actions,
                                          policy_net=policy_net,
                                          epsilon_end=EPS_END,
                                          epsilon_start=EPS_START,
                                          epsilon_decay=EPS_DECAY,
                                          device=device)

    # Perform action
    observation, reward, done, info = env.step(action.item())

    # Shape reward
    original_reward = reward
    shaped_reward = reward

    # Retrieve current screen
    screen = observation

    # Iterate overall reward shaping mechanisms
    for reward_shaping in REWARD_SHAPINGS:
        if reward_shaping["arguments"]["additional_reward"] != 0:
            shaped_reward += reward_shaping["method"](environment_name=ENVIRONMENT_NAME,
                                                      screen=screen,
                                                      reward=reward,
                                                      done=done,
                                                      info=info,
                                                      **reward_shaping["arguments"])

    # # Plot intermediate screen
    # if total_frames % 50 == 0:
    #     InputExtractor.plot_screen(InputExtractor.get_sharp_screen(env=env, device=device), "Frame " + str(
    #         total_frames) + " / shaped reward " + str(round(shaped_reward, 4)))

    # Use shaped reward for further processing
    reward = shaped_reward

    # Add reward to episode reward
    episode_original_reward += original_reward
    episode_shaped_reward += shaped_reward

    # Transform reward into a tensor
    reward = torch.tensor([reward], device=device)

    # Observe new state
    last_screen = current_screen
    current_screen = InputExtractor.get_screen(env=env, device=device)

    # Update next state
    next_state = current_screen - last_screen

    # Store the transition in memory
    memory.push(state, action, next_state, reward)

    # Move to the next state
    state = next_state

    # Perform one step of the optimization (on the target network)
    loss = ModelOptimizer.optimize_model(policy_net=policy_net,
                                         target_net=target_net,
                                         optimizer=optimizer,
                                         memory=memory,
                                         batch_size=BATCH_SIZE,
                                         gamma=GAMMA,
                                         device=device)

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

            # Save model
            ModelStorage.saveModel(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
                                   total_frames=total_frames,
                                   total_episodes=total_episodes,
                                   net=target_net,
                                   optimizer=optimizer,
                                   memory=memory,
                                   loss=loss,
                                   environment_name=ENVIRONMENT_NAME,
                                   environment_wrappers=ENVIRONMENT_WRAPPERS,
                                   batch_size=BATCH_SIZE,
                                   gamma=GAMMA,
                                   eps_start=EPS_START,
                                   eps_end=EPS_END,
                                   eps_decay=EPS_DECAY,
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
        env.reset()
        last_screen = InputExtractor.get_screen(env=env, device=device)
        current_screen = InputExtractor.get_screen(env=env, device=device)
        state = current_screen - last_screen

        # Increment counter
        total_episodes += 1

    # Increment counter
    episode_frames += 1

print('Complete')
# env.render()
# env.close()
# plt.ioff()
# plt.show()
