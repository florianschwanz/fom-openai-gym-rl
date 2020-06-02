import os
import sys
import time
from itertools import count

import gym
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm_notebook as tqdm

# Make library available in path
lib_path = os.path.join(os.getcwd(), 'lib')
if not (lib_path in sys.path):
    sys.path.insert(0, lib_path)

# Import library classes
from replay_memory import ReplayMemory
from deep_q_network import DeepQNetwork
from action_selector import ActionSelector
from input_extractor import InputExtractor
from model_optimizer import ModelOptimizer
from environment_enum import Environment
from pong_reward_shaper import PongRewardShaper
from reward_shape_enum import RewardShape
from performance_logger import PerformanceLogger

# Define setup
ENVIRONMENT_NAME = Environment.PONG_DETERMINISTIC_v4
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
REPLAY_MEMORY_SIZE = 10_000
NUM_EPISODES = 5_000
REWARD_SHAPINGS = [
]

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable interactive mode of matplotlib
plt.ion()

# Initialize environment
env = gym.make(ENVIRONMENT_NAME.value).unwrapped
# Reset environment
env.reset()
# Plot initial screen
InputExtractor.plot_screen(InputExtractor.get_sharp_screen(env=env, device=device), 'Example extracted screen')

######################################################################
# Training
# --------
#
# Hyper-parameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = InputExtractor.get_screen(env=env, device=device)
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

# Initialize policy net and target net
policy_net = DeepQNetwork(screen_height, screen_width, n_actions).to(device)
target_net = DeepQNetwork(screen_height, screen_width, n_actions).to(device)

# Since both nets are initialized randomly we need to copy the state of one into the other to make sure they are equal
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Initialize optimizer
optimizer = optim.RMSprop(policy_net.parameters())
# Initialize replay memory
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` Tensor. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more episodes, such as 300+ for meaningful
# duration improvements.
#

episode_original_reward = 0
episode_shaped_reward = 0
episode_losses = []

total_original_rewards = []
total_shaped_rewards = []

total_frames = 0
total_start_time = time.time()

# Iterate over episodes
progress_bar = tqdm(range(NUM_EPISODES), unit='episode')
for i_episode in progress_bar:

    episode_start_time = time.time()

    # Reset episode variables
    episode_original_reward = 0
    episode_shaped_reward = 0

    # Initialize the environment and state
    env.reset()
    last_screen = InputExtractor.get_screen(env=env, device=device)
    current_screen = InputExtractor.get_screen(env=env, device=device)
    state = current_screen - last_screen

    # Run episode until status done is reached
    for i_frame in count():

        total_frames += 1

        # Select and perform an action
        action = ActionSelector.select_action(state=state,
                                              n_actions=n_actions,
                                              policy_net=policy_net,
                                              epsilon_end=EPS_END,
                                              epsilon_start=EPS_START,
                                              epsilon_decay=EPS_DECAY,
                                              device=device)

        # Do step
        observation, reward, done, info = env.step(action.item())

        # Shape reward
        original_reward = reward
        shaped_reward = reward

        if ENVIRONMENT_NAME == Environment.PONG_v0 \
                or ENVIRONMENT_NAME == Environment.PONG_v4 \
                or ENVIRONMENT_NAME == Environment.PONG_DETERMINISTIC_v0 \
                or ENVIRONMENT_NAME == Environment.PONG_DETERMINISTIC_v4 \
                or ENVIRONMENT_NAME == Environment.PONG_NO_FRAMESKIP_v0 \
                or ENVIRONMENT_NAME == Environment.PONG_NO_FRAMESKIP_v4:
            # Plot screen after scoring
            if original_reward == 1:
                InputExtractor.plot_screen(InputExtractor.get_sharp_screen(env=env, device=device), 'GOOOOAAAAL!')

            reward_shaper = PongRewardShaper(observation, reward, done, info)

            if RewardShape.PONG_CENTER_RACKET_ON_BALL in REWARD_SHAPINGS:
                shaped_reward += reward_shaper.reward_center_ball()
            if RewardShape.PONG_RACKET_CLOSE_TO_BALL in REWARD_SHAPINGS:
                shaped_reward += reward_shaper.reward_close_to_ball()
            if RewardShape.PONG_PROXIMITY_TO_BALL_LINEAR in REWARD_SHAPINGS:
                shaped_reward += reward_shaper.reward_vertical_proximity_to_ball_linear()
            if RewardShape.PONG_PROXIMITY_TO_BALL_QUADRATIC in REWARD_SHAPINGS:
                shaped_reward += reward_shaper.reward_vertical_proximity_to_ball_quadratic()

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

        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

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
                PerformanceLogger.log_episode_short(total_episodes=i_episode + 1,
                                                    total_frames=total_frames,
                                                    total_duration=total_duration,
                                                    total_original_rewards=total_original_rewards,
                                                    total_shaped_rewards=total_shaped_rewards,
                                                    episode_frames=i_frame + 1,
                                                    episode_original_reward=episode_original_reward,
                                                    episode_shaped_reward=episode_shaped_reward,
                                                    episode_loss=loss.item(),
                                                    episode_duration=episode_duration)
            break

    # Update the target network, copying all weights and biases from policy net into target net
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
