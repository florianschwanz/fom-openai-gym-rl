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

# Import library classes
from deep_q_network import RainbowCnnDQN
from environment_enum import Environment
from model_optimizer import ModelOptimizer
from performance_logger import PerformanceLogger
from replay_buffer import ReplayBuffer
from wrappers import make_atari, wrap_deepmind, wrap_pytorch

RUN_DIRECTORY = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

# Only use defined parameters if there is no previous model being loaded
FINISHED_FRAMES = 0
FINISHED_EPISODES = 0

# Define setup
ENVIRONMENT_NAME = os.getenv('ENVIRONMENT_NAME', Environment.PONG_NO_FRAMESKIP_v4)
BATCH_SIZE = os.getenv('BATCH_SIZE', 32)
GAMMA = os.getenv('GAMMA', 0.99)
NUM_ATOMS = os.getenv('NUM_ATOMS', 51)
VMIN = os.getenv('VMIN', -10)
VMAX = os.getenv('VMAX', 10)
TARGET_UPDATE = os.getenv('TARGET_UPDATE', 1_000)
REPLAY_MEMORY_SIZE = os.getenv('REPLAY_MEMORY', 100_000)
NUM_FRAMES = int(os.getenv('NUM_FRAMES', 1_000_000))

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda(True) if USE_CUDA else autograd.Variable(
    *args, **kwargs)


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


# Initialize environment
env = make_atari(ENVIRONMENT_NAME.value)
env = wrap_deepmind(env, frame_stack=True, scale=False)
env = wrap_pytorch(env)

# Initialize policy net and target net
policy_net = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, NUM_ATOMS, VMIN, VMAX, USE_CUDA).to(device)
target_net = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, NUM_ATOMS, VMIN, VMAX, USE_CUDA).to(device)

# Initialize optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=0.0001)
update_target(policy_net, target_net)
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
    next_state, reward, done, _ = env.step(action)

    # Shape reward
    original_reward = reward
    shaped_reward = reward

    # Add reward to episode reward
    episode_original_reward += original_reward
    episode_shaped_reward += shaped_reward

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
            PerformanceLogger.log_episode(directory=RUN_DIRECTORY,
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
