import os
import sys

import numpy as np
import torch
import torch.autograd as autograd
import torch.optim as optim

# Make library available in path
lib_path = os.path.join(os.getcwd(), 'lib')
if not (lib_path in sys.path):
    sys.path.insert(0, lib_path)

# Import library classes
from deep_q_network import RainbowCnnDQN
from environment_enum import Environment
from model_optimizer import ModelOptimizer
from replay_buffer import ReplayBuffer
from tqdm import tqdm
from wrappers import make_atari, wrap_deepmind, wrap_pytorch

# Define setup
ENVIRONMENT_NAME = Environment.PONG_NO_FRAMESKIP_v4
BATCH_SIZE = 32
GAMMA = 0.99
NUM_ATOMS = 51
VMIN = -10
VMAX = 10
REPLAY_INITIAL = 20_000
REPLAY_MEMORY_SIZE = 100_000
NUM_FRAMES = 1_000_000

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda(True) if USE_CUDA else autograd.Variable(
    *args, **kwargs)


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

# Initialize environment
env = make_atari(ENVIRONMENT_NAME)
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

losses = []
all_rewards = []
episode_reward = 0

# Initialize the environment and state
state = env.reset()

# Iterate over frames
progress_bar = tqdm(range(NUM_FRAMES), unit='frames')
for total_frames in progress_bar:
    # Select and perform an action
    action = policy_net.act(state)

    # Perform action
    next_state, reward, done, _ = env.step(action)

    # Store the transition in memory
    memory.push(state, action, reward, next_state, done)

    # Move to the next state
    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    # Perform one step of the optimization (on the target network)
    if len(memory) > REPLAY_INITIAL:
        loss = ModelOptimizer.compute_td_loss(policy_net=policy_net,
                                              target_net=target_net,
                                              optimizer=optimizer,
                                              memory=memory,
                                              batch_size=BATCH_SIZE,
                                              num_atoms=NUM_ATOMS,
                                              vmin=VMIN,
                                              vmax=VMAX,
                                              USE_CUDA=USE_CUDA)
        losses.append(loss.item())

    if total_frames % 3000 == 0:
        avg_return = np.mean(all_rewards[-50:])
        print("Average return (last 50 episodes): {:.2f}".format(avg_return))

    if total_frames % 1000 == 0:
        update_target(policy_net, target_net)

policy_net.act()
