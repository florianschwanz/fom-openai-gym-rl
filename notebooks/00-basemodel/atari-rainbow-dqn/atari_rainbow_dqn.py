import os
import sys

# Make library available in path
lib_path = os.path.join(os.getcwd(), 'lib')
if not (lib_path in sys.path):
    sys.path.insert(0, lib_path)

import numpy as np

import torch
import torch.optim as optim
import torch.autograd as autograd

from deep_q_network import RainbowCnnDQN
from model_optimizer import ModelOptimizer
from replay_buffer import ReplayBuffer

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda(True) if USE_CUDA else autograd.Variable(
    *args, **kwargs)


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


# update_target(current_model, target_model)




from wrappers import make_atari, wrap_deepmind, wrap_pytorch

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env, frame_stack=True, scale=False)
env = wrap_pytorch(env)

NUM_ATOMS = 20  # Initial 51
VMIN = -5  # Initial -10
VMAX = 5  # Initial 10

policy_net = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, NUM_ATOMS, VMIN, VMAX, USE_CUDA)
target_net = RainbowCnnDQN(env.observation_space.shape, env.action_space.n, NUM_ATOMS, VMIN, VMAX, USE_CUDA)

if USE_CUDA:
    policy_net = policy_net.cuda()
    target_net = target_net.cuda()

optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)  # Initial: 0.0001
update_target(policy_net, target_net)

replay_initial = 20000
memory = ReplayBuffer(30000)  # Initial:100.000

num_frames = 1000000
from tqdm import tqdm_notebook as tqdm

# update_target(current_model, target_model)
# current_model = torch.load('model')

BATCH_SIZE = 64  # Initial 32
GAMMA = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = env.reset()

# Iterate over frames
progress_bar = tqdm(range(1000000), unit='episode')
for frame_idx in progress_bar:
    # Select and perform an action
    action = policy_net.act(state)

    # Perform action
    next_state, reward, done, _ = env.step(action)


    memory.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(memory) > replay_initial:
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

    if frame_idx % 3000 == 0:
        avg_return = np.mean(all_rewards[-50:])
        print("Average return (last 50 episodes): {:.2f}".format(avg_return))

    if frame_idx % 1000 == 0:
        update_target(policy_net, target_net)

policy_net.act()
