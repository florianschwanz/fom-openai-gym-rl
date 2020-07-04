import glob
import os
import random
import sys
import time
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
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
common_reward_shaper_path = os.path.join(os.getcwd(), '..', 'common', 'reward_shaper')
if not (common_reward_shaper_path in sys.path):
    sys.path.insert(0, common_reward_shaper_path)

# Import library classes
from lib.action_selector import ActionSelector
from lib.breakout_reward_shaper import BreakoutRewardShaper
from lib.deep_q_network import DeepQNetwork
from lib.deep_q_network import Phi
from lib.deep_q_network import Gnet
from lib.deep_q_network import Fnet
from lib.environment_builder import EnvironmentBuilder
from lib.environment_builder import EnvironmentWrapper
from lib.environment_enum import Environment
from lib.freeway_reward_shaper import FreewayRewardShaper
from lib.model_optimizer import ModelOptimizer
from lib.model_storage import ModelStorage
from lib.input_extractor import InputExtractor
from lib.performance_logger import PerformanceLogger
from lib.performance_plotter import PerformancePlotter
from lib.pong_reward_shaper import PongRewardShaper
from lib.potential_based_reward_shaper import PotentialBasedRewardShaper
from lib.replay_memory import ReplayMemory
from lib.spaceinvaders_reward_shaper import SpaceInvadersRewardShaper
from lib.screen_animator import ScreenAnimator
from lib.screen_plotter import ScreenPlotter


# Path to output to be loaded
RUN_NAME = os.getenv('RUN_NAME', str(uuid.uuid4()))
RUN_TO_LOAD = os.getenv('RUN_TO_LOAD', None)
OUTPUT_DIRECTORY = os.getenv('OUTPUT_DIRECTORY', "./output/")
CONFIG_DIRECTORY = os.getenv('CONFIG_DIRECTORY', "./config/")

if RUN_TO_LOAD != None:
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
else:
    RUN_DIRECTORY = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    # Only use defined parameters if there is no previous output being loaded
    FINISHED_FRAMES = 0
    FINISHED_EPISODES = 0
    TOTAL_ORIGINAL_REWARDS = []
    TOTAL_SHAPED_REWARDS = []
    TOTAL_LOSSES = []

    # Define setup
    ENVIRONMENT_ID = os.getenv('ENVIRONMENT_ID', Environment.BREAKOUT_NO_FRAMESKIP_V0.value)
    ENVIRONMENT = Environment(ENVIRONMENT_ID)
    ENVIRONMENT_WRAPPERS = [
        EnvironmentWrapper.KEEP_ORIGINAL_OBSERVATION,
        EnvironmentWrapper.NOOP_RESET,
        EnvironmentWrapper.MAX_AND_SKIP,
        EnvironmentWrapper.EPISODIC_LIFE,
        EnvironmentWrapper.FIRE_RESET,
        EnvironmentWrapper.WARP_FRAME,
        EnvironmentWrapper.IMAGE_TO_PYTORCH,
        EnvironmentWrapper.FRAME_STACK,
    ]
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
    GAMMA = float(os.getenv('GAMMA', 0.99))
    EPS_START = float(os.getenv('EPS_START', 1.0))
    EPS_END = float(os.getenv('EPS_END', 0.01))
    EPS_DECAY = int(os.getenv('EPS_DECAY', 10_000))
    NUM_ATOMS = int(os.getenv('NUM_ATOMS', 51))
    VMIN = int(os.getenv('VMIN', -10))
    VMAX = int(os.getenv('VMAX', 10))
    TARGET_UPDATE_RATE = int(os.getenv('TARGET_UPDATE_RATE', 10))
    MODEL_SAVE_RATE = int(os.getenv('MODEL_SAVE_RATE', 10))
    REPLAY_MEMORY_SIZE = int(os.getenv('REPLAY_MEMORY_SIZE', 100_000))
    NUM_FRAMES = int(os.getenv('NUM_FRAMES', 1_000_000))

    #########################
    eta =float(os.getenv('eta', 1))
    beta =float(os.getenv('beta', 0.2))
    lambda1 =float(os.getenv('lambda', 0.1))
    #######################################


    REWARD_PONG_PLAYER_RACKET_HITS_BALL = float(os.getenv('REWARD_PONG_PLAYER_RACKET_HITS_BALL', 0.0))
    REWARD_PONG_PLAYER_RACKET_COVERS_BALL = float(os.getenv('REWARD_PONG_PLAYER_RACKET_COVERS_BALL', 0.0))
    REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR = float(
        os.getenv('REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR', 0.0))
    REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC = float(
        os.getenv('REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0))
    REWARD_PONG_OPPONENT_RACKET_HITS_BALL = float(os.getenv('REWARD_PONG_OPPONENT_RACKET_HITS_BALL', 0.0))
    REWARD_PONG_OPPONENT_RACKET_COVERS_BALL = float(os.getenv('REWARD_PONG_OPPONENT_RACKET_COVERS_BALL', 0.0))
    REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR = float(
        os.getenv('REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR', 0.0))
    REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC = float(
        os.getenv('REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0))
    REWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL = float(os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL', 0.0))
    REWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL = float(os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL', 0.0))
    REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR = float(
        os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR', 0.0))
    REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC = float(
        os.getenv('REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC', 0.0))
    REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE = float(
        os.getenv('REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE', 0.0))
    REWARD_FREEWAY_CHICKEN_VERTICAL_POSITION = float(os.getenv('REWARD_FREEWAY_CHICKEN_VERTICAL_POSITION', 0.0))
    REWARD_POTENTIAL_BASED = float(os.getenv('REWARD_POTENTIAL_BASED', 0.0))

    # Log parameters
    
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
    {"method": SpaceInvadersRewardShaper().reward_player_avoids_line_of_fire,
     "arguments": {"additional_reward": REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE}},
    {"method": FreewayRewardShaper().reward_chicken_vertical_position,
     "arguments": {"additional_reward": REWARD_FREEWAY_CHICKEN_VERTICAL_POSITION}},
    {"method": PotentialBasedRewardShaper().reward,
     "arguments": {"additional_reward": REWARD_POTENTIAL_BASED}},
]

# Set seed to get reproducible results
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable interactive mode of matplotlib
plt.ion()

# Initialize environment
env = EnvironmentBuilder.make_environment_with_wrappers(ENVIRONMENT.value, ENVIRONMENT_WRAPPERS)
# Reset environment
env.reset()

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = InputExtractor.get_screen(env=env, device=device)
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

# Only use defined parameters if there is no previous output being loaded
if RUN_TO_LOAD != None:
    # Initialize and load policy net and target net
    policy_net = DeepQNetwork(screen_height, screen_width, n_actions).to(device)
    policy_net.load_state_dict(MODEL_STATE_DICT)
    policy_net.eval()

    target_net = DeepQNetwork(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(MODEL_STATE_DICT)
    target_net.eval()
else:
    # Initialize policy net and target net
    policy_net = DeepQNetwork(screen_height, screen_width, n_actions).to(device)
    ######################################
    encoder = Phi().to(device)
    forward_model = Fnet().to(device)
    inverse_model = Gnet().to(device)
    ######################################
    target_net = DeepQNetwork(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

# Only use defined parameters if there is no previous output being loaded
if RUN_TO_LOAD != None:
    # Initialize and load optimizer
    optimizer = optim.RMSprop(policy_net.parameters())
    optimizer.load_state_dict(OPTIMIZER_STATE_DICT)

    # Load memory
    memory = REPLAY_MEMORY
else:
    # Initialize optimizer
    #optimizer = optim.RMSprop(policy_net.parameters())
    # Initialize replay memory
    memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    ######################################
    all_model_params = list(policy_net.parameters()) + list(encoder.parameters()) 
    all_model_params += list(forward_model.parameters()) + list(inverse_model.parameters())
    optimizer = optim.Adam(lr=0.001, params=all_model_params)
    forward_loss = torch.nn.MSELoss(reduction='none')#torch.nn.PairwiseDistance()#
    inverse_loss = torch.nn.CrossEntropyLoss(reduction='none')
    ######################################

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
env.reset()
last_screen = InputExtractor.get_screen(env=env, device=device)
current_screen = InputExtractor.get_screen(env=env, device=device)
state = current_screen - last_screen

# Iterate over frames
progress_bar = tqdm(iterable=range(NUM_FRAMES), unit='frames', initial=FINISHED_FRAMES, desc="Train model")
for total_frames in progress_bar:
    total_frames += FINISHED_FRAMES

    # Select action
    action = ActionSelector.select_action(state=state,
                                          n_actions=n_actions,
                                          total_frames=total_frames,
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

    # Iterate over all reward shaping mechanisms
    for reward_shaping in REWARD_SHAPINGS:
        if reward_shaping["arguments"]["additional_reward"] != 0:
            shaped_reward += reward_shaping["method"](environment=ENVIRONMENT,
                                                      screen=screen,
                                                      reward=reward,
                                                      done=done,
                                                      info=info,
                                                      **reward_shaping["arguments"],
                                                      current_episode_reward=(
                                                              episode_original_reward + original_reward),
                                                      max_episode_reward=max_episode_original_reward,
                                                      min_episode_reward=min_episode_original_reward
                                                      )

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
                                         device=device,
                                         all_model_params=all_model_params,
                                         eta=eta,
                                         beta=beta,
                                         lambda1=lambda1,
                                         inverse_loss= inverse_loss,
                                         forward_loss= forward_loss,
                                         encoder= encoder, 
                                         forward_model= forward_model, 
                                         inverse_model= inverse_model )

    #if total_episodes != 0 and total_episodes % MODEL_SAVE_RATE == 0 and total_frames % 2 == 0:
        # Plot screen for gif
     #   ScreenPlotter.save_screen_plot(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
      #                                 total_frames=total_frames,
       #                                env=env,
        #                               title="gif-screenshot",
         #                              device=device,
          #                             prune=False)

    if done:
        print(done)
        #print(reward.data.cpu().numpy()[0])
        print(episode_original_reward )
        # Reset the environment and state
        state = env.reset()
        last_screen = InputExtractor.get_screen(env=env, device=device)
        current_screen = InputExtractor.get_screen(env=env, device=device)
        state = current_screen - last_screen

        if info["ale.lives"] == 0:
            # Track episode time
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            total_duration = episode_end_time - total_start_time

            # Add rewards to total reward
            total_original_rewards.append(episode_original_reward)
            total_shaped_rewards.append(episode_shaped_reward)

            #if loss is not None:
             #   PerformanceLogger.log_episode(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
              #                                max_frames=NUM_FRAMES,
               #                               total_episodes=total_episodes + 1,
                #                              total_frames=total_frames,
                 #                             total_duration=total_duration,
                  #                            total_original_rewards=total_original_rewards,
                   #                           total_shaped_rewards=total_shaped_rewards,
                    #                          episode_frames=episode_frames + 1,
                     #                         episode_original_reward=episode_original_reward,
                      #                        episode_shaped_reward=episode_shaped_reward,
                       #                       episode_loss=loss.item(),
                        #                      episode_duration=episode_duration)

                # Update the target network, copying all weights and biases from policy net into target net
            if total_episodes != 0 and total_episodes % TARGET_UPDATE_RATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            #if total_episodes != 0 and total_episodes % MODEL_SAVE_RATE == 0:
                # Save output
                #ModelStorage.saveModel(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
                 #                      total_frames=total_frames,
                  #                     total_episodes=total_episodes,
                   #                    total_original_rewards=total_original_rewards,
                    #                   total_shaped_rewards=total_shaped_rewards,
                     #                  total_losses=total_losses,
                      #                 net=target_net,
                       #                optimizer=optimizer,
                        #               memory=memory,
                         #              loss=loss,
                          #             environment=ENVIRONMENT,
                           #            environment_wrappers=ENVIRONMENT_WRAPPERS,
                            #           batch_size=BATCH_SIZE,
                             #          gamma=GAMMA,
                              #         eps_start=EPS_START,
                               #        eps_end=EPS_END,
                                #       eps_decay=EPS_DECAY,
                                 #      num_atoms=NUM_ATOMS,
                                  #     vmin=VMIN,
                                   #    vmax=VMAX,
                                    #   target_update_rate=TARGET_UPDATE_RATE,
                                     #  model_save_rate=MODEL_SAVE_RATE,
                                      # replay_memory_size=REPLAY_MEMORY_SIZE,
                                       #num_frames=NUM_FRAMES,
                                      # reward_pong_player_racket_hits_ball=REWARD_PONG_PLAYER_RACKET_HITS_BALL,
                                      # reward_pong_player_racket_covers_ball=REWARD_PONG_PLAYER_RACKET_COVERS_BALL,
                                      # reward_pong_player_racket_close_to_ball_linear=REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR,
                                      # reward_pong_player_racket_close_to_ball_quadratic=REWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC,
                                      # reward_pong_opponent_racket_hits_ball=REWARD_PONG_OPPONENT_RACKET_HITS_BALL,
                                       #reward_pong_opponent_racket_covers_ball=REWARD_PONG_OPPONENT_RACKET_COVERS_BALL,
                                       #reward_pong_opponent_racket_close_to_ball_linear=REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR,
                                       #reward_pong_opponent_racket_close_to_ball_quadratic=REWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC,
                                       #reward_breakout_player_racket_hits_ball=REWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL,
                                       #reward_breakout_player_racket_covers_ball=REWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL,
                                       #reward_breakout_player_racket_close_to_ball_linear=REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR,
                                       #reward_breakout_player_racket_close_to_ball_quadratic=REWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC,
                                       #reward_spaceinvaders_player_avoids_line_of_fire=REWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE,
                                       #reward_freeway_chicken_vertical_position=REWARD_FREEWAY_CHICKEN_VERTICAL_POSITION,
                                       #reward_potential_based=REWARD_POTENTIAL_BASED
                                       #)

                #PerformancePlotter.save_values_plot(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
                 #                                   total_frames=total_frames,
                  #                                  values=total_original_rewards,
                   #                                 title="original rewards",
                    #                                xlabel="episode",
                     #                               ylabel="reward")

                #PerformancePlotter.save_values_plot(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
                 #                                   total_frames=total_frames,
                  #                                  values=total_shaped_rewards,
                   #                                 title="shaped rewards",
                    #                                xlabel="episode",
                     #                               ylabel="reward")

                #PerformancePlotter.save_values_plot(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
                 #                                   total_frames=total_frames,
                  #                                  values=total_losses,
                   #                                 title="losses",
                    #                                xlabel="frame",
                     #                               ylabel="loss")

                # ScreenPlotter.save_screen_plot(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
                #                                total_frames=total_frames,
                #                                env=env,
                #                                title="screenshot",
                #                                device=device)

                #ScreenAnimator.save_screen_animation(directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
                 #                                    total_episodes=total_episodes,
                  #                                   title="gif-screenshot")

                #elegramLogger.log_episode(run_name=RUN_NAME,
                 #                          output_directory=OUTPUT_DIRECTORY + RUN_DIRECTORY,
                  #                         conf_directory=CONFIG_DIRECTORY,
                   #                        max_frames=NUM_FRAMES,
                    #                       total_episodes=total_episodes + 1,
                     #                      total_frames=total_frames,
                      #                     total_duration=total_duration,
                       #                    total_original_rewards=total_original_rewards,
                        #                   total_shaped_rewards=total_shaped_rewards,
                         #                  episode_frames=episode_frames + 1,
                          #                 episode_original_reward=episode_original_reward,
                           #                episode_shaped_reward=episode_shaped_reward,
                            #               episode_loss=loss.item(),
                             #              episode_duration=episode_duration)

            # Reset episode variables
            episode_frames = 0
            episode_original_reward = 0
            episode_shaped_reward = 0
            episode_start_time = time.time()

            # Increment counter
            total_episodes += 1

    # Increment counter
    episode_frames += 1

print('Complete')
