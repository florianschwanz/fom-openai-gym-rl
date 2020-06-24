import glob
import os

import torch


class ModelStorage:
    # Maximum number of files we want to store
    MAX_FILES = 3

    def saveModel(directory,
                  total_frames,
                  total_episodes,
                  total_original_rewards,
                  total_shaped_rewards,
                  total_losses,
                  net,
                  optimizer,
                  memory,
                  loss,
                  environment,
                  environment_wrappers,
                  batch_size,
                  learning_rate,
                  gamma,
                  eps_start,
                  eps_end,
                  eps_decay,
                  num_atoms,
                  vmin,
                  vmax,
                  target_update_rate,
                  model_save_rate,
                  replay_memory_size,
                  num_frames,
                  reward_pong_player_racket_hits_ball,
                  reward_pong_player_racket_covers_ball,
                  reward_pong_player_racket_close_to_ball_linear,
                  reward_pong_player_racket_close_to_ball_quadratic,
                  reward_pong_opponent_racket_hits_ball,
                  reward_pong_opponent_racket_covers_ball,
                  reward_pong_opponent_racket_close_to_ball_linear,
                  reward_pong_opponent_racket_close_to_ball_quadratic,
                  reward_breakout_player_racket_hits_ball,
                  reward_breakout_player_racket_covers_ball,
                  reward_breakout_player_racket_close_to_ball_linear,
                  reward_breakout_player_racket_close_to_ball_quadratic,
                  reward_spaceinvaders_player_avoids_line_of_fire,
                  reward_freeway_distance_walked,
                  reward_freeway_distance_to_car,
                  reward_potential_based):
        """
        Saves output into a file
        """

        # Make path if not yet exists
        if not os.path.exists(directory):
            os.mkdir(directory)

        torch.save({
            'total_frames': total_frames,
            'total_episodes': total_episodes,
            'total_original_rewards': total_original_rewards,
            'total_shaped_rewards': total_shaped_rewards,
            'total_losses': total_losses,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'replay_memory': memory,
            'loss': loss,
            'environment': environment,
            'environment_wrappers': environment_wrappers,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'eps_start': eps_start,
            'eps_end': eps_end,
            'eps_decay': eps_decay,
            'num_atoms': num_atoms,
            'vmin': vmin,
            'vmax': vmax,
            'target_update_rate': target_update_rate,
            'model_save_rate': model_save_rate,
            'replay_memory_size': replay_memory_size,
            'num_frames': num_frames,
            'reward_pong_player_racket_hits_ball': reward_pong_player_racket_hits_ball,
            'reward_pong_player_racket_covers_ball': reward_pong_player_racket_covers_ball,
            'reward_pong_player_racket_close_to_ball_linear': reward_pong_player_racket_close_to_ball_linear,
            'reward_pong_player_racket_close_to_ball_quadratic': reward_pong_player_racket_close_to_ball_quadratic,
            'reward_pong_opponent_racket_hits_ball': reward_pong_opponent_racket_hits_ball,
            'reward_pong_opponent_racket_covers_ball': reward_pong_opponent_racket_covers_ball,
            'reward_pong_opponent_racket_close_to_ball_linear': reward_pong_opponent_racket_close_to_ball_linear,
            'reward_pong_opponent_racket_close_to_ball_quadratic': reward_pong_opponent_racket_close_to_ball_quadratic,
            'reward_breakout_player_racket_hits_ball': reward_breakout_player_racket_hits_ball,
            'reward_breakout_player_racket_covers_ball': reward_breakout_player_racket_covers_ball,
            'reward_breakout_player_racket_close_to_ball_linear': reward_breakout_player_racket_close_to_ball_linear,
            'reward_breakout_player_racket_close_to_ball_quadratic': reward_breakout_player_racket_close_to_ball_quadratic,
            'reward_spaceinvaders_player_avoids_line_of_fire': reward_spaceinvaders_player_avoids_line_of_fire,
            'reward_freeway_distance_walked': reward_freeway_distance_walked,
            'reward_freeway_distance_to_car': reward_freeway_distance_to_car,
            'reward_potential_based': reward_potential_based
        }, directory + "/target_net-frame-{:07d}".format(total_frames) + ".model")

        # Prune old model files
        ModelStorage.prune_storage(directory)

    def loadModel(path):
        """
        Loads output from a given path
        :param path path to saved output
        :return:
        """
        checkpoint = torch.load(path)

        return checkpoint['total_frames'], \
               checkpoint['total_episodes'], \
               checkpoint['total_original_rewards'], \
               checkpoint['total_shaped_rewards'], \
               checkpoint['total_losses'], \
               checkpoint['model_state_dict'], \
               checkpoint['optimizer_state_dict'], \
               checkpoint['replay_memory'], \
               checkpoint['loss'], \
               checkpoint['environment'], \
               checkpoint['environment_wrappers'], \
               checkpoint['batch_size'], \
               checkpoint['learning_rate'], \
               checkpoint['gamma'], \
               checkpoint['eps_start'], \
               checkpoint['eps_end'], \
               checkpoint['eps_decay'], \
               checkpoint['num_atoms'], \
               checkpoint['vmin'], \
               checkpoint['vmax'], \
               checkpoint['target_update_rate'], \
               checkpoint['model_save_rate'], \
               checkpoint['replay_memory_size'], \
               checkpoint['num_frames'], \
               checkpoint['reward_pong_player_racket_hits_ball'], \
               checkpoint['reward_pong_player_racket_covers_ball'], \
               checkpoint['reward_pong_player_racket_close_to_ball_linear'], \
               checkpoint['reward_pong_player_racket_close_to_ball_quadratic'], \
               checkpoint['reward_pong_opponent_racket_hits_ball'], \
               checkpoint['reward_pong_opponent_racket_covers_ball'], \
               checkpoint['reward_pong_opponent_racket_close_to_ball_linear'], \
               checkpoint['reward_pong_opponent_racket_close_to_ball_quadratic'], \
               checkpoint['reward_breakout_player_racket_hits_ball'], \
               checkpoint['reward_breakout_player_racket_covers_ball'], \
               checkpoint['reward_breakout_player_racket_close_to_ball_linear'], \
               checkpoint['reward_breakout_player_racket_close_to_ball_quadratic'], \
               checkpoint['reward_spaceinvaders_player_avoids_line_of_fire'], \
               checkpoint['reward_freeway_distance_walked'], \
               checkpoint['reward_freeway_distance_to_car'], \
               checkpoint['reward_potential_based']

    def prune_storage(directory):
        list_of_files = glob.glob(directory + "/*.model")

        while len(list_of_files) > ModelStorage.MAX_FILES:
            oldest_file = min(list_of_files, key=os.path.getctime)
            os.remove(oldest_file)
            list_of_files = glob.glob(directory + "/*.model")
