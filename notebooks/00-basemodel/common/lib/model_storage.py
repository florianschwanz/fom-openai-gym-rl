import glob
import os

import torch


class ModelStorage:
    # Maximum number of files we want to store
    MAX_FILES = 2

    FILE_EXTENTION_NET = ".net.pickle"
    FILE_EXTENTION_OPTIMIZER = ".optimizer.pickle"
    FILE_EXTENTION_MEMORY = ".memory.pickle"
    FILE_EXTENTION_ENVIRONMENT = ".environment.pickle"
    FILE_EXTENTION_CONFIG = ".config.pickle"
    FILE_EXTENTION_REWARDS = ".rewards.pickle"
    FILE_EXTENTION_STATS = ".stats.pickle"

    def saveNet(output_directory, run_directory, total_frames, net, name):
        file_extension = "." + name + ModelStorage.FILE_EXTENTION_NET
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)

        torch.save({
            'net_state_dict': net.state_dict(),
        }, target_directory + "/frame-{:07d}".format(total_frames) + file_extension)

        ModelStorage.prune_storage(target_directory, file_extension)

    def loadNet(output_directory, run_directory, name):
        file_extension = "." + name + ModelStorage.FILE_EXTENTION_NET
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)
        checkpoint = ModelStorage.load_checkpoint(target_directory, file_extension)

        return checkpoint['net_state_dict']

    def saveOptimizer(output_directory, run_directory, total_frames, optimizer):
        file_extension = ModelStorage.FILE_EXTENTION_OPTIMIZER
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)

        torch.save({
            'optimizer_state_dict': optimizer.state_dict()
        }, target_directory + "/frame-{:07d}".format(total_frames) + file_extension)

        ModelStorage.prune_storage(target_directory, file_extension)

    def loadOptimizer(output_directory, run_directory):
        file_extension = ModelStorage.FILE_EXTENTION_OPTIMIZER
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)
        checkpoint = ModelStorage.load_checkpoint(target_directory, file_extension)

        return checkpoint['optimizer_state_dict']

    MEMORY_CHUNKS = 5

    def saveMemoryChunks(output_directory, run_directory, total_frames, memory_chunks):
        file_extension = ModelStorage.FILE_EXTENTION_MEMORY
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)

        for index, chunk in enumerate(memory_chunks):
            file_extension =  "." + str(index) +  ModelStorage.FILE_EXTENTION_MEMORY

            torch.save({
                'replay_memory_chunk': chunk,
            }, target_directory + "/frame-{:07d}".format(total_frames) + file_extension)

            ModelStorage.prune_storage(target_directory, file_extension)

    def loadMemoryChunks(output_directory, run_directory):
        file_extension = ModelStorage.FILE_EXTENTION_MEMORY
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)

        list_of_files = glob.glob(target_directory + "/*" + file_extension)

        chunks = []

        for file in list_of_files:
            checkpoint = ModelStorage.load_checkpoint_file(file)
            chunks.append(checkpoint['replay_memory_chunk'])

        return chunks

    def saveEnvironment(output_directory, run_directory, total_frames, environment, environment_wrappers):
        file_extension = ModelStorage.FILE_EXTENTION_ENVIRONMENT
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)

        torch.save({
            'environment': environment,
            'environment_wrappers': environment_wrappers,
        }, target_directory + "/frame-{:07d}".format(total_frames) + file_extension)

        ModelStorage.prune_storage(target_directory, file_extension)

    def loadEnvironment(output_directory, run_directory):
        file_extension = ModelStorage.FILE_EXTENTION_ENVIRONMENT
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)
        checkpoint = ModelStorage.load_checkpoint(target_directory, file_extension)

        return checkpoint['environment'], \
               checkpoint['environment_wrappers']

    def saveConfig(output_directory, run_directory, total_frames,
                   batch_size,
                   learning_rate,
                   gamma,
                   eps_start,
                   eps_end,
                   eps_decay,
                   num_atoms,
                   vmin,
                   vmax,
                   eta,
                   beta,
                   lambda1,
                   normalize_shaped_reward,
                   reward_shaping_dropout_rate,
                   target_update_rate,
                   model_save_rate,
                   episode_log_rate,
                   replay_memory_size,
                   num_frames,
                   ):
        file_extension = ModelStorage.FILE_EXTENTION_CONFIG
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)

        torch.save({
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'eps_start': eps_start,
            'eps_end': eps_end,
            'eps_decay': eps_decay,
            'num_atoms': num_atoms,
            'vmin': vmin,
            'vmax': vmax,
            'eta': eta,
            'beta': beta,
            'lambda1': lambda1,
            'normalize_shaped_reward': normalize_shaped_reward,
            'reward_shaping_dropout_rate': reward_shaping_dropout_rate,
            'target_update_rate': target_update_rate,
            'model_save_rate': model_save_rate,
            'episode_log_rate': episode_log_rate,
            'replay_memory_size': replay_memory_size,
            'num_frames': num_frames
        }, target_directory + "/frame-{:07d}".format(total_frames) + file_extension)

        ModelStorage.prune_storage(target_directory, file_extension)

    def loadConfig(output_directory, run_directory):
        file_extension = ModelStorage.FILE_EXTENTION_CONFIG
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)
        checkpoint = ModelStorage.load_checkpoint(target_directory, file_extension)

        return checkpoint['batch_size'], \
               checkpoint['learning_rate'], \
               checkpoint['gamma'], \
               checkpoint['eps_start'], \
               checkpoint['eps_end'], \
               checkpoint['eps_decay'], \
               checkpoint['num_atoms'], \
               checkpoint['vmin'], \
               checkpoint['vmax'], \
               checkpoint['eta'], \
               checkpoint['beta'], \
               checkpoint['lambda1'], \
               checkpoint['normalize_shaped_reward'], \
               checkpoint['reward_shaping_dropout_rate'], \
               checkpoint['target_update_rate'], \
               checkpoint['model_save_rate'], \
               checkpoint['episode_log_rate'], \
               checkpoint['replay_memory_size'], \
               checkpoint['num_frames']

    def saveRewards(output_directory, run_directory, total_frames,
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
                    reward_potential_based
                    ):
        file_extension = ModelStorage.FILE_EXTENTION_REWARDS
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)

        torch.save({
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
            'reward_potential_based': reward_potential_based,
        }, target_directory + "/frame-{:07d}".format(total_frames) + file_extension)

        ModelStorage.prune_storage(target_directory, file_extension)

    def loadRewards(output_directory, run_directory):
        file_extension = ModelStorage.FILE_EXTENTION_REWARDS
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)
        checkpoint = ModelStorage.load_checkpoint(target_directory, file_extension)

        return checkpoint['reward_pong_player_racket_hits_ball'], \
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

    def saveStats(output_directory, run_directory, total_frames,
                  total_episodes,
                  total_original_rewards,
                  total_shaped_rewards,
                  total_losses):
        file_extension = ModelStorage.FILE_EXTENTION_STATS
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)

        torch.save({
            'total_frames': total_frames,
            'total_episodes': total_episodes,
            'total_original_rewards': total_original_rewards,
            'total_shaped_rewards': total_shaped_rewards,
            'total_losses': total_losses
        }, target_directory + "/frame-{:07d}".format(total_frames) + file_extension)

        ModelStorage.prune_storage(target_directory, file_extension)

    def loadStats(output_directory, run_directory):
        file_extension = ModelStorage.FILE_EXTENTION_STATS
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)
        checkpoint = ModelStorage.load_checkpoint(target_directory, file_extension)

        return checkpoint['total_frames'], \
               checkpoint['total_episodes'], \
               checkpoint['total_original_rewards'], \
               checkpoint['total_shaped_rewards'], \
               checkpoint['total_losses']

    def prepare_directory(output_directory, run_directory):
        target_directory = output_directory + "/" + run_directory
        symlink_directory = "latest"

        # Make path if not yet exists
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        if not os.path.exists(target_directory):
            os.mkdir(target_directory)

        # Create symlink
        if os.path.islink(output_directory + "/" + symlink_directory):
            os.unlink(output_directory + "/" + symlink_directory)
        os.symlink(run_directory, output_directory + "/" + symlink_directory)

        return target_directory

    def load_checkpoint(target_directory, file_extension):
        list_of_files = glob.glob(target_directory + "/*" + file_extension)
        file_to_load = max(list_of_files, key=os.path.getctime)
        return torch.load(file_to_load)

    def load_checkpoint_file(file_to_load):
        return torch.load(file_to_load)

    def prune_storage(prune_directory, file_extension):
        list_of_files = glob.glob(prune_directory + "/*" + file_extension)

        while len(list_of_files) > ModelStorage.MAX_FILES:
            oldest_file = min(list_of_files, key=os.path.getctime)
            os.remove(oldest_file)
            list_of_files = glob.glob(prune_directory + "/*." + file_extension)
