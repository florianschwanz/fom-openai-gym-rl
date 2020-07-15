import glob
import os
from zipfile import ZipFile

import torch


class ModelStorage:
    # Maximum number of files we want to store
    MAX_FILES = 1

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

        if checkpoint != None:
            return checkpoint.get('net_state_dict', None)
        else:
            return None

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

        return checkpoint.get('optimizer_state_dict', None)

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
            chunks.append(checkpoint.get('replay_memory_chunk', None))

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

        return checkpoint.get('environment', None), \
               checkpoint.get('environment_wrappers', None)

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

        return checkpoint.get('batch_size', 32), \
               checkpoint.get('learning_rate', 0.0001), \
               checkpoint.get('gamma', 0.99), \
               checkpoint.get('eps_start', 1.0), \
               checkpoint.get('eps_end', 0.01), \
               checkpoint.get('eps_decay', 10_000), \
               checkpoint.get('num_atoms', 51), \
               checkpoint.get('vmin', -10), \
               checkpoint.get('vmax', 10), \
               checkpoint.get('eta', 0.0), \
               checkpoint.get('beta', 0.0), \
               checkpoint.get('lambda1', 0.0), \
               checkpoint.get('normalize_shaped_reward', False), \
               checkpoint.get('reward_shaping_dropout_rate', 0.0), \
               checkpoint.get('target_update_rate', 10), \
               checkpoint.get('model_save_rate', 10), \
               checkpoint.get('episode_log_rate', 10), \
               checkpoint.get('replay_memory_size', 100_000), \
               checkpoint.get('num_frames', 1_000_000)

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
                    reward_breakout_ball_hitting_upper_block,
                    reward_space_invaders_player_avoids_line_of_fire,
                    reward_freeway_distance_walked,
                    reward_freeway_distance_to_car,
                    reward_ms_pacman_far_from_enemy,
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
            'reward_breakout_ball_hitting_upper_block': reward_breakout_ball_hitting_upper_block,
            'reward_space_invaders_player_avoids_line_of_fire': reward_space_invaders_player_avoids_line_of_fire,
            'reward_freeway_distance_walked': reward_freeway_distance_walked,
            'reward_freeway_distance_to_car': reward_freeway_distance_to_car,
            'reward_ms_pacman_far_from_enemy': reward_ms_pacman_far_from_enemy,
            'reward_potential_based': reward_potential_based,
        }, target_directory + "/frame-{:07d}".format(total_frames) + file_extension)

        ModelStorage.prune_storage(target_directory, file_extension)

    def loadRewards(output_directory, run_directory):
        file_extension = ModelStorage.FILE_EXTENTION_REWARDS
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)
        checkpoint = ModelStorage.load_checkpoint(target_directory, file_extension)

        return checkpoint.get('reward_pong_player_racket_hits_ball', 0.0), \
               checkpoint.get('reward_pong_player_racket_covers_ball', 0.0), \
               checkpoint.get('reward_pong_player_racket_close_to_ball_linear', 0.0), \
               checkpoint.get('reward_pong_player_racket_close_to_ball_quadratic', 0.0), \
               checkpoint.get('reward_pong_opponent_racket_hits_ball', 0.0), \
               checkpoint.get('reward_pong_opponent_racket_covers_ball', 0.0), \
               checkpoint.get('reward_pong_opponent_racket_close_to_ball_linear', 0.0), \
               checkpoint.get('reward_pong_opponent_racket_close_to_ball_quadratic', 0.0), \
               checkpoint.get('reward_breakout_player_racket_hits_ball', 0.0), \
               checkpoint.get('reward_breakout_player_racket_covers_ball', 0.0), \
               checkpoint.get('reward_breakout_player_racket_close_to_ball_linear', 0.0), \
               checkpoint.get('reward_breakout_player_racket_close_to_ball_quadratic', 0.0), \
               checkpoint.get('reward_breakout_ball_hitting_upper_block', 0.0), \
               checkpoint.get('reward_space_invaders_player_avoids_line_of_fire', 0.0), \
               checkpoint.get('reward_freeway_distance_walked', 0.0), \
               checkpoint.get('reward_freeway_distance_to_car', 0.0), \
               checkpoint.get('reward_ms_pacman_far_from_enemy', 0.0), \
               checkpoint.get('reward_potential_based', 0.0)

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

        return checkpoint.get('total_frames', 0), \
               checkpoint.get('total_episodes', 0), \
               checkpoint.get('total_original_rewards', 0), \
               checkpoint.get('total_shaped_rewards', 0), \
               checkpoint.get('total_losses', 0)

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

        if len(list_of_files) > 0:
            file_to_load = max(list_of_files, key=os.path.getctime)
            return torch.load(file_to_load)
        else:
            print("No file to load with extension " + file_extension)
            return None

    def load_checkpoint_file(file_to_load):
        return torch.load(file_to_load)

    def zip_model(output_directory, run_directory, total_frames):

        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)

        zip_file = ZipFile(target_directory + "/frame-{:07d}".format(total_frames) + ".zip", 'w')

        list_of_files = glob.glob(target_directory + "/*.pickle")
        for file in list_of_files:
            zip_file.write(file)

        zip_file.close()

        # Prune unzipped pickle files
        ModelStorage.prune_storage(target_directory, "pickle", 0)
        # Prune zip files
        ModelStorage.prune_storage(target_directory, "zip", 2)

    def unzip_model(output_directory, run_directory):
        target_directory = ModelStorage.prepare_directory(output_directory, run_directory)

        list_of_zip_files = glob.glob(target_directory + "/*.zip")

        if len(list_of_zip_files) > 0:
            # Get latest zip file
            file_to_unzip = max(list_of_zip_files, key=os.path.getctime)

            # Extract zip file
            with ZipFile(file_to_unzip, 'r') as zip_file:
                zip_file.extractall()

    def prune_storage(prune_directory, file_extension, max_files=MAX_FILES):
        list_of_files = glob.glob(prune_directory + "/*" + file_extension)

        while len(list_of_files) > max_files:
            oldest_file = min(list_of_files, key=os.path.getctime)
            os.remove(oldest_file)
            list_of_files = glob.glob(prune_directory + "/*." + file_extension)
