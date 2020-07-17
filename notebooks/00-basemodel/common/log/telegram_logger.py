import glob
import os

import numpy as np
import telegram_send


class TelegramLogger:

    TELEGRAM_FILE_LIMIT_MB = 50

    def log_parameters(run_name, output_directory, run_directory, conf_directory, conf_file, environment_id, batch_size,
                       learning_rate, gamma, eps_start, eps_end, eps_decay, num_atoms, vmin, vmax, eta, beta, lambda1,
                       normalize_shaped_reward, reward_shaping_dropout_rate,
                       target_update_rate, model_save_rate, episode_log_rate,
                       replay_memory_size, num_frames,
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
                       reward_potential_based):
        if conf_file == None:
            return

        telegram_line = "<b>run " + run_name + "</b>" \
                        + "\n" \
                        + "\nenvironment id " + str(environment_id) \
                        + "\nbatch size " + str(batch_size) \
                        + "\nlearning_rate " + str(learning_rate) \
                        + "\ngamma " + str(gamma) \
                        + "\neps start " + str(eps_start) \
                        + "\neps end " + str(eps_end) \
                        + "\neps decay " + str(eps_decay) \
                        + "\nnum atoms " + str(num_atoms) \
                        + "\nvmin " + str(vmin) \
                        + "\nvmax " + str(vmax) \
                        + "\neta " + str(eta) \
                        + "\nbeta " + str(beta) \
                        + "\nlambda1 " + str(lambda1) \
                        + "\nnormalize shaped reward " + str(normalize_shaped_reward) \
                        + "\nreward shaping dropout rate " + str(reward_shaping_dropout_rate) \
                        + "\ntarget update rate " + str(target_update_rate) \
                        + "\nsave model rate " + str(model_save_rate) \
                        + "\nepisode log rate " + str(episode_log_rate) \
                        + "\nreplay memory size " + str(replay_memory_size) \
                        + "\nnum frames " + str(num_frames) \
                        + "\n" \
                        + TelegramLogger.build_reward_parameter("player racket hits ball",
                                                                reward_pong_player_racket_hits_ball) \
                        + TelegramLogger.build_reward_parameter("player racket covers ball",
                                                                reward_pong_player_racket_covers_ball) \
                        + TelegramLogger.build_reward_parameter("player racket close to ball linear",
                                                                reward_pong_player_racket_close_to_ball_linear) \
                        + TelegramLogger.build_reward_parameter("player racket close to ball quadratic",
                                                                reward_pong_player_racket_close_to_ball_quadratic) \
                        + TelegramLogger.build_reward_parameter("opponent racket hits ball",
                                                                reward_pong_opponent_racket_hits_ball) \
                        + TelegramLogger.build_reward_parameter("opponent racket covers ball",
                                                                reward_pong_opponent_racket_covers_ball) \
                        + TelegramLogger.build_reward_parameter("opponent racket close to ball linear",
                                                                reward_pong_opponent_racket_close_to_ball_linear) \
                        + TelegramLogger.build_reward_parameter("opponent racket close to ball quadratic",
                                                                reward_pong_opponent_racket_close_to_ball_quadratic) \
                        + TelegramLogger.build_reward_parameter("player racket hits ball",
                                                                reward_breakout_player_racket_hits_ball) \
                        + TelegramLogger.build_reward_parameter("player racket covers ball",
                                                                reward_breakout_player_racket_covers_ball) \
                        + TelegramLogger.build_reward_parameter("player racket close to ball linear",
                                                                reward_breakout_player_racket_close_to_ball_linear) \
                        + TelegramLogger.build_reward_parameter("player racket close to ball quadratic",
                                                                reward_breakout_player_racket_close_to_ball_quadratic) \
                        + TelegramLogger.build_reward_parameter("ball hitting upper block",
                                                                reward_breakout_ball_hitting_upper_block) \
                        + TelegramLogger.build_reward_parameter("player avoids line of fire",
                                                                reward_space_invaders_player_avoids_line_of_fire) \
                        + TelegramLogger.build_reward_parameter("chicken distance walked",
                                                                reward_freeway_distance_walked) \
                        + TelegramLogger.build_reward_parameter("chicken distance to car",
                                                                reward_freeway_distance_to_car) \
                        + TelegramLogger.build_reward_parameter("far from enemy", reward_ms_pacman_far_from_enemy) \
                        + TelegramLogger.build_reward_parameter("potential based", reward_potential_based)

        # Get config path
        list_of_configs = glob.glob(conf_directory + "/" + conf_file)
        config_path = max(list_of_configs, key=os.path.getctime)

        # Send line to telegram
        telegram_send.send(messages=[telegram_line], parse_mode="html", conf=config_path)

    def build_reward_parameter(name, value):
        return "\n" + str(name) + " " + str(value) if value != 0.0 else ""

    def log_episode(run_name, output_directory, run_directory, conf_directory, conf_file, max_frames, total_episodes,
                    total_frames,
                    total_duration, total_original_rewards, total_shaped_rewards, episode_frames,
                    episode_original_reward, episode_shaped_reward, episode_loss, episode_duration):
        if conf_file == None:
            return

        target_directory = output_directory + "/" + run_directory

        avg_original_reward_per_episode = np.mean(total_original_rewards[-50:])
        avg_shaped_reward_per_episode = np.mean(total_shaped_rewards[-50:])

        # Assemble line
        telegram_line = "<b>run " + run_name + "</b>\n" \
                        + "\nframes {:8d}".format(total_frames) + "/" + str(max_frames) \
                        + "\nepisode {:5d}".format(total_episodes) \
                        + "\nepisode reward " + str(round(episode_original_reward, 2)) \
                        + " / shaped " + str(round(episode_shaped_reward, 2)) \
                        + "\naverage reward " + str(round(avg_original_reward_per_episode, 2)) \
                        + " / shaped " + str(round(avg_shaped_reward_per_episode, 2)) \
                        + "\nloss " + str(round(episode_loss, 4))

        # Get animation path
        list_of_files = glob.glob(target_directory + "/*.gif")
        gif_path = max(list_of_files, key=os.path.getctime)
        # Get config path
        list_of_configs = glob.glob(conf_directory + "/" + conf_file)
        config_path = max(list_of_configs, key=os.path.getctime)

        # Send line to telegram
        if os.path.getsize(gif_path) /(1024*1024) < TelegramLogger.TELEGRAM_FILE_LIMIT_MB:
           with open(gif_path, "rb") as f:
               telegram_send.send(messages=[telegram_line], animations=[f], parse_mode="html", conf=config_path)
        else:
            telegram_send.send(messages=[telegram_line], parse_mode="html", conf=config_path)

    def log_results(run_name, output_directory, run_directory, conf_directory, conf_file):
        target_directory = output_directory + "/" + run_directory

        # Retrieve file globs
        log_csv_path_glob = glob.glob(target_directory + "/log.csv")
        log_txt_path_glob = glob.glob(target_directory + "/log.txt")
        parameters_txt_path_glob = glob.glob(target_directory + "/parameters.txt")
        losses_png_path_glob = glob.glob(target_directory + "/*losses.png")
        original_rewards_png_path_glob = glob.glob(target_directory + "/*original-rewards.png")
        shaped_rewards_png_path_glob = glob.glob(target_directory + "/*shaped-rewards.png")

        if len(log_csv_path_glob) > 0 \
                and len(log_txt_path_glob) > 0 \
                and len(parameters_txt_path_glob) > 0 \
                and len(losses_png_path_glob) > 0 \
                and len(original_rewards_png_path_glob) > 0 \
                and len(shaped_rewards_png_path_glob) > 0:
            # Get result paths
            log_csv_path = max(log_csv_path_glob, key=os.path.getctime)
            log_txt_path = max(log_txt_path_glob, key=os.path.getctime)
            parameters_txt_path = max(parameters_txt_path_glob, key=os.path.getctime)
            losses_png_path = max(losses_png_path_glob, key=os.path.getctime)
            original_rewards_png_path = max(original_rewards_png_path_glob, key=os.path.getctime)
            shaped_rewards_png_path = max(shaped_rewards_png_path_glob, key=os.path.getctime)
            finish_flag_png_path = target_directory + "/../../../common/images/2560px-F1_chequered_flag.svg.png"

            # Get config path
            list_of_configs = glob.glob(conf_directory + "/" + conf_file)
            config_path = max(list_of_configs, key=os.path.getctime)

            telegram_line = "<b>run " + run_name + "</b> completed!"

            # Send line to telegram
            with open(log_csv_path, "rb") as log_csv, \
                    open(log_txt_path, "rb") as log_txt, \
                    open(parameters_txt_path, "rb") as parameters_txt, \
                    open(losses_png_path, "rb") as losses_png, \
                    open(original_rewards_png_path, "rb") as original_rewards_png, \
                    open(shaped_rewards_png_path, "rb") as shaped_rewards_png, \
                    open(finish_flag_png_path, "rb") as finish_flag_png:
                telegram_send.send(messages=[telegram_line],
                                   files=[log_csv, log_txt, parameters_txt],
                                   images=[losses_png, original_rewards_png, shaped_rewards_png, finish_flag_png],
                                   parse_mode="html",
                                   conf=config_path)
