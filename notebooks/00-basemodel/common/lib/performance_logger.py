import os
from datetime import datetime

import numpy as np


class PerformanceLogger:

    def log_parameters(directory, environment_id, batch_size, learning_rate, gamma, eps_start, eps_end, eps_decay,
                       num_atoms, vmin, vmax, target_update_rate, model_save_rate, episode_log_rate,
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
                       reward_spaceinvaders_player_avoids_line_of_fire,
                       reward_freeway_distance_walked,
                       reward_freeway_distance_to_car,
                       reward_potential_based):
        # Make path if not yet exists
        if not os.path.exists(directory):
            os.mkdir(directory)

        line = "ENVIRONMENT_ID=" + str(environment_id) \
               + "\nBATCH_SIZE=" + str(batch_size) \
               + "\nLEARNING_RATE=" + str(learning_rate) \
               + "\nGAMMA=" + str(gamma) \
               + "\nEPS_START=" + str(eps_start) \
               + "\nEPS_END=" + str(eps_end) \
               + "\nEPS_DECAY=" + str(eps_decay) \
               + "\nNUM_ATOMS=" + str(num_atoms) \
               + "\nVMIN=" + str(vmin) \
               + "\nVMAX=" + str(vmax) \
               + "\nTARGET_UPDATE_RATE=" + str(target_update_rate) \
               + "\nMODEL_SAVE_RATE=" + str(model_save_rate) \
               + "\nEPISODE_LOG_RATE=" + str(episode_log_rate) \
               + "\nREPLAY_MEMORY_SIZE=" + str(replay_memory_size) \
               + "\nNUM_FRAMES=" + str(num_frames) \
               + "\nREWARD_PONG_PLAYER_RACKET_HITS_BALL=" + str(reward_pong_player_racket_hits_ball) \
               + "\nREWARD_PONG_PLAYER_RACKET_COVERS_BALL=" + str(reward_pong_player_racket_covers_ball) \
               + "\nREWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR=" + str(
            reward_pong_player_racket_close_to_ball_linear) \
               + "\nREWARD_PONG_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC=" + str(
            reward_pong_player_racket_close_to_ball_quadratic) \
               + "\nREWARD_PONG_OPPONENT_RACKET_HITS_BALL=" + str(reward_pong_opponent_racket_hits_ball) \
               + "\nREWARD_PONG_OPPONENT_RACKET_COVERS_BALL=" + str(reward_pong_opponent_racket_covers_ball) \
               + "\nREWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_LINEAR=" + str(
            reward_pong_opponent_racket_close_to_ball_linear) \
               + "\nREWARD_PONG_OPPONENT_RACKET_CLOSE_TO_BALL_QUADRATIC=" + str(
            reward_pong_opponent_racket_close_to_ball_quadratic) \
               + "\nREWARD_BREAKOUT_PLAYER_RACKET_HITS_BALL=" + str(reward_breakout_player_racket_hits_ball) \
               + "\nREWARD_BREAKOUT_PLAYER_RACKET_COVERS_BALL=" + str(reward_breakout_player_racket_covers_ball) \
               + "\nREWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_LINEAR=" + str(
            reward_breakout_player_racket_close_to_ball_linear) \
               + "\nREWARD_BREAKOUT_PLAYER_RACKET_CLOSE_TO_BALL_QUADRATIC=" + str(
            reward_breakout_player_racket_close_to_ball_quadratic) \
               + "\nREWARD_SPACEINVADERS_PLAYER_AVOIDS_LINE_OF_FIRE=" + str(
            reward_spaceinvaders_player_avoids_line_of_fire) \
               + "\nREWARD_FREEWAY_DISTANCE_WALKED=" + str(reward_freeway_distance_walked) \
               + "\nREWARD_FREEWAY_DISTANCE_TO_CAR=" + str(reward_freeway_distance_to_car) \
               + "\nREWARD_POTENTIAL_BASED=" + str(reward_potential_based)

        # Print log
        print(line)

        # Write log into file
        log_file = open(directory + "/parameters.txt", "a")
        log_file.write(line + "\n")
        log_file.close()

    def log_episode(directory, max_frames, total_episodes, total_frames, total_duration, total_original_rewards,
                    total_shaped_rewards, episode_frames, episode_original_reward,
                    episode_shaped_reward, episode_loss, episode_duration):
        # Make path if not yet exists
        if not os.path.exists(directory):
            os.mkdir(directory)

        avg_frames_per_minute = total_frames / (total_duration / 60)
        # avg_episodes_per_minute = total_episodes / (total_duration / 60)
        avg_original_reward_per_episode = np.mean(total_original_rewards[-50:])
        avg_shaped_reward_per_episode = np.mean(total_shaped_rewards[-50:])

        line = (datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + " {:8d}".format(total_frames) + "/" + str(max_frames) + "f"
                + " {:5d}".format(total_episodes) + "e"
                # + " {: 5d}".format(episode_frames) + "f"
                # + " {: 4d}".format(round(episode_duration)) + "s"
                + " {:4d}".format(round(avg_frames_per_minute)) + "f/min"
                + "     "
                + " r={:1f}".format(round(episode_original_reward, 2))
                + " rs={:3f}".format(round(episode_shaped_reward, 2))
                + " avgr={:3f}".format(round(avg_original_reward_per_episode, 2))
                + " avgrs={:3f}".format(round(avg_shaped_reward_per_episode, 2))
                + " l=" + str(round(episode_loss, 4)))

        # Print log
        print(line)

        # Write log into file
        log_file = open(directory + "/log.txt", "a")
        log_file.write(line + "\n")
        log_file.close()

        csv = str(total_frames) + "," \
              + str(episode_duration) + "," \
              + str(episode_original_reward) + "," \
              + str(episode_loss) + "," \
              + str(episode_shaped_reward) + "," \
 \
            # Write csv into file
        log_file = open(directory + "/log.csv", "a")
        log_file.write(csv + "\n")
        log_file.close()

    def log_final(self):
        pass
