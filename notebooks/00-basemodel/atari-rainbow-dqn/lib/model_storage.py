import os

import torch


class ModelStorage:

    def saveModel(directory, total_frames, total_episodes, net, optimizer, memory, loss, environment_name,
                  # environment_wrappers,
                  batch_size, gamma, num_atoms, vmin, vmax, target_update, replay_memory_size, num_frames,
                  # reward_shapings
                  ):
        """
        Saves model into a file
        """

        path = "./model/" + directory

        # Make path if not yet exists
        if not os.path.exists(path):
            os.mkdir(path)

        torch.save({
            'total_frames': total_frames,
            'total_episodes': total_episodes,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'replay_memory': memory,
            'loss': loss,
            'environment_name': environment_name,
            # 'environment_wrappers': environment_wrappers,
            'batch_size': batch_size,
            'gamma': gamma,
            'num_atoms': num_atoms,
            'vmin': vmin,
            'vmax': vmax,
            'target_update': target_update,
            'replay_memory_size': replay_memory_size,
            'num_frames': num_frames,
            # 'reward_shapings': reward_shapings
        }, path + "/target_net-frame-{:07d}".format(total_frames) + ".model")
