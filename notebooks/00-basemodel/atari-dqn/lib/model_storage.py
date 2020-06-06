import torch


class ModelStorage:

    def saveModel(total_frames, total_episodes, net, optimizer, memory, loss, environment_name, environment_wrappers,
                  batch_size, gamma, eps_start, eps_end, eps_decay, target_update, replay_memory_size, num_frames,
                  reward_shapings):
        """
        Saves model into a file
        """
        torch.save({
            'total_frames': total_frames,
            'total_episodes': total_episodes,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'replay_memory': memory,
            'loss': loss,
            'environment_name': environment_name, 'environment_wrappers': environment_wrappers,
            'batch_size': batch_size, 'gamma': gamma,
            'eps_start': eps_start, 'eps_end': eps_end, 'eps_decay': eps_decay, 'target_update': target_update,
            'replay_memory_size': replay_memory_size,
            'num_frames': num_frames,
            'reward_shapings': reward_shapings
        }, "./model/target_net-frame-{:07d}".format(total_frames) + ".model")

    def loadModel(path):
        """
        Loads model from a given path
        :param path path to saved model
        :return:
        """
        checkpoint = torch.load(path)

        return checkpoint['total_frames'], \
               checkpoint['total_episodes'], \
               checkpoint['model_state_dict'], \
               checkpoint['optimizer_state_dict'], \
               checkpoint['replay_memory'], \
               checkpoint['loss'], \
               checkpoint['environment_name'], \
               checkpoint['environment_wrappers'], \
               checkpoint['batch_size'], \
               checkpoint['gamma'], \
               checkpoint['eps_start'], \
               checkpoint['eps_end'], \
               checkpoint['eps_decay'], \
               checkpoint['target_update'], \
               checkpoint['replay_memory_size'], \
               checkpoint['num_frames'], \
               checkpoint['reward_shapings']
