import torch


class ModelStorage:

    def saveModel(total_frames, net, optimizer, loss, environment_name, environment_wrappers, batch_size, gamma,
                  eps_start, eps_end, eps_decay, target_update, replay_memory_size, num_frames, reward_shapings):
        torch.save({
            'total_frames': total_frames,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'environment_name': environment_name, 'environment_wrappers': environment_wrappers,
            'batch_size': batch_size, 'gamma': gamma,
            'eps_start': eps_start, 'eps_end': eps_end, 'eps_decay': eps_decay, 'target_update': target_update,
            'replay_memory_size': replay_memory_size,
            'num_frames': num_frames,
            'reward_shapings': reward_shapings
        }, "./model/target_net-frame-{:07d}".format(total_frames) + ".model")
