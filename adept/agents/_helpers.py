import torch


def obs_to_device(obs, device):
    """
    :param obs: An OrderedDict observation batch with keys mapping to different components of the observation.
    :param device: torch.device to move the tensors to
    :return:
    """
    batch = []
    for channels in zip(*obs.values()):
        tensor = torch.cat([channel.to(device).float() for channel in channels if isinstance(channel, torch.Tensor)])
        batch.append(tensor)
    return torch.stack(batch)
