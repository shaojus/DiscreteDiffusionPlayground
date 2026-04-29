from hydra.utils import instantiate


def build_dataset(data_cfg, device):
    return instantiate(data_cfg, device=str(device))
