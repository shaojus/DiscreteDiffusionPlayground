from hydra.utils import instantiate


def build_optimizer(optimizer_cfg, model):
    return instantiate(optimizer_cfg, params=model.parameters())
