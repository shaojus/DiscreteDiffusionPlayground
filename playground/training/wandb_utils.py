from omegaconf import OmegaConf

import wandb


def setup_wandb(cfg):
    enabled = bool(cfg.get("wandb", {}).get("enabled", True))
    if not enabled:
        return False

    init_kwargs = dict(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    if cfg.wandb.get("group") is not None:
        init_kwargs["group"] = cfg.wandb.group
    wandb.init(**init_kwargs)
    return True


def wandb_log(log_dict, step, enabled):
    if not enabled:
        return
    wandb.log(log_dict, step=step)
