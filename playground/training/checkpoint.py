import torch
from omegaconf import OmegaConf


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format at {path}: expected dict.")
    if "model" not in ckpt:
        raise ValueError(f"Invalid checkpoint at {path}: missing 'model' key.")
    return ckpt


def restore_from_checkpoint(model, optimizer, ckpt, strict=True, load_optimizer=True):
    model.load_state_dict(ckpt["model"], strict=bool(strict))

    optimizer_loaded = False
    if optimizer is not None and load_optimizer and isinstance(ckpt.get("optimizer"), dict):
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer_loaded = True

    start_step = int(ckpt.get("step", 0))
    return {"start_step": start_step, "optimizer_loaded": optimizer_loaded}


def save_checkpoint(path, model, cfg, optimizer=None, step=None):
    payload = {
        "model": model.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if step is not None:
        payload["step"] = int(step)
    torch.save(payload, path)
