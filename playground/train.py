import os
import sys

# Ensure absolute imports resolve to this checkout when run as a script
# (e.g., `python playground/train.py` from sweep agents).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import trange

from playground.data.factory import build_dataset
from playground.training.eval import run_eval
from playground.training.optim import build_optimizer
from playground.training.wandb_utils import setup_wandb, wandb_log
from playground.utils.plotting import plot_samples


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _resolve_model_cfg(model_cfg):
    resolved = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))

    if resolved.get("d_hid") is None:
        d_hid_factor = int(resolved.get("d_hid_factor", 4))
        resolved.d_hid = int(resolved.d_model) * d_hid_factor

    if "d_hid_factor" in resolved:
        del resolved["d_hid_factor"]

    return resolved


def _sample_true_xy(ds, n_samples):
    it = iter(ds)
    seqs = []
    for _ in range(n_samples):
        seq = next(it)
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().numpy()
        seqs.append(seq)
    return np.stack([ds.decode(seq) for seq in seqs], axis=0)


def _model_tag(model_cfg):
    target = str(model_cfg.get("_target_", "model"))
    model_name = target.split(".")[-1].lower()
    d_model = int(model_cfg.get("d_model", 0))
    n_layers = int(model_cfg.get("n_layers", 0))
    d_hid = int(model_cfg.get("d_hid", 0))
    return f"{model_name}_d{d_model}_l{n_layers}_h{d_hid}"


def _resolve_save_path(cfg, model_cfg):
    raw = str(cfg.train.get("save_path", "auto"))

    # Backward-compat for literal sweep placeholders.
    run_id = os.getenv("WANDB_RUN_ID")
    if run_id is None and wandb.run is not None:
        run_id = wandb.run.id
    if run_id:
        raw = raw.replace("${envvar:WANDB_RUN_ID}", run_id)
        raw = raw.replace("WANDB_RUN_ID", run_id)

    if raw.lower() in {"", "auto", "none", "null"}:
        tag = _model_tag(model_cfg)
        if run_id:
            tag = f"{tag}_{run_id}"
        return os.path.join("ckpts", f"{tag}.pt")

    return raw


def _log_true_distribution(cfg, ds, enabled):
    if not enabled or not bool(cfg.eval.get("log_true_distribution", False)):
        return

    n_samples = int(cfg.eval.true_n_samples)
    true_xy = _sample_true_xy(ds, n_samples)
    fig = plot_samples(ds, true_xy)
    wandb_log({"true_distribution_plot": wandb.Image(fig)}, step=0, enabled=enabled)

    import matplotlib.pyplot as plt

    plt.close(fig)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device)

    ds = build_dataset(cfg.data, device)
    L = 2 * int(cfg.data.n_bits)
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size))

    wandb_enabled = setup_wandb(cfg)

    model_cfg = _resolve_model_cfg(cfg.model)
    if bool(cfg.get("debug", False)):
        print(
            f"[debug] model dims: d_model={int(model_cfg.d_model)} "
            f"d_hid={int(model_cfg.d_hid)} n_layers={int(model_cfg.n_layers)}"
        )

    model = instantiate(model_cfg, max_len=L, device=device).to(device)
    n_params = int(count_params(model))
    print(f"[info] trainable params: {n_params:,}")

    if wandb_enabled:
        wandb.config.update({"model/params": n_params}, allow_val_change=True)
    wandb_log({"model/params": n_params}, step=0, enabled=wandb_enabled)

    opt = build_optimizer(cfg.optimizer, model)
    save_path = _resolve_save_path(cfg, model_cfg)
    print(f"[info] checkpoint path: {save_path}")

    _log_true_distribution(cfg, ds, wandb_enabled)

    it = iter(loader)
    model.train()
    for step in trange(int(cfg.train.steps)):
        x = next(it).to(device).long()
        loss = model.training_loss(x)

        opt.zero_grad()
        loss.backward()

        grad_clip_norm = cfg.train.get("grad_clip_norm")
        if grad_clip_norm is not None:
            clip_grad_norm_(model.parameters(), float(grad_clip_norm))

        opt.step()

        if (step % int(cfg.log.every)) == 0:
            wandb_log({"loss": float(loss)}, step=step, enabled=wandb_enabled)

        if ((step + 1) % int(cfg.eval.interval)) == 0 and step > 0:
            model.eval()
            log_dict = run_eval(model, ds, cfg, L, device)
            wandb_log(log_dict, step=step, enabled=wandb_enabled)
            model.train()

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)},
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
