import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from playground.utils.metrics import divergence_metrics_plus
from playground.utils.plotting import (
    make_position_stats_heatmap,
    make_transition_event_heatmap,
    plot_samples,
)


def _extract_remdm_tracking_logs(traj):
    first = traj["first_unmask_step"].float()
    first[first < 0] = float(traj["x_history"].shape[0] - 1)

    final = traj["finalization_step"].float()
    remasks = traj["num_remasks"].float()
    unmasks = traj["num_unmasks"].float()

    mask_hist = traj["mask_history"].float()
    masked_fraction_by_step = mask_hist.mean(dim=(1, 2)).cpu().numpy()

    logs = {}

    mean_first = first.mean(dim=0).cpu().numpy()
    mean_final = final.mean(dim=0).cpu().numpy()
    mean_remasks = remasks.mean(dim=0).cpu().numpy()
    mean_unmasks = unmasks.mean(dim=0).cpu().numpy()

    for i, v in enumerate(mean_first):
        logs[f"traj/first_unmask_pos_{i}"] = float(v)
    for i, v in enumerate(mean_final):
        logs[f"traj/finalization_pos_{i}"] = float(v)
    for i, v in enumerate(mean_remasks):
        logs[f"traj/remasks_pos_{i}"] = float(v)
    for i, v in enumerate(mean_unmasks):
        logs[f"traj/unmasks_pos_{i}"] = float(v)
    for i, v in enumerate(masked_fraction_by_step):
        logs[f"traj/masked_fraction_step_{i}"] = float(v)

    logs["traj/mean_first_unmask"] = float(first.mean().item())
    logs["traj/mean_finalization"] = float(final.mean().item())
    logs["traj/mean_remasks"] = float(remasks.mean().item())
    logs["traj/mean_unmasks"] = float(unmasks.mean().item())

    positions = torch.arange(first.shape[1], device=first.device).float()
    mean_first_t = first.mean(dim=0)
    mean_final_t = final.mean(dim=0)

    if first.shape[1] > 1:
        logs["traj/corr_position_first_unmask"] = float(
            torch.corrcoef(torch.stack([positions, mean_first_t]))[0, 1].item()
        )
        logs["traj/corr_position_finalization"] = float(
            torch.corrcoef(torch.stack([positions, mean_final_t]))[0, 1].item()
        )

    return logs


def run_eval(model, ds, cfg, L, device):
    with torch.no_grad():
        toks = model.sample(int(cfg.eval.n_samples), L, device)
        toks_np = toks.detach().cpu().numpy()
        gen_xy = np.stack([ds.decode(seq) for seq in toks_np], axis=0)
        mets = divergence_metrics_plus(ds, gen_xy)

    log_dict = {f"eval/{k}": v for k, v in mets.items()}

    wandb_enabled = bool(cfg.get("wandb", {}).get("enabled", True))
    fig = plot_samples(ds, gen_xy)
    if wandb_enabled:
        log_dict["samples_plot"] = wandb.Image(fig)
    plt.close(fig)

    if hasattr(model, "sample_tracked"):
        n_track = min(512, int(cfg.eval.n_samples))
        with torch.no_grad():
            traj = model.sample_tracked(n_track, L, device)
        log_dict.update(_extract_remdm_tracking_logs(traj))

        if wandb_enabled:
            pos_fig = make_position_stats_heatmap(traj)
            log_dict["traj/position_stats_heatmap"] = wandb.Image(pos_fig)
            plt.close(pos_fig)

            evt_fig = make_transition_event_heatmap(
                traj,
                sample_idx=0,
                mask_token=getattr(model, "mask_token", 2),
            )
            log_dict["traj/transition_events_sample0"] = wandb.Image(evt_fig)
            plt.close(evt_fig)

    return log_dict
