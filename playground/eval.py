import os, json
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import torch
import numpy as np

from playground.data.gmm_binary_stream import GMMBinaryStream
from playground.data.checkerboard import CheckerboardBinaryStream
from playground.utils.metrics import divergence_metrics_plus


def _build_dataset(cfg: DictConfig, device: torch.device):
    if cfg.data.dataset == "checkerboard":
        return CheckerboardBinaryStream(
            n_cells=getattr(cfg.data, "n_cells", 8),
            R=float(getattr(cfg.data, "R", 8.0)),
            n_bits=int(cfg.data.n_bits),
            interleave=bool(getattr(cfg.data, "interleave", False)),
            reverse=bool(getattr(cfg.data, "reverse", False)),
            p_white=float(getattr(cfg.data, "p_white", 0.0)),
            device=str(device),
        )
    elif cfg.data.dataset == "gmm":
        return GMMBinaryStream(
            n_mixes=int(getattr(cfg.data, "n_mixes", 8)),
            R=getattr(cfg.data, "R", None),
            log_var_scal=float(getattr(cfg.data, "log_var_scal", 0.0)),
            n_bits=int(cfg.data.n_bits),
            interleave=bool(getattr(cfg.data, "interleave", False)),
            reverse=bool(getattr(cfg.data, "reverse", False)),
            device=str(device),
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset}")


def _get_model_cfg_from_ckpt(ckpt, fallback_cfg_model):
    if isinstance(ckpt, dict):
        c = ckpt.get("cfg", None)
        if isinstance(c, dict) and "model" in c:
            return c["model"]
    return fallback_cfg_model


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device(cfg.device)
    L = 2 * int(cfg.data.n_bits)

    ckpt = torch.load(cfg.eval.ckpt_path, map_location=device)

    mcfg = _get_model_cfg_from_ckpt(ckpt, cfg.model)

    model = instantiate(mcfg, max_len=L, device=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    ds = _build_dataset(cfg, device)

    with torch.no_grad():
        toks = model.sample(int(cfg.eval.n_samples), L, device)

    toks_np = toks.detach().cpu().numpy()
    gen_xy = np.stack([ds.decode(seq) for seq in toks_np], axis=0)

    mets = divergence_metrics_plus(
        ds,
        gen_xy,
        n_true=int(getattr(cfg.eval, "n_true", 60_000)),
        grid=int(getattr(cfg.eval, "grid", 200)),
        alpha=float(getattr(cfg.eval, "alpha", 0.0)),
        two_sample_max_n=int(getattr(cfg.eval, "two_sample_max_n", 10_000)),
        rng_seed=int(getattr(cfg, "seed", 0)),
    )

    os.makedirs(os.path.dirname(cfg.eval.out_json), exist_ok=True)
    with open(cfg.eval.out_json, "w") as f:
        json.dump(mets, f, indent=2)

    print("Saved metrics to", cfg.eval.out_json)
    print(json.dumps(mets, indent=2))


if __name__ == "__main__":
    main()
